# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to guard dangerous floating-point operands so that
unconditional evaluation becomes safe.

**Conditional guarding** (always active):
This is a prerequisite for :any:`ConditionalFPGuardToMerge` (T1).
Denominators/arguments in **both** the true-branch and the
else-branch of conditionals inside horizontal loops are clamped with
``MAX``/``MIN`` so that the conditional body can be evaluated on all
SIMD lanes without hardware exceptions.

For the true-branch a *causal link* between the condition and the
dangerous operand is required (the condition must reference the same
variable).  For the else-branch all dangerous operations are clamped
unconditionally, because the condition provides no safety guarantee
for the else-branch.

**Unconditional guarding** (opt-in via ``guard_unconditional=True``):
After statement-function inlining, dangerous operations (EXP with
division arguments, divisions by near-zero values, etc.) may appear
as direct assignments in horizontal loop bodies — outside any
conditional.  When ``guard_unconditional`` is enabled, a second pass
scans every ``Assignment`` node directly in horizontal loop bodies
(skipping nodes already inside conditionals) and applies the same
safe clamps.  This is needed for ``cpui-*`` pipelines where inlining
exposes operations that were previously hidden inside statement
functions like ``FOEEWM``, ``FOKOOP``, ``FOEDEM``.
"""

from loki.batch import Transformation
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer,
    FindInlineCalls, SubstituteExpressionsSkipLHS
)
from loki.expression import (
    symbols as sym, Quotient, Comparison, FloatLiteral,
    ExpressionRetriever
)
from loki.types import ProcedureType, SymbolAttributes
from loki.tools import as_tuple

from loki.transformations.utilities import check_routine_sequential


__all__ = ['SafeDenominatorGuard']


class SafeDenominatorGuard(Transformation):
    """
    Guard dangerous floating-point operands so that unconditional
    evaluation (via ``MERGE``) or hardware-strict FP modes become safe.

    **Conditional guarding** (always active): For each conditional
    inside a horizontal vector loop, the transformation detects:

    * Division where the denominator is guarded by a ``> 0`` check
    * ``SQRT``/``LOG`` calls where the argument is guarded against
      negative/zero
    * ``EXP`` calls where the argument is guarded against overflow

    The dangerous operand is wrapped with a safe clamp:

    * Division: ``denom`` -> ``MAX(denom, TINY(1.0))``
    * SQRT: ``arg`` -> ``MAX(arg, 0.0)``
    * LOG: ``arg`` -> ``MAX(arg, TINY(1.0))``
    * EXP: ``arg`` -> ``MIN(arg, 500.0)``

    The true-branch requires a *causal link* (the condition must
    reference the same variable as the dangerous operand).  The
    else-branch is clamped unconditionally — after ``MERGE`` conversion
    both branches are evaluated on every SIMD lane, and the condition
    provides no safety guarantee for the else-branch.

    **Unconditional guarding** (opt-in via *guard_unconditional*):
    When enabled, a second pass scans ``Assignment`` nodes that sit
    directly in horizontal loop bodies (i.e. **not** inside any
    ``Conditional``) and applies the same safe clamps without requiring
    a causal link.  This is needed after statement-function inlining
    (``cpui-*`` pipelines), which exposes ``EXP``, division, ``SQRT``,
    and ``LOG`` operations that were previously hidden inside statement
    functions (``FOEEWM``, ``FOKOOP``, ``FOEDEM``, etc.).

    After this transformation, :any:`ConditionalFPGuardToMerge` can
    safely convert the conditional to a ``MERGE`` call.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the horizontal data dimension
        and iteration space.
    guard_unconditional : bool, optional
        If ``True``, also guard dangerous operations in unconditional
        assignments directly inside horizontal loop bodies.  Default
        is ``False`` (only guard inside conditionals, preserving
        backward-compatible behaviour).
    """

    #: Dangerous intrinsic functions and their safe wrappers.
    #: Each entry maps: function_name -> (wrapper_func, bound_expr_builder)
    DANGEROUS_INTRINSICS = frozenset({'SQRT', 'LOG', 'EXP'})

    def __init__(self, horizontal, guard_unconditional=False):
        self.horizontal = horizontal
        self.guard_unconditional = guard_unconditional

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role', 'kernel')
        if role == 'driver':
            return

        if check_routine_sequential(routine):
            return

        cond_map = {}
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if not self._is_horizontal_loop(loop):
                continue
            # Only process top-level conditionals in the loop body.
            # Nested conditionals (ELSEIF chains) are handled
            # recursively inside _guard_dangerous_ops.
            for node in loop.body:
                if isinstance(node, ir.Conditional):
                    new_cond = self._guard_dangerous_ops(node, scope=routine)
                    if new_cond is not None:
                        cond_map[node] = new_cond

        if cond_map:
            routine.body = Transformer(cond_map).visit(routine.body)

        # ----------------------------------------------------------
        # Second pass: guard unconditional assignments in loop bodies
        # ----------------------------------------------------------
        # After statement-function inlining, dangerous ops (EXP with
        # division argument, 1/X**2, etc.) may sit directly in the
        # horizontal loop body as plain Assignment nodes — outside any
        # Conditional.  The first pass above only touches Conditionals,
        # so these are missed.  When guard_unconditional is enabled we
        # scan every direct Assignment child of each horizontal loop
        # and apply the same safe clamps unconditionally (no causal
        # link needed — there is no condition to link against).
        if self.guard_unconditional:
            assign_map = {}
            for loop in FindNodes(ir.Loop).visit(routine.body):
                if not self._is_horizontal_loop(loop):
                    continue
                for node in loop.body:
                    if not isinstance(node, ir.Assignment):
                        continue
                    dangerous_ops = self._find_dangerous_ops((node,))
                    for op_type, operand in dangerous_ops:
                        safe_expr = self._build_safe_wrapper(
                            op_type, operand, scope=routine
                        )
                        if safe_expr is not None:
                            assign_map[operand] = safe_expr

            if assign_map:
                routine.body = SubstituteExpressionsSkipLHS(
                    assign_map
                ).visit(routine.body)

    def _is_horizontal_loop(self, loop):
        """Check if a loop iterates over the horizontal dimension."""
        return loop.variable.name.lower() in [
            idx.lower() for idx in self.horizontal.indices
        ]

    def _guard_dangerous_ops(self, cond, scope=None):
        """
        Analyse a conditional and wrap dangerous operands with safe clamps.

        Both the true-branch (``body``) and the else-branch
        (``else_body``) are inspected.  For the true-branch a *causal
        link* between the condition and the dangerous operand is
        required (the condition must reference the same variable).  For
        the else-branch **all** dangerous operations are clamped
        unconditionally, because the condition provides no safety
        guarantee for the else-branch — after ``MERGE`` conversion both
        branches are evaluated on every SIMD lane.

        Returns a new :any:`Conditional` with the guarded branches, or
        ``None`` if no dangerous ops were found in either branch.
        """
        # ----------------------------------------------------------
        # True branch — requires causal link
        # ----------------------------------------------------------
        guarded_vars = self._extract_guarded_variables(cond.condition)

        body_map = {}
        if guarded_vars:
            dangerous_ops = self._find_dangerous_ops(cond.body)
            for op_type, operand in dangerous_ops:
                operand_vars = FindVariables().visit(operand)
                operand_var_names = {v.name.lower() for v in operand_vars}

                linked = False
                for gvar_name, _guard_type in guarded_vars:
                    if gvar_name in operand_var_names:
                        linked = True
                        break

                if not linked:
                    continue

                safe_expr = self._build_safe_wrapper(op_type, operand, scope=scope)
                if safe_expr is not None:
                    body_map[operand] = safe_expr

        # ----------------------------------------------------------
        # Else branch — no causal link required
        # ----------------------------------------------------------
        else_map = {}
        new_else_body = cond.else_body
        if cond.else_body:
            # Recursively process nested Conditionals (ELSEIF chains)
            else_list = list(cond.else_body)
            for i, node in enumerate(else_list):
                if isinstance(node, ir.Conditional):
                    inner = self._guard_dangerous_ops(node, scope=scope)
                    if inner is not None:
                        else_list[i] = inner
            new_else_body = as_tuple(else_list)

            # Guard dangerous ops in Assignment nodes of the else body
            else_dangerous_ops = self._find_dangerous_ops(new_else_body)
            for op_type, operand in else_dangerous_ops:
                safe_expr = self._build_safe_wrapper(op_type, operand, scope=scope)
                if safe_expr is not None:
                    else_map[operand] = safe_expr

        # Check whether any inner conditional was recursively updated
        else_changed = (new_else_body is not cond.else_body)

        if not body_map and not else_map and not else_changed:
            return None

        # Apply substitution only to the RHS of assignments (and full
        # non-assignment nodes), never to the LHS.  This prevents turning
        # ``X(JL) = expr`` into ``MAX(X(JL), TINY(1.0)) = expr`` when
        # X(JL) also appears as a denominator on the RHS elsewhere.
        new_body = cond.body
        if body_map:
            new_body = SubstituteExpressionsSkipLHS(body_map).visit(cond.body)

        if else_map:
            new_else_body = SubstituteExpressionsSkipLHS(else_map).visit(new_else_body)

        return cond.clone(body=as_tuple(new_body), else_body=as_tuple(new_else_body))

    def _extract_guarded_variables(self, condition):
        """
        Extract variable names guarded by a comparison in the condition.

        Returns a list of ``(var_name_lower, guard_type)`` tuples where
        ``guard_type`` is one of ``'positive'``, ``'nonzero'``,
        ``'bounded'``.
        """
        retriever = ExpressionRetriever(lambda e: isinstance(e, Comparison))
        comparisons = retriever.retrieve(condition)

        guarded = []
        for comp in comparisons:
            left = comp.left
            right = comp.right
            op = comp.operator

            # Pattern: VAR > 0, VAR >= 0, VAR /= 0, VAR > EPSILON, etc.
            if op in ('>', '>='):
                # Left side is the guarded variable
                left_vars = FindVariables().visit(left)
                for v in left_vars:
                    guarded.append((v.name.lower(), 'positive'))
            elif op in ('<', '<='):
                # Right side is the guarded variable (reversed comparison)
                right_vars = FindVariables().visit(right)
                for v in right_vars:
                    guarded.append((v.name.lower(), 'bounded'))
                # Also: the left side might be bounded from above
                left_vars = FindVariables().visit(left)
                for v in left_vars:
                    guarded.append((v.name.lower(), 'bounded'))
            elif op == '!=':
                left_vars = FindVariables().visit(left)
                for v in left_vars:
                    guarded.append((v.name.lower(), 'nonzero'))

        return guarded

    def _find_dangerous_ops(self, body):
        """
        Find dangerous floating-point operations in the body.

        Returns a list of ``(op_type, operand)`` tuples where:

        * ``op_type`` is ``'DIVISION'``, ``'SQRT'``, ``'LOG'``, or ``'EXP'``
        * ``operand`` is the expression node that needs guarding
        """
        ops = []

        # Find divisions (Quotient nodes)
        quot_retriever = ExpressionRetriever(lambda e: isinstance(e, Quotient))
        for assign in body:
            if isinstance(assign, ir.Assignment):
                quotients = quot_retriever.retrieve(assign.rhs)
                for q in quotients:
                    ops.append(('DIVISION', q.denominator))

        # Find dangerous intrinsic calls
        for assign in body:
            if isinstance(assign, ir.Assignment):
                calls = FindInlineCalls().visit(assign.rhs)
                for call in calls:
                    if call.name.upper() in self.DANGEROUS_INTRINSICS:
                        if call.parameters:
                            ops.append((call.name.upper(), call.parameters[0]))

        return ops

    def _build_safe_wrapper(self, op_type, operand, scope=None):
        """
        Build a safe wrapper expression for a dangerous operand.

        Parameters
        ----------
        op_type : str
            The type of dangerous operation (``'DIVISION'``, ``'SQRT'``,
            ``'LOG'``, ``'EXP'``).
        operand : Expression
            The expression to wrap.

        Returns
        -------
        :any:`InlineCall` or ``None``
            The safe wrapper expression.
        """
        # ptype = ProcedureType(name='TINY', is_function=True, is_intrinsic=True)
        dtype = ProcedureType('TINY', is_function=True, is_intrinsic=True)
        ptype = SymbolAttributes(dtype=dtype) 
        tiny_call = sym.InlineCall(
            function=sym.ProcedureSymbol('TINY', scope=scope, type=ptype), # type=ptype.clone(name='TINY')),
            parameters=(FloatLiteral(1.0),)
        )
        
        if op_type == 'DIVISION':
            # MAX(denom, TINY(1.0))
            return sym.InlineCall(
                function=sym.ProcedureSymbol('MAX', scope=scope, type=ptype), # type=ptype.clone(name='MAX')),
                parameters=(operand, tiny_call)
            )
        elif op_type == 'SQRT':
            # MAX(arg, 0.0)
            return sym.InlineCall(
                function=sym.ProcedureSymbol('MAX', scope=scope, type=ptype), # type=ptype.clone(name='TINY')),
                parameters=(operand, FloatLiteral(0.0))
            )
        elif op_type == 'LOG':
            # MAX(arg, TINY(1.0))
            return sym.InlineCall(
                function=sym.ProcedureSymbol('MAX', scope=scope, type=ptype), # type=ptype.clone(name='TINY')),
                parameters=(operand, tiny_call)
            )
        elif op_type == 'EXP':
            # MIN(arg, 500.0)
            return sym.InlineCall(
                function=sym.ProcedureSymbol('MIN', scope=scope, type=ptype), # type=ptype.clone(name='TINY')),
                parameters=(operand, FloatLiteral(500.0))
            )

        return None
