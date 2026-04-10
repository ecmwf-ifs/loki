# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to convert conditional floating-point guards inside
horizontal vector loops into Fortran ``MERGE`` intrinsic calls.

This addresses the dominant vectorisation blocker (Intel optrpt ``#15326``,
~37 loops across CLOUDSC and CUBASEN) where the compiler refuses to
speculatively execute masked-off FP operations.

Supports:

* Simple ``IF / ELSE / ENDIF`` blocks with pure assignments.
* ``IF / ELSEIF / ... / ELSE / ENDIF`` chains — converted to nested
  ``MERGE`` calls (right-to-left folding).
* Accumulation patterns (``X(JL) = X(JL) + expr``) when
  *allow_accumulations* is enabled (the default).  This is safe when
  there are no loop-carried dependencies across the horizontal
  dimension — a convention that holds for IFS physics.
"""

from loki.batch import Transformation
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer
)
from loki.types import ProcedureType, SymbolAttributes
from loki.expression import symbols as sym
from loki.tools import as_tuple

from loki.transformations.utilities import check_routine_sequential


__all__ = ['ConditionalFPGuardToMerge']


class ConditionalFPGuardToMerge(Transformation):
    """
    Replace conditional-assignment patterns inside horizontal vector
    loops with Fortran ``MERGE`` intrinsic calls.

    ``MERGE`` expresses the conditional as a value-selection rather than
    control-flow branching, which the compiler can map directly to masked
    SIMD instructions.

    Simple IF/ELSE example::

        DO JL = KIDIA, KFDIA
          IF (LLMASK(JL)) THEN
            ZRESULT(JL) = ZARG(JL) * 2.0_JPRB
          END IF
        END DO

    becomes::

        DO JL = KIDIA, KFDIA
          ZRESULT(JL) = MERGE(ZARG(JL) * 2.0_JPRB, ZRESULT(JL), LLMASK(JL))
        END DO

    ELSEIF chain example::

        IF (ZVAL(JL) > 1.0) THEN
          ZA(JL) = ZB(JL)
        ELSE IF (ZVAL(JL) > 0.5) THEN
          ZA(JL) = ZB(JL) * 0.5
        ELSE
          ZA(JL) = 0.0
        END IF

    becomes::

        ZA(JL) = MERGE(ZB(JL), MERGE(ZB(JL)*0.5, 0.0, ZVAL(JL) > 0.5), ZVAL(JL) > 1.0)

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the horizontal data dimension
        and iteration space.
    allow_accumulations : bool, optional
        If *True* (default), accumulation patterns where the LHS appears
        in its own RHS (e.g. ``X(JL) = X(JL) + expr``) are eligible for
        conversion.  This is safe when there are no loop-carried
        dependencies across the horizontal dimension.  Set to *False* to
        restore the conservative behaviour that skips accumulations.
    """

    def __init__(self, horizontal, allow_accumulations=True):
        self.horizontal = horizontal
        self.allow_accumulations = allow_accumulations

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

            # Collect IDs of conditionals that are part of elseif chains
            # (i.e., used as the else_body of another conditional).
            # These will be processed as part of their parent chain.
            elseif_children = set()
            for cond in FindNodes(ir.Conditional).visit(loop.body):
                if cond.has_elseif and cond.else_body:
                    for child in cond.else_body:
                        if isinstance(child, ir.Conditional):
                            elseif_children.add(id(child))

            for cond in FindNodes(ir.Conditional).visit(loop.body):
                # Skip conditionals that are part of an elseif chain
                if id(cond) in elseif_children:
                    continue
                if self._is_convertible(cond, loop):
                    new_stmts = self._build_merge_assignments(cond)
                    if new_stmts:
                        cond_map[cond] = new_stmts

        if cond_map:
            routine.body = Transformer(cond_map).visit(routine.body)

    # -----------------------------------------------------------------
    # Horizontal-loop detection
    # -----------------------------------------------------------------

    def _is_horizontal_loop(self, loop):
        """Check if a loop iterates over the horizontal dimension."""
        return loop.variable.name.lower() in [
            idx.lower() for idx in self.horizontal.indices
        ]

    # -----------------------------------------------------------------
    # Branch validation helpers
    # -----------------------------------------------------------------

    def _is_branch_convertible(self, body_nodes, loop_var):
        """
        Validate that *body_nodes* (a tuple of IR nodes from one branch
        of a conditional) consists solely of :any:`Assignment` nodes
        whose LHS is an array indexed by the horizontal loop variable.

        If *allow_accumulations* is False, assignments where the LHS
        name appears in the RHS are rejected.

        Returns
        -------
        set or None
            The set of lower-cased LHS variable names on success, or
            *None* if the branch is not convertible.
        """
        if not body_nodes:
            return None

        lhs_names = set()
        for node in body_nodes:
            if not isinstance(node, ir.Assignment):
                return None

            # LHS should be an array with horizontal index
            if not isinstance(node.lhs, sym.Array):
                return None
            if not self._expr_references_variable(node.lhs, loop_var):
                return None

            # Accumulation guard (optional)
            if not self.allow_accumulations:
                lhs_name = node.lhs.name.lower()
                rhs_vars = FindVariables().visit(node.rhs)
                if any(v.name.lower() == lhs_name for v in rhs_vars):
                    return None

            lhs_names.add(node.lhs.name.lower())

        return lhs_names

    # -----------------------------------------------------------------
    # ELSEIF chain helpers
    # -----------------------------------------------------------------

    def _flatten_elseif_chain(self, cond, loop_var):
        """
        Walk an ELSEIF chain and return a structured representation.

        Returns
        -------
        (branches, else_body) or None
            *branches* is a list of ``(condition, body_nodes, lhs_names)``
            tuples, one per IF / ELSEIF branch.  *else_body* is the
            tuple of assignment nodes from the final ELSE clause, or
            ``()`` if there is no final ELSE.  Returns *None* if any
            branch fails validation.
        """
        branches = []
        current = cond

        while True:
            lhs_names = self._is_branch_convertible(current.body, loop_var)
            if lhs_names is None:
                return None
            branches.append((current.condition, current.body, lhs_names))

            if current.has_elseif and current.else_body:
                # The else_body should contain a single Conditional node
                # representing the next ELSEIF branch.
                next_cond = None
                for child in current.else_body:
                    if isinstance(child, ir.Conditional):
                        next_cond = child
                        break
                if next_cond is None:
                    return None
                current = next_cond
            else:
                # End of chain — collect final else_body if present
                else_body = current.else_body if current.else_body else ()
                break

        # Validate final else_body if present
        if else_body:
            else_lhs = self._is_branch_convertible(else_body, loop_var)
            if else_lhs is None:
                return None
        else:
            else_lhs = set()

        # All branches (and else) must write to the same set of LHS
        # variables.  Missing variables in a branch get their identity
        # (LHS) as the value — but the *first* branch defines the
        # canonical set (union of all branches).
        all_lhs = set()
        for _, _, names in branches:
            all_lhs |= names
        all_lhs |= else_lhs

        # Validate that *every* condition in the chain references the
        # horizontal loop variable (data-dependent mask).
        for branch_cond, _, _ in branches:
            cond_vars = FindVariables().visit(branch_cond)
            if not any(v.name.lower() == loop_var for v in cond_vars):
                return None

        return branches, else_body, all_lhs

    # -----------------------------------------------------------------
    # Top-level convertibility check
    # -----------------------------------------------------------------

    def _is_convertible(self, cond, loop):
        """
        Check whether a :any:`Conditional` can be converted to ``MERGE``.

        Handles both simple IF/ELSE and ELSEIF chains.
        """
        loop_var = loop.variable.name.lower()

        # --- ELSEIF chains ---
        if cond.has_elseif:
            result = self._flatten_elseif_chain(cond, loop_var)
            return result is not None

        # --- Simple IF / ELSE ---
        body_lhs = self._is_branch_convertible(cond.body, loop_var)
        if body_lhs is None:
            return False

        # Check else_body if present
        if cond.else_body:
            else_lhs = self._is_branch_convertible(cond.else_body, loop_var)
            if else_lhs is None:
                return False
            # Else assignments must target the same variables
            if else_lhs != body_lhs:
                return False

        # Condition must reference the horizontal loop variable
        cond_vars = FindVariables().visit(cond.condition)
        if not any(v.name.lower() == loop_var for v in cond_vars):
            return False

        return True

    # -----------------------------------------------------------------
    # MERGE construction
    # -----------------------------------------------------------------

    def _build_merge_assignments(self, cond):
        """
        Build ``MERGE``-based assignment statements from a convertible
        :any:`Conditional`.

        For ELSEIF chains the MERGE calls are nested right-to-left::

            IF(c1)   v=r1
            ELSEIF(c2) v=r2
            ELSE       v=r3   -->  v = MERGE(r1, MERGE(r2, r3, c2), c1)

        Parameters
        ----------
        cond : :any:`Conditional`
            The conditional node to convert.

        Returns
        -------
        tuple of :any:`Assignment`
            The replacement assignment statements using ``MERGE``.
        """
        if cond.has_elseif:
            return self._build_elseif_merge(cond)
        return self._build_simple_merge(cond)

    def _build_simple_merge(self, cond):
        """Build MERGE assignments for a simple IF / ELSE conditional."""
        # Build a map from LHS name -> else-body RHS for matching
        else_map = {}
        if cond.else_body:
            for assign in cond.else_body:
                if isinstance(assign, ir.Assignment):
                    else_map[assign.lhs.name.lower()] = assign.rhs

        new_assignments = []
        for assign in cond.body:
            lhs = assign.lhs
            rhs = assign.rhs
            lhs_name = lhs.name.lower()

            # Determine the false-branch value
            if lhs_name in else_map:
                false_val = else_map[lhs_name]
            else:
                # No else: retain previous value (LHS itself)
                false_val = lhs.clone()

            merge_call = sym.InlineCall(
                function=sym.ProcedureSymbol('MERGE', type=SymbolAttributes(ProcedureType(name='MERGE', is_function=True, is_intrinsic=True))),
                parameters=as_tuple([rhs, false_val, cond.condition])
            )

            new_assignments.append(ir.Assignment(lhs=lhs, rhs=merge_call))

        return as_tuple(new_assignments)

    def _build_elseif_merge(self, cond):
        """
        Build nested ``MERGE`` assignments for an ELSEIF chain.

        Walks the chain via :meth:`_flatten_elseif_chain` (which has
        already been validated by :meth:`_is_convertible`), then folds
        branches right-to-left into nested ``MERGE`` calls.
        """
        loop_var = None  # not needed again; chain already validated
        # Re-flatten to get structured data (cheap — just walks nodes)
        # We need the loop_var for the helper; extract from condition.
        # Use a dummy loop_var — we know it's valid because
        # _is_convertible already succeeded.  Collect from first branch.
        result = self._flatten_elseif_chain_unchecked(cond)
        branches, else_body, all_lhs = result

        # For each LHS variable, build a right-to-left nested MERGE.
        # Collect the canonical LHS node from the first branch that
        # writes to each variable.
        lhs_nodes = {}  # lhs_name -> lhs ir node
        for _, body_nodes, _ in branches:
            for assign in body_nodes:
                name = assign.lhs.name.lower()
                if name not in lhs_nodes:
                    lhs_nodes[name] = assign.lhs

        # Build RHS maps per branch and for else
        branch_rhs_maps = []
        for branch_cond, body_nodes, _ in branches:
            rhs_map = {}
            for assign in body_nodes:
                rhs_map[assign.lhs.name.lower()] = assign.rhs
            branch_rhs_maps.append((branch_cond, rhs_map))

        else_rhs_map = {}
        if else_body:
            for assign in else_body:
                if isinstance(assign, ir.Assignment):
                    else_rhs_map[assign.lhs.name.lower()] = assign.rhs

        new_assignments = []
        for lhs_name in sorted(all_lhs):
            lhs_node = lhs_nodes[lhs_name]

            # Start with the innermost value (the final else, or identity)
            if lhs_name in else_rhs_map:
                inner = else_rhs_map[lhs_name]
            else:
                inner = lhs_node.clone()

            # Fold right-to-left: iterate branches in reverse
            for branch_cond, rhs_map in reversed(branch_rhs_maps):
                if lhs_name in rhs_map:
                    true_val = rhs_map[lhs_name]
                else:
                    true_val = lhs_node.clone()

                inner = sym.InlineCall(
                    function=sym.ProcedureSymbol('MERGE', type=SymbolAttributes(dtype=ProcedureType(name='MERGE', is_function=True, is_intrinsic=True))),
                    parameters=as_tuple([true_val, inner, branch_cond])
                )

            new_assignments.append(ir.Assignment(lhs=lhs_node, rhs=inner))

        return as_tuple(new_assignments)

    def _flatten_elseif_chain_unchecked(self, cond):
        """
        Walk an ELSEIF chain and return ``(branches, else_body, all_lhs)``
        without validation.  Only called after :meth:`_is_convertible`
        has already validated the chain.
        """
        branches = []
        current = cond
        all_lhs = set()

        while True:
            lhs_names = set()
            for node in current.body:
                if isinstance(node, ir.Assignment):
                    lhs_names.add(node.lhs.name.lower())
            branches.append((current.condition, current.body, lhs_names))
            all_lhs |= lhs_names

            if current.has_elseif and current.else_body:
                next_cond = None
                for child in current.else_body:
                    if isinstance(child, ir.Conditional):
                        next_cond = child
                        break
                if next_cond is None:
                    break
                current = next_cond
            else:
                break

        else_body = current.else_body if current.else_body else ()
        if else_body:
            for node in else_body:
                if isinstance(node, ir.Assignment):
                    all_lhs.add(node.lhs.name.lower())

        return branches, else_body, all_lhs

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _expr_references_variable(expr, var_name):
        """
        Check whether an expression tree references a variable with
        the given name (case-insensitive).
        """
        variables = FindVariables().visit(expr)
        return any(v.name.lower() == var_name.lower() for v in variables)
