# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to rewrite ``MERGE(true_val, false_val, mask)`` into
explicit arithmetic masking so that compilers can vectorise under
``-fp-model strict``.

Under strict floating-point semantics the compiler cannot speculate on
the value of ``MERGE`` arguments, so it serialises the conditional
branch.  By converting the selection into an arithmetic expression
using an **integer** ``MERGE`` (which vectorises freely), we regain
SIMD throughput without relaxing the FP model.

The rewrite is::

    result = false_val * (1.0_JPRB - REAL(MERGE(1, 0, mask), JPRB))
           + true_val  * REAL(MERGE(1, 0, mask), JPRB)

Optionally, each branch operand can be clamped with
``MIN(MAX(expr, -HUGE(1.0)), HUGE(1.0))`` to guarantee finiteness
and avoid ``Inf * 0.0 = NaN`` at the cost of extra instructions.

This transformation (T1b) is designed to run **after**
:any:`ConditionalFPGuardToMerge` (T1) and **before**
:any:`SplitLoopForVectorisation` (T7) in the CPU vectorisation
pipeline.
"""

from loki.batch import Transformation
from loki.ir import (
    nodes as ir, FindNodes
)
from loki.ir.expr_visitors import ExpressionTransformer
from loki.expression import symbols as sym
from loki.expression.mappers import LokiIdentityMapper
from loki.types import ProcedureType, SymbolAttributes
from loki.tools import as_tuple

from loki.transformations.utilities import check_routine_sequential


__all__ = ['MergeToExplicitMask']


# -----------------------------------------------------------------
# Expression mapper: bottom-up MERGE rewriting
# -----------------------------------------------------------------

class _MergeToArithmeticMapper(LokiIdentityMapper):
    """
    Bottom-up expression mapper that replaces every
    ``MERGE(true_val, false_val, mask)`` with an arithmetic masking
    expression using an integer ``MERGE(1, 0, mask)``.

    Parameters
    ----------
    kind : expression node or None
        The kind symbol (e.g. ``JPRB``) for the ``REAL`` cast and the
        ``1.0`` literal.  ``None`` means plain ``REAL`` / ``1.0``.
    clamp_results : bool
        When *True*, wrap each branch operand with
        ``MIN(MAX(expr, -HUGE(1.0)), HUGE(1.0))`` to bound infinities.
    """

    def __init__(self, kind=None, clamp_results=False):
        super().__init__()
        self.kind = kind
        self.clamp_results = clamp_results

    # -- helpers --------------------------------------------------

    @staticmethod
    def _make_int_merge(mask):
        """Build ``MERGE(1, 0, mask)`` as an integer InlineCall."""
        merge_sym = sym.ProcedureSymbol(
            'MERGE',
            type=SymbolAttributes(
                ProcedureType(name='MERGE', is_function=True,
                              is_intrinsic=True)
            )
        )
        return sym.InlineCall(
            function=merge_sym,
            parameters=as_tuple([
                sym.IntLiteral(1),
                sym.IntLiteral(0),
                mask,
            ])
        )

    def _make_real_cast(self, int_merge):
        """Build ``REAL(int_merge [, kind=K])``."""
        return sym.Cast('REAL', int_merge, kind=self.kind)

    def _make_one(self):
        """Build ``1.0`` (or ``1.0_JPRB``) as a FloatLiteral."""
        return sym.FloatLiteral(1.0, kind=self.kind)

    def _clamp(self, expr):
        """Wrap *expr* in ``MIN(MAX(expr, -HUGE(1.0)), HUGE(1.0))``."""
        if not self.clamp_results:
            return expr

        huge_sym = sym.ProcedureSymbol(
            'HUGE',
            type=SymbolAttributes(
                ProcedureType(name='HUGE', is_function=True,
                              is_intrinsic=True)
            )
        )
        huge_call = sym.InlineCall(
            function=huge_sym,
            parameters=as_tuple([self._make_one()])
        )
        neg_huge = sym.Product((-1, huge_call))

        max_sym = sym.ProcedureSymbol(
            'MAX',
            type=SymbolAttributes(
                ProcedureType(name='MAX', is_function=True,
                              is_intrinsic=True)
            )
        )
        max_call = sym.InlineCall(
            function=max_sym,
            parameters=as_tuple([expr, neg_huge])
        )

        min_sym = sym.ProcedureSymbol(
            'MIN',
            type=SymbolAttributes(
                ProcedureType(name='MIN', is_function=True,
                              is_intrinsic=True)
            )
        )
        return sym.InlineCall(
            function=min_sym,
            parameters=as_tuple([max_call, huge_call])
        )

    # -- main entry point: override map_inline_call ---------------

    def map_inline_call(self, expr, *args, **kwargs):
        # First recurse into children (handles nested MERGEs bottom-up)
        new_expr = super().map_inline_call(expr, *args, **kwargs)

        # Check if this is a MERGE call with 3 parameters
        if not (hasattr(new_expr, 'function') and
                new_expr.function.name.upper() == 'MERGE' and
                len(new_expr.parameters) == 3):
            return new_expr

        true_val = new_expr.parameters[0]
        false_val = new_expr.parameters[1]
        mask = new_expr.parameters[2]

        # Build the integer MERGE and REAL cast
        int_merge = self._make_int_merge(mask)
        real_mask = self._make_real_cast(int_merge)
        one = self._make_one()

        # Optionally clamp branch operands
        true_val = self._clamp(true_val)
        false_val = self._clamp(false_val)

        # Build: false_val * (1.0 - real_mask) + true_val * real_mask
        one_minus_mask = sym.Sum((one, sym.Product((-1, real_mask))))
        false_term = sym.Product((false_val, one_minus_mask))
        true_term = sym.Product((true_val, real_mask))

        return sym.Sum((false_term, true_term))


# -----------------------------------------------------------------
# IR-level transformer
# -----------------------------------------------------------------

class _MergeRewriteTransformer(ExpressionTransformer):
    """
    Applies :class:`_MergeToArithmeticMapper` to every expression
    node in the IR sub-tree.
    """

    def __init__(self, kind=None, clamp_results=False, **kwargs):
        super().__init__(**kwargs)
        self.expr_mapper = _MergeToArithmeticMapper(
            kind=kind, clamp_results=clamp_results
        )


# -----------------------------------------------------------------
# Transformation class
# -----------------------------------------------------------------

class MergeToExplicitMask(Transformation):
    """
    Rewrite ``MERGE(true_val, false_val, mask)`` calls inside
    horizontal vector loops into explicit arithmetic masking
    expressions that vectorise under ``-fp-model strict``.

    The rewrite replaces each ``MERGE`` with::

        false_val * (1.0_K - REAL(MERGE(1, 0, mask), K))
        + true_val * REAL(MERGE(1, 0, mask), K)

    where ``K`` is the kind of the assignment LHS (e.g. ``JPRB``).

    Parameters
    ----------
    horizontal : :any:`Dimension`
        Dimension object describing the horizontal data dimension.
    fp_strict : bool, optional
        When *False* (the default) the transformation is a no-op.
        Set to *True* to enable the MERGE rewriting.
    clamp_results : bool, optional
        When *True*, each branch operand is wrapped with
        ``MIN(MAX(expr, -HUGE(1.0)), HUGE(1.0))`` to guarantee
        finiteness and avoid ``Inf * 0.0 = NaN``.  Default *False*
        accepts the residual risk.
    """

    # Process in reverse order so that kernels are handled before
    # their callers (standard Loki convention for kernel transforms).
    reverse_traversal = True

    def __init__(self, horizontal, fp_strict=False, clamp_results=False):
        self.horizontal = horizontal
        self.fp_strict = fp_strict
        self.clamp_results = clamp_results

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs.get('role', 'kernel')
        if role == 'driver':
            return

        if not self.fp_strict:
            return

        if check_routine_sequential(routine):
            return

        horizontal = self.horizontal

        # Iterate over horizontal loops
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if loop.variable != horizontal.index:
                continue

            # Process each assignment in this loop that contains MERGE
            self._rewrite_merges_in_loop(loop, routine)

    def _rewrite_merges_in_loop(self, loop, routine):
        """
        For each :any:`Assignment` inside *loop* whose RHS contains a
        ``MERGE`` call, apply the arithmetic rewrite.

        The kind for the ``REAL`` cast is determined from the
        assignment LHS type.
        """
        from loki.ir import FindInlineCalls  # local import to match style

        assignments = FindNodes(ir.Assignment).visit(loop.body)
        assign_map = {}

        for assign in assignments:
            # Quick check: does this RHS contain any MERGE calls?
            calls = FindInlineCalls().visit(assign.rhs)
            has_merge = any(c.name.upper() == 'MERGE' for c in calls)
            if not has_merge:
                continue

            # Determine kind from LHS type
            kind = None
            if hasattr(assign.lhs, 'type') and assign.lhs.type is not None:
                kind = assign.lhs.type.kind

            # Apply the mapper to the RHS expression only
            mapper = _MergeToArithmeticMapper(kind=kind,
                                              clamp_results=self.clamp_results)
            new_rhs = mapper(assign.rhs)

            if new_rhs is not assign.rhs:
                assign_map[assign] = assign.clone(rhs=new_rhs)

        if assign_map:
            from loki.ir import Transformer  # local import
            routine.body = Transformer(assign_map).visit(routine.body)
