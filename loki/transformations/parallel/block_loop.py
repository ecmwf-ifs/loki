# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation utilities to remove and generate parallel block loops.
"""

from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, Transformer, pragma_regions_attached,
    is_loki_pragma
)
from loki.scope import SymbolAttributes
from loki.tools import as_tuple
from loki.types import BasicType


__all__ = ['remove_block_loops', 'add_block_loops']


def remove_block_loops(routine, dimension):
    """
    Remove any outer block :any:`Loop` from a given :any:`Subroutine.

    The loops are identified accoerding to a given :any:`Dimension`
    object, and will remove auxiliary assignments of index and bound
    variables, as commongly used in IFS-style block loops.

    Parameters
    ----------
    routine: :any:`Subroutine`
        Subroutine from which to remove block loops
    dimension : :any:`Dimension`
        The dimension object describing loop variables
    """
    idx = dimension.index
    variables = as_tuple(dimension.indices)
    variables += as_tuple(dimension.lower)
    variables += as_tuple(dimension.upper)

    class RemoveBlockLoopTransformer(Transformer):
        """
        :any:`Transformer` to remove driver-level block loops.
        """

        def visit_Loop(self, loop, **kwargs):  # pylint: disable=unused-argument
            if not loop.variable == idx:
                return loop

            to_remove = tuple(
                a for a in FindNodes(ir.Assignment).visit(loop.body)
                if a.lhs in variables
            )
            return tuple(n for n in loop.body if n not in to_remove)

    routine.body = RemoveBlockLoopTransformer().visit(routine.body)


def add_block_loops(routine, dimension, default_type=None):
    """
    Insert IFS-style (NPROMA) driver block-loops in ``!$loki
    parallel`` regions.

    The provided :any:`Dimension` object describes the variables to
    used when generating the loop and default assignments. It
    encapsulates IFS-specific convention, where a strided loop over
    points, defined by ``dimension.index``, ``dimension.bounds`` and
    ``dimension.step`` is created, alongside assignments that define
    the corresponding block index and upper bound, defined by
    ``dimension.indices[1]`` and ``dimension.upper[1]`` repsectively.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The routine in which to add block loops.
    dimension : :any:`Dimension`
        The dimension object describing the block loop variables.
    default_type : :any:`SymbolAttributes`, optional
        Default type to use when creating variables; defaults to
        ``integer(kind=JPIM)``.
    """

    _default = SymbolAttributes(BasicType.INTEGER, kind='JPIM')
    dtype = default_type if default_type else _default

    # TODO: Explain convention in docstring
    lidx = routine.parse_expr(dimension.index)
    bidx = routine.parse_expr(dimension.indices[1])
    bupper = routine.parse_expr(dimension.upper[1])

    # Ensure that local integer variables are declared
    for v in (lidx, bupper, bidx):
        if not v in routine.variable_map:
            routine.variables += (v.clone(type=dtype),)

    def _create_block_loop(body, scope):
        """
        Generate block loop object, including indexing preamble
        """

        bsize = scope.parse_expr(dimension.step)
        lupper = scope.parse_expr(dimension.upper[0])
        lrange = sym.LoopRange((sym.Literal(1), lupper, bsize))

        expr_tail = scope.parse_expr(f'{lupper}-{lidx}+1')
        expr_max = sym.InlineCall(
            function=sym.ProcedureSymbol('MIN', scope=scope), parameters=(bsize, expr_tail)
        )
        preamble = (ir.Assignment(lhs=bupper, rhs=expr_max),)
        preamble += (ir.Assignment(
            lhs=bidx, rhs=scope.parse_expr(f'({lidx}-1)/{bsize}+1')
        ),)

        return ir.Loop(variable=lidx, bounds=lrange, body=preamble + body)

    class InsertBlockLoopTransformer(Transformer):

        def visit_PragmaRegion(self, region, **kwargs):
            """
            (Re-)insert driver-level block loops into marked parallel region.
            """
            if not is_loki_pragma(region.pragma, starts_with='parallel'):
                return region

            scope = kwargs.get('scope')

            loop = _create_block_loop(body=region.body, scope=scope)

            region._update(body=(ir.Comment(''), loop))
            return region

    with pragma_regions_attached(routine):
        routine.body = InsertBlockLoopTransformer().visit(routine.body, scope=routine)
