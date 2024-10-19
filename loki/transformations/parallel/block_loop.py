# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation utilities to remove and generate parallel block loops.
"""

from loki.expression import symbols as sym, parse_expr
from loki.ir import (
    nodes as ir, FindNodes, Transformer, pragma_regions_attached,
    is_loki_pragma
)
from loki.scope import SymbolAttributes
from loki.types import BasicType


__all__ = ['remove_block_loops', 'add_block_loops']


def remove_block_loops(routine):
    """
    Remove any outer block :any:`Loop` from a given :any:`Subroutine.
    """

    class RemoveBlockLoopTransformer(Transformer):
        """
        :any:`Transformer` to remove driver-level block loops.
        """

        def visit_Loop(self, loop, **kwargs):  # pylint: disable=unused-argument
            if not loop.variable == 'JKGLO':
                return loop

            to_remove = tuple(
                a for a in FindNodes(ir.Assignment).visit(loop.body)
                if a.lhs in ['ICST', 'ICEND', 'IBL']
            )
            return tuple(n for n in loop.body if n not in to_remove)

    routine.body = RemoveBlockLoopTransformer().visit(routine.body)


def add_block_loops(routine, dimension):
    """
    Insert IFS-style driver block-loops (NPROMA).
    """

    # TODO: The abuse of `Dimension` here includes some back-bending
    # hackery due to the funcky way in which the block-loop bounds are
    # done in IFS!

    # Ensure that local integer variables are declared
    index = parse_expr(dimension.index, routine)
    upper = parse_expr(dimension.bounds_expressions[1][1], routine)
    bidx = parse_expr(dimension.index_expressions[1], routine)
    for v in (index, upper, bidx):
        if not v in routine.variable_map:
            routine.variables += (
                v.clone(type=SymbolAttributes(BasicType.INTEGER, kind='JPIM')),
            )

    def _create_block_loop(body, scope):
        """
        Generate block loop object, including indexing preamble
        """

        # This is a hack; it's meant to be the upper limit, but we use it as stride!
        bsize = parse_expr(dimension.bounds_expressions[1][0], scope=scope)
        size = parse_expr(dimension.size, scope=scope)
        lrange = sym.LoopRange((sym.Literal(1), size, bsize))

        expr_tail = parse_expr(f'{size}-{index}+1', scope=scope)
        expr_max = sym.InlineCall(
            function=sym.ProcedureSymbol('MIN', scope=scope), parameters=(bsize, expr_tail)
        )
        preamble = (ir.Assignment(lhs=upper, rhs=expr_max),)
        preamble += (ir.Assignment(
            lhs=bidx, rhs=parse_expr(f'({index}-1)/{bsize}+1', scope=scope)
        ),)

        return ir.Loop(variable=index, bounds=lrange, body=preamble + body)

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
