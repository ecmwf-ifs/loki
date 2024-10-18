# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation utilities to remove and generate parallel block loops.
"""

from loki.ir import nodes as ir, FindNodes, Transformer
from loki.tools import as_tuple


__all__ = ['remove_block_loops']


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
