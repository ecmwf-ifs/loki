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


__all__ = ['remove_block_loops']


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
