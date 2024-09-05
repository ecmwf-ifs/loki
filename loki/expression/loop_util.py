# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pymbolic.primitives as pmbl
from loki.expression import symbols as sym, simplify, Simplification, negate

__all__ = [
    'iteration_number', 'iteration_index'
]


"""
Utility functions that can be used to simplify working with loops and the
Loki LoopRange object
"""

def iteration_number(iter_idx, loop_range: sym.LoopRange) -> pmbl.Expression:
    """
    Returns the normalized iteration number of the iteration variable

    Given the loop iteration index for an iteration in a loop defined by the
    :any:´sym.LoopRange´ this method returns the normalized iteration index given by
    iter_num = (iter_idx - start + step)/step = (iter_idx-start)/step + 1

    Parameters
    ----------
    iter_idx : :any:`Variable`, :any:`Expression`, or :any:`IntLiteral`
        corresponding to a valid iteration index for the parameter `loop_range`
    loop_range: :any:`LoopRange`
    """
    if loop_range.step is None:
        expr = sym.Sum((sym.Sum((iter_idx, negate(loop_range.start))), sym.IntLiteral(1)))

    else:
        expr = sym.Sum(
            (sym.Quotient(sym.Sum((iter_idx, negate(loop_range.start))), loop_range.step),
             sym.IntLiteral(1)))
    return simplify(expr, enabled_simplifications=Simplification.IntegerArithmetic)


def iteration_index(iter_num, loop_range: sym.LoopRange) -> pmbl.Expression:
    """
    Returns the iteration index of the loop based on the iteration number

    Given the normalized iteration number for an iteration in a loop defined by the
    :any:´sym.LoopRange´ this method returns the iteration index given by
    iter_idx = (iter_num-1)*step+start

    Parameters
    ----------
    iter_num : :any:`Variable`, :any:`Expression`, or :any:`IntLiteral`
        corresponding to a valid iteration number for the parameter `loop_range`
    loop_range: :any:`LoopRange`
    """
    if loop_range.step is None:
        expr = sym.Sum((iter_num, sym.IntLiteral(-1), loop_range.start))

    else:
        expr = sym.Sum((sym.Product((sym.Sum((iter_num, sym.IntLiteral(-1))), loop_range.step)),
                        loop_range.start))
    return simplify(expr, enabled_simplifications=Simplification.IntegerArithmetic)
