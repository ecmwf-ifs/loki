# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import (
    Sum, Product, IntLiteral, Scalar, Array, RangeIndex, 
    DeferredTypeSymbol, SubstituteExpressions
    )
from loki.ir import CallStatement
from loki.visitors import FindNodes
from loki.tools import as_tuple


__all__ = [
    'fix_scalar_syntax'
]

def check_if_scalar_syntax(arg, dummy):
    if isinstance(arg, Array) and isinstance(dummy, Array):
        if arg.dimensions:
            n_dummy_ranges = sum(1 for d in arg.dimensions if isinstance(d, RangeIndex))
            if n_dummy_ranges == 0:
                return True
    return False


def construct_range_index(lower, length):

    if lower == IntLiteral(1):
        new_high = length
    elif isinstance(lower, IntLiteral) and isinstance(length, IntLiteral):
        new_high = IntLiteral(value = length.value + lower.value - 1)
    elif isinstance(lower, IntLiteral):
        new_high = Sum((length,IntLiteral(value = lower.value - 1)))
    elif isinstance(length, IntLiteral):
        new_high = Sum((lower,IntLiteral(value = length.value - 1)))
    else:
        new_high = Sum((lower, length, Product((-1, IntLiteral(1)))))

    return RangeIndex((lower, new_high))


def merge_parents(parent, symbol):

    new_parent = parent.clone()
    for p in symbol.parents[1:]:
        new_parent = DeferredTypeSymbol(name=p.name_parts[-1], scope=parent.scope, parent=new_parent)
    return symbol.clone(parent=new_parent, scope=parent.scope)


def process_symbol(symbol, caller, call):

    if isinstance(symbol, IntLiteral):
        return symbol

    elif isinstance(symbol, Scalar):
        if symbol in call.routine.arguments:
            return call.arg_map[symbol]

    elif isinstance(symbol, DeferredTypeSymbol):
        if symbol.parents[0] in call.routine.arguments:
            return merge_parents(call.arg_map[symbol.parents[0]], symbol)

    if call.routine in caller.members and symbol in caller.variables:
        return symbol

    raise RuntimeError('[Loki::fix_scalar_syntax] Unable to resolve argument dimension. Module variable?')


def construct_length(xrange, routine, call):

    new_start = process_symbol(xrange.start, routine, call)
    new_stop  = process_symbol(xrange.stop, routine, call)

    if isinstance(new_start, IntLiteral) and isinstance(new_stop, IntLiteral):
        return IntLiteral(value = new_stop.value - new_start.value + 1)
    elif isinstance(new_start, IntLiteral):
        return Sum((new_stop, Product((-1,(IntLiteral(value = new_start.value - 1))))))
    elif isinstance(new_stop, IntLiteral):
        return Sum((IntLiteral(value = new_stop.value + 1), Product((-1,new_start))))
    else:
        return Sum((new_stop, Product((-1,new_start)), IntLiteral(1)))


def fix_scalar_syntax(routine):
    """
    Housekeeping routine to replace scalar syntax when passing arrays as arguments
    For example, a call like

    call myroutine(a(i,j))

    where myroutine looks like

    subroutine myroutine(a)
        real :: a(5)
    end subroutine myroutine

    should be changed to

    call myroutine(a(i:i+5,j)


    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine where calls will be changed
    """

    calls = FindNodes(CallStatement).visit(routine.body)

    for call in calls:

        new_arg_map = {}

        for dummy, arg in call.arg_map.items():
            if check_if_scalar_syntax(arg, dummy):

                new_dims = []
                for s, lower in zip(dummy.shape, arg.dimensions):

                    if isinstance(s, RangeIndex):
                        new_dims += [construct_range_index(lower, construct_length(s, routine, call))]
                    else:
                        new_dims += [construct_range_index(lower, process_symbol(s, routine, call))]

                if len(arg.dimensions) > len(dummy.shape):
                    new_dims += [d for d in arg.dimensions[len(dummy.shape):]]

                new_dims = as_tuple(new_dims)
                new_arg = arg.clone(dimensions=new_dims)

                print(arg, new_arg)
                new_arg_map[arg] = new_arg

    routine.body = SubstituteExpressions(new_arg_map).visit(routine.body)

