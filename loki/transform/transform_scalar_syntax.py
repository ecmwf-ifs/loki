# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.expression import (
    Sum, Product, IntLiteral, Scalar, Array, RangeIndex, DeferredTypeSymbol
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

    #Define one and minus one for later

    one = IntLiteral(1)
    minus_one = Product((-1, IntLiteral(1)))

    if lower == one:
        new_high = length
    elif isinstance(lower, IntLiteral) and isinstance(length, IntLiteral):
        new_high = IntLiteral(value = length.value + lower.value - 1)
    elif isinstance(lower, IntLiteral):
        new_high = Sum((length,IntLiteral(value = lower.value - 1)))
    elif isinstance(length, IntLiteral):
        new_high = Sum((lower,IntLiteral(value = length.value - 1)))
    else:
        new_high = Sum((lower, length, minus_one))
        
    return RangeIndex((lower, new_high))


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
                print(routine)
                print(call)
                print(arg, dummy)
                new_dims = []
                for s, lower in zip(dummy.shape, arg.dimensions):
                    
                    if isinstance(s, IntLiteral):
                        new_dims += [construct_range_index(lower, s)]

                    elif isinstance(s, Scalar):
                        if s in call.routine.arguments:
                            new_dims += [construct_range_index(lower,call.arg_map[s])]
                        elif call.routine in routine.members and s in routine.variables:
                            new_dims += [construct_range_index(lower,s)]
                        else:
                            raise RuntimeError('[Loki::fix_scalar_syntax] Unable to resolve argument dimension. Module variable?')

                    elif isinstance(s, DeferredTypeSymbol):

                        if s.parents[0] in call.routine.arguments:
                            print(s, s.parents[0], s.parents[0].scope)
                            print(call.arg_map[s.parents[0]])
                            print()


                if len(arg.dimensions) > len(dummy.shape):
                    new_dims += [d for d in arg.dimensions[len(dummy.shape):]]

                new_dims = as_tuple(new_dims)
                new_arg = arg.clone(dimensions=new_dims)
                print('new_arg: ', new_arg)
                print()







