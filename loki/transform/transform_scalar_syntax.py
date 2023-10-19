# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pymbolic.primitives as pmbl

from loki.expression import (
    Sum, Product, IntLiteral, Array, RangeIndex,
    SubstituteExpressions
    )
from loki.ir import CallStatement
from loki.visitors import FindNodes, Transformer
from loki.tools import as_tuple
from loki.types import BasicType


__all__ = [
    'fix_scalar_syntax'
]

def check_if_scalar_syntax(arg, dummy):
    """
    Check if an array argument, arg,
    is passed to an array dummy argument, dummy,
    using scalar syntax. i.e. arg(1,1) -> d(m,n)

    Parameters
    ----------
    arg:   variable
    dummy: variable
    """
    if isinstance(arg, Array) and isinstance(dummy, Array):
        if arg.dimensions:
            if not any(isinstance(d, RangeIndex) for d in arg.dimensions):
                return True
    return False


def single_sum(expr):
    """
    Return a Sum object of expr if expr is not an instance of pymbolic.primitives.Sum.
    Otherwise return expr

    Parameters
    ----------
    expr: any pymbolic expression
    """
    if isinstance(expr, pmbl.Sum):
        return expr
    return Sum((expr,))


def product_value(expr):
    """
    If expr is an instance of pymbolic.primitives.Product, try to evaluate it
    If it is possible, return the value as an int.
    If it is not possible, try to simplify the the product and return as a Product
    If it is not a pymbolic.primitives.Product , return expr

    Note: Negative numbers and subtractions in Sums are represented as Product of
          the integer -1 and the symbol. This complicates matters.

    Parameters
    ----------
    expr: any pymbolic expression
    """
    if isinstance(expr, pmbl.Product):
        m = 1
        new_children = []
        for c in expr.children:
            if isinstance(c, IntLiteral):
                m = m*c.value
            elif isinstance(c, int):
                m = m*c
            else:
                new_children += [c]

        if m == 0:
            return 0
        if not new_children:
            return m

        if m > 1:
            m = IntLiteral(m)
        elif m < -1:
            m = Product((-1, IntLiteral(abs(m))))

        return m*Product(as_tuple(new_children))

    return expr


def simplify_sum(expr):
    """
    If expr is an instance of pymbolic.primitives.Sum,
    try to simplify it by evaluating any Products and adding up ints and IntLiterals.
    If the sum can be reduced to a number, it returns an IntLiteral
    If the Sum reduces to one expression, it returns that expression

    Parameters
    ----------
    expr: any pymbolic expression
    """

    if isinstance(expr, pmbl.Sum):
        n = 0
        new_children = []
        for c in expr.children:
            c = product_value(c)
            if isinstance(c, IntLiteral):
                n += c.value
            elif isinstance(c, int):
                n += c
            else:
                new_children += [c]

        if new_children:
            if n > 0:
                new_children += [IntLiteral(n)]
            elif n < 0:
                new_children += [Product((-1,IntLiteral(abs(n))))]

            if len(new_children) > 1:
                return Sum(as_tuple(new_children))
            return new_children[0]

        else:
            return IntLiteral(n)

    return expr


def construct_range_index(lower, length):
    """
    Construct a range index from lower to lower + length - 1

    Parameters
    ----------
    lower : any pymbolic expression
    length: any pymbolic expression
    """

    new_high = simplify_sum(single_sum(length) + lower - IntLiteral(1))

    return RangeIndex((lower, new_high))


def process_symbol(symbol, caller, call):
    """
    Map symbol in call.routine to the appropriate symbol in caller,
    taking any parents into account

    Parameters
    ----------
    symbol: Loki variable in call.routine
    caller: Subroutine object containing call
    call  : Call object
    """

    if isinstance(symbol, IntLiteral):
        return symbol

    if not symbol.parents:
        if symbol in call.routine.arguments:
            return call.arg_map[symbol]

    elif symbol.parents[0] in call.routine.arguments:
        return SubstituteExpressions(call.arg_map).visit(symbol.clone(scope=caller))

    if call.routine in caller.members and symbol in caller.variables:
        return symbol

    raise RuntimeError('[Loki::fix_scalar_syntax] Unable to resolve argument dimension. Module variable?')


def construct_length(xrange, caller, call):
    """
    Construct an expression for the length of xrange,
    defined in call.routine, in caller.

    Parameters
    ----------
    xrange: RangeIndex object defined in call.routine
    caller: Subroutine object
    call  : call contained in caller
    """

    new_start = process_symbol(xrange.start, caller, call)
    new_stop  = process_symbol(xrange.stop, caller, call)

    return simplify_sum(single_sum(new_stop) - new_start + IntLiteral(1))


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

    #List calls in routine, but make sure we have the called routine definition
    calls = (c for c in FindNodes(CallStatement).visit(routine.body) if not c.procedure_type is BasicType.DEFERRED)
    call_map = {}

    for call in calls:

        new_args = []

        found_scalar = False
        for dummy, arg in call.arg_map.items():
            if check_if_scalar_syntax(arg, dummy):
                found_scalar = True

                new_dims = []
                for s, lower in zip(dummy.shape, arg.dimensions):

                    if isinstance(s, RangeIndex):
                        new_dims += [construct_range_index(lower, construct_length(s, routine, call))]
                    else:
                        new_dims += [construct_range_index(lower, process_symbol(s, routine, call))]

                if len(arg.dimensions) > len(dummy.shape):
                    new_dims += arg.dimensions[len(dummy.shape):]
                new_args += [arg.clone(dimensions=as_tuple(new_dims)),]
            else:
                new_args += [arg,]

        if found_scalar:
            call_map[call] = call.clone(arguments = as_tuple(new_args))

    if call_map:
        routine.body = Transformer(call_map).visit(routine.body)
