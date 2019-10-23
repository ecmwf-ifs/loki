"""
Expression search utilities that use SymPy's ``find` mechanism to
retrieve different types of symbols and functions using query
definitions.

Note, that SymPy's ``find` method might not be the fastest and can
be improved upon as demonstrated here:
https://github.com/opesci/devito/blob/master/devito/symbolics/search.py
"""

#from .symbol_types import BoolArray
from pymbolic.mapper.dependency import DependencyMapper

__all__ = ['retrieve_symbols', 'retrieve_functions', 'retrieve_variables']


#def q_symbol(expr):
#    return expr.is_Symbol
#
#
#def q_scalar(expr):
#    return expr.is_Number or expr.is_Symbol
#
#
#def q_function(expr):
#    return expr.is_Function
#
#
#def q_variable(expr):
#    if expr.is_Function and (not expr.is_Boolean or isinstance(expr, BoolArray)):
#        # Prevent picking up boolean function (And, Or, Not, etc)
#        return expr.is_Array
#    else:
#        return expr.is_Symbol


def retrieve_symbols(expr):
    depmap = DependencyMapper()
    return depmap(expr)
#    return expr.find(q_symbol)


def retrieve_functions(expr):
    depmap = DependencyMapper()
    return depmap(expr)
#    return expr.find(q_function)


def retrieve_variables(expr):
    depmap = DependencyMapper()
    return depmap(expr)
#    return expr.find(q_variable)
