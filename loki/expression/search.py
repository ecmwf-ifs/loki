"""
Expression search utilities that use SymPy's ``find` mechanism to
retrieve different types of symbols and functions using query
definitions.

Note, that SymPy's ``find` method might not be the fastest and can
be improved upon as demonstrated here:
https://github.com/opesci/devito/blob/master/devito/symbolics/search.py
"""

#from .symbol_types import BoolArray
from pymbolic.mapper import WalkMapper

__all__ = ['retrieve_symbols', 'retrieve_functions', 'retrieve_variables', 'ExpressionRetriever']


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

class ExpressionRetriever(WalkMapper):

    def __init__(self, query):
        super(ExpressionRetriever, self).__init__()

        self.query = query
        self.exprs = list() 

    def post_visit(self, expr, *args, **kwargs):
        if self.query(expr):
            self.exprs.append(expr)

    map_scalar = WalkMapper.map_variable

    def map_array(self, expr, *args, **kwargs):
        self.visit(expr)
        if expr.dimensions:
            for d in expr.dimensions:
                self.rec(d, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_logic_literal = WalkMapper.map_constant
    map_float_literal = WalkMapper.map_constant
    map_int_literal = WalkMapper.map_constant
    map_inline_call = WalkMapper.map_call_with_kwargs
    map_parenthesised_add = WalkMapper.map_sum
    map_parenthesised_mul = WalkMapper.map_product
    map_parenthesised_pow = WalkMapper.map_power

    def map_range_index(self, expr, *args, **kwargs):
        self.visit(expr)
        if expr.lower:
            self.rec(expr.lower, *args, **kwargs)
        if expr.upper:
            self.rec(expr.upper, *args, **kwargs)
        if expr.step:
            self.rec(expr.step, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)


def retrieve_symbols(expr):
    from pymbolic.primitives import Variable
    retriever = ExpressionRetriever(lambda e: isinstance(e, Variable))
    retriever(expr)
    return retriever.exprs


def retrieve_functions(expr):
    from pymbolic.primitives import Call
    retriever = ExpressionRetriever(lambda e: isinstance(e, Call))
    retriever(expr)
    return retriever.exprs


def retrieve_variables(expr):
    from pymbolic.primitives import Variable
    retriever = ExpressionRetriever(lambda e: isinstance(e, Variable))
    retriever(expr)
    return retriever.exprs


def retrieve_inline_calls(expr):
    from loki.expression.symbol_types import InlineCall
    retriever = ExpressionRetriever(lambda e: isinstance(e, InlineCall))
    retriever(expr)
    return retriever.exprs
