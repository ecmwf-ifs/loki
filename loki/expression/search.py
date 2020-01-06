"""
Expression search utilities that use Pymbolic's mapping mechanism to
retrieve different types of symbols and functions using query
definitions.
"""

from pymbolic.primitives import Expression
from loki.expression.symbol_types import InlineCall, Scalar, Array
from loki.expression.visitors import ExpressionRetriever

__all__ = ['retrieve_expressions', 'retrieve_variables']


def retrieve_expressions(expr):
    retriever = ExpressionRetriever(lambda e: isinstance(e, Expression))
    retriever(expr)
    return retriever.exprs


def retrieve_variables(expr):
    retriever = ExpressionRetriever(lambda e: isinstance(e, (Scalar, Array)))
    retriever(expr)
    return retriever.exprs


def retrieve_inline_calls(expr):
    retriever = ExpressionRetriever(lambda e: isinstance(e, InlineCall))
    retriever(expr)
    return retriever.exprs
