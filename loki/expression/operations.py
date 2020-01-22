"""
Sub-classes of Pymbolic's native operations that allow us to express
niche things like mathematically irrelevant parenthesis that
nevertheless change code results.
"""

from pymbolic.primitives import Sum, Product, Power
from six.moves import intern

from loki.expression.visitors import LokiStringifyMapper


class ParenthesisedAdd(Sum):
    """
    Specialised version of :class:`Sum` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    mapper_method = intern("map_parenthesised_add")

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class ParenthesisedMul(Product):
    """
    Specialised version of :class:`Product` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    mapper_method = intern("map_parenthesised_mul")

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()


class ParenthesisedPow(Power):
    """
    Specialised version of :class:`Power` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    mapper_method = intern("map_parenthesised_pow")

    def make_stringifier(self, originating_stringifier=None):
        return LokiStringifyMapper()
