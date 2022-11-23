# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Sub-classes of Pymbolic's native operations that allow us to express
niche things like mathematically irrelevant parenthesis that
nevertheless change code results.
"""

from sys import intern
import pymbolic.primitives as pmbl

from loki.expression.symbols import (
    StringLiteral, ExprMetadataMixin, Sum, Product, Quotient, Power,
    loki_make_stringifier
)


class ParenthesisedAdd(Sum):
    """
    Specialised version of :class:`Sum` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    mapper_method = intern("map_parenthesised_add")
    make_stringifier = loki_make_stringifier


class ParenthesisedMul(Product):
    """
    Specialised version of :class:`Product` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    mapper_method = intern("map_parenthesised_mul")
    make_stringifier = loki_make_stringifier


class ParenthesisedDiv(Quotient):
    """
    Specialised version of :class:`Quotient` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    mapper_method = intern("map_parenthesised_div")
    make_stringifier = loki_make_stringifier


class ParenthesisedPow(Power):
    """
    Specialised version of :class:`Power` that always pretty-prints and
    code-generates with explicit parentheses.
    """

    mapper_method = intern("map_parenthesised_pow")
    make_stringifier = loki_make_stringifier


class StringConcat(ExprMetadataMixin, pmbl._MultiChildExpression):
    """
    Implements string concatenation in a way similar to :class:`Sum`.
    """

    def __add__(self, other):
        if isinstance(other, (StringConcat, StringLiteral, pmbl.Variable)):
            return StringConcat((self, other))
        if not other:
            return self
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (StringConcat, StringLiteral, pmbl.Variable)):
            return StringConcat((other, self))
        if not other:
            return self
        return NotImplemented

    def __bool__(self):
        if len(self.children) == 1:
            return bool(self.children[0])
        return True

    __nonzero__ = __bool__

    mapper_method = intern("map_string_concat")
