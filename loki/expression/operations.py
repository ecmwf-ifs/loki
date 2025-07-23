# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Sub-classes of Pymbolic's native operations that allow us to inject
Fortran-specific features, such as case-insensitive string comparison
and bracket-aware and sub-expression grouping. Here we also add
additional technical operations, such as cast and references.
"""

from sys import intern
import pymbolic.primitives as pmbl

from loki.tools import as_tuple

from loki.expression.literals import StringLiteral
from loki.expression.mixins import loki_make_stringifier, StrCompareMixin


__all__ = [
    'Sum', 'Product', 'Quotient', 'Power',
    'Comparison', 'LogicalAnd', 'LogicalOr', 'LogicalNot',
    'StringConcat', 'Cast', 'Reference', 'Dereference'
]


class Sum(StrCompareMixin, pmbl.Sum):
    """Representation of a sum."""


class Product(StrCompareMixin, pmbl.Product):
    """Representation of a product."""


class Quotient(StrCompareMixin, pmbl.Quotient):
    """Representation of a quotient."""


class Power(StrCompareMixin, pmbl.Power):
    """Representation of a power."""


class Comparison(StrCompareMixin, pmbl.Comparison):
    """Representation of a comparison operation."""


class LogicalAnd(StrCompareMixin, pmbl.LogicalAnd):
    """Representation of an 'and' in a logical expression."""


class LogicalOr(StrCompareMixin, pmbl.LogicalOr):
    """Representation of an 'or' in a logical expression."""


class LogicalNot(StrCompareMixin, pmbl.LogicalNot):
    """Representation of a negation in a logical expression."""


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


class StringConcat(pmbl._MultiChildExpression):
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


class Cast(StrCompareMixin, pmbl.Call):
    """
    Internal representation of a data type cast.
    """

    init_arg_names = ('name', 'expression', 'kind')

    def __init__(self, name, expression, kind=None, **kwargs):
        assert kind is None or isinstance(kind, pmbl.Expression)
        self.kind = kind
        super().__init__(pmbl.make_variable(name), as_tuple(expression), **kwargs)

    def __getinitargs__(self):
        return (self.name, self.expression, self.kind)

    mapper_method = intern('map_cast')

    @property
    def name(self):
        return self.function.name

    @property
    def expression(self):
        return self.parameters


class Reference(StrCompareMixin, pmbl.Expression):
    """
    Internal representation of a Reference.

    .. warning:: Experimental! Allowing compound
        ``Reference(Variable(...))`` to appear
        with behaviour akin to a symbol itself
        for easier processing in mappers.

    **C/C++ only**, no corresponding concept in Fortran.
    Referencing refers to taking the address of an
    existing variable (to set a pointer variable).
    """
    init_arg_names = ('expression',)

    def __getinitargs__(self):
        return (self.expression, )

    def __init__(self, expression):
        assert isinstance(expression, pmbl.Expression)
        self.expression = expression

    @property
    def name(self):
        """
        Allowing the compound ``Reference(Variable(name))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.name

    @property
    def type(self):
        """
        Allowing the compound ``Reference(Variable(type))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.type

    @property
    def scope(self):
        """
        Allowing the compound ``Reference(Variable(scope))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.scope

    @property
    def initial(self):
        """
        Allowing the compound ``Reference(Variable(initial))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.initial

    mapper_method = intern('map_c_reference')


class Dereference(StrCompareMixin, pmbl.Expression):
    """
    Internal representation of a Dereference.

    .. warning:: Experimental! Allowing compound
        ``Dereference(Variable(...))`` to appear
        with behaviour akin to a symbol itself
        for easier processing in mappers.

    **C/C++ only**, no corresponding concept in Fortran.
    Dereferencing (a pointer) refers to retrieving the value
    from a memory address (that is pointed by the pointer).
    """
    init_arg_names = ('expression', )

    def __getinitargs__(self):
        return (self.expression, )

    def __init__(self, expression):
        assert isinstance(expression, pmbl.Expression)
        self.expression = expression

    @property
    def name(self):
        """
        Allowing the compound ``Dereference(Variable(name))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.name

    @property
    def type(self):
        """
        Allowing the compound ``Dereference(Variable(type))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.type

    @property
    def scope(self):
        """
        Allowing the compound ``Dereference(Variable(scope))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.scope

    @property
    def initial(self):
        """
        Allowing the compound ``Dereference(Variable(initial))`` to appear
        with behaviour akin to a symbol itself for easier processing in mappers.
        """
        return self.expression.initial

    mapper_method = intern('map_c_dereference')
