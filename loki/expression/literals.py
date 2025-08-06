# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Literal symbols representing numbers, strings and logicals.
"""

from sys import intern
import pymbolic.primitives as pmbl
from pymbolic.mapper.evaluator import UnknownVariableError

from loki.types import BasicType
from loki.expression.mixins import StrCompareMixin


__all__ = [
    '_Literal', 'FloatLiteral', 'IntLiteral', 'LogicLiteral',
    'StringLiteral', 'IntrinsicLiteral', 'Literal', 'LiteralList'
]


class _Literal(pmbl.Leaf):
    """
    Base class for literals.

    This exists to overcome the problem of a disfunctional
    :meth:`__getinitargs__` in any:`pymbolic.primitives.Leaf`.
    """

    def __getinitargs__(self):
        return ()


class FloatLiteral(StrCompareMixin, _Literal):
    """
    A floating point constant in an expression.

    Note that its :data:`value` is stored as a string to avoid any
    representation issues that could stem from converting it to a
    Python floating point number.

    It can have a specific type associated, which backends can use to cast
    or annotate the constant to make sure the specified type is used.

    Parameters
    ----------
    value : str
        The value of that literal.
    kind : optional
        The kind information for that literal.
    """

    def __init__(self, value, **kwargs):
        # We store float literals as strings to make sure no information gets
        # lost in the conversion
        self.value = str(value)
        self.kind = kwargs.pop('kind', None)
        super().__init__(**kwargs)

    def __hash__(self):
        return hash((self.value, self.kind))

    def __eq__(self, other):
        if isinstance(other, FloatLiteral):
            return self.value == other.value and self.kind == other.kind

        try:
            return float(self.value) == float(other)
        except (TypeError, ValueError, UnknownVariableError):
            return False

    def __lt__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) < float(other.value)
        try:
            return float(self.value) < float(other)
        except ValueError:
            return super().__lt__(other)

    def __le__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) <= float(other.value)
        try:
            return float(self.value) <= float(other)
        except ValueError:
            return super().__le__(other)

    def __gt__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) > float(other.value)
        try:
            return float(self.value) > float(other)
        except ValueError:
            return super().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, FloatLiteral):
            return float(self.value) >= float(other.value)
        try:
            return float(self.value) >= float(other)
        except ValueError:
            return super().__ge__(other)

    init_arg_names = ('value', 'kind')

    def __getinitargs__(self):
        return (self.value, self.kind)

    mapper_method = intern('map_float_literal')


class IntLiteral(StrCompareMixin, _Literal):
    """
    An integer constant in an expression.

    It can have a specific type associated, which backends can use to cast
    or annotate the constant to make sure the specified type is used.

    Parameters
    ----------
    value : int
        The value of that literal.
    kind : optional
        The kind information for that literal.
    """

    def __init__(self, value, **kwargs):
        self.value = int(value)
        self.kind = kwargs.pop('kind', None)
        super().__init__(**kwargs)

    def __hash__(self):
        return hash((self.value, self.kind))

    def __eq__(self, other):
        if isinstance(other, IntLiteral):
            return self.value == other.value and self.kind == other.kind
        if isinstance(other, (int, float, complex)):
            return self.value == other

        try:
            return self.value == int(other)
        except (TypeError, ValueError):
            return False

    def __lt__(self, other):
        if isinstance(other, IntLiteral):
            return self.value < other.value
        if isinstance(other, int):
            return self.value < other
        return super().__lt__(other)

    def __le__(self, other):
        if isinstance(other, IntLiteral):
            return self.value <= other.value
        if isinstance(other, int):
            return self.value <= other
        return super().__le__(other)

    def __gt__(self, other):
        if isinstance(other, IntLiteral):
            return self.value > other.value
        if isinstance(other, int):
            return self.value > other
        return super().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, IntLiteral):
            return self.value >= other.value
        if isinstance(other, int):
            return self.value >= other
        return super().__ge__(other)

    init_arg_names = ('value', 'kind')

    def __getinitargs__(self):
        return (self.value, self.kind)

    def __int__(self):
        return self.value

    def __bool__(self):
        return bool(self.value)

    mapper_method = intern('map_int_literal')


# Register IntLiteral as a constant class in Pymbolic
pmbl.register_constant_class(IntLiteral)


class LogicLiteral(StrCompareMixin, _Literal):
    """
    A boolean constant in an expression.

    Parameters
    ----------
    value : bool
        The value of that literal.
    """

    def __init__(self, value, **kwargs):
        self.value = str(value).lower() in ('true', '.true.')
        super().__init__(**kwargs)

    init_arg_names = ('value', )

    def __getinitargs__(self):
        return (self.value, )

    def __bool__(self):
        return self.value

    mapper_method = intern('map_logic_literal')


class StringLiteral(StrCompareMixin, _Literal):
    """
    A string constant in an expression.

    Parameters
    ----------
    value : str
        The value of that literal. Enclosing quotes are removed.
    """

    def __init__(self, value, **kwargs):
        # Remove quotation marks
        if value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]

        self.value = value

        super().__init__(**kwargs)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, StringLiteral):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False

    init_arg_names = ('value', )

    def __getinitargs__(self):
        return (self.value, )

    mapper_method = intern('map_string_literal')


class IntrinsicLiteral(StrCompareMixin, _Literal):
    """
    Any literal not represented by a dedicated class.

    Its value is stored as string and returned unaltered.
    This is currently used for complex and BOZ constants and to retain
    array constructor expressions with type spec or implied-do.

    Parameters
    ----------
    value : str
        The value of that literal.
    """

    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    init_arg_names = ('value', )

    def __getinitargs__(self):
        return (self.value, )

    mapper_method = intern('map_intrinsic_literal')


class Literal:
    """
    Factory class to instantiate the best-matching literal node.

    This always returns a :class:`IntLiteral`, :class:`FloatLiteral`,
    :class:`StringLiteral`, :class:`LogicLiteral` or, as a fallback,
    :class:`IntrinsicLiteral`, selected by using any provided :data:`type`
    information or inspecting the Python data type of :data: value.

    Parameters
    ----------
    value :
        The value of that literal.
    kind : optional
        The kind information for that literal.
    """

    @staticmethod
    def _from_literal(value, **kwargs):

        cls_map = {BasicType.INTEGER: IntLiteral, BasicType.REAL: FloatLiteral,
                   BasicType.LOGICAL: LogicLiteral, BasicType.CHARACTER: StringLiteral}

        _type = kwargs.pop('type', None)
        if _type is None:
            if isinstance(value, int):
                _type = BasicType.INTEGER
            elif isinstance(value, float):
                _type = BasicType.REAL
            elif isinstance(value, str):
                if str(value).lower() in ('.true.', 'true', '.false.', 'false'):
                    _type = BasicType.LOGICAL
                else:
                    _type = BasicType.CHARACTER

        return cls_map[_type](value, **kwargs)

    def __new__(cls, value, **kwargs):
        try:
            obj = cls._from_literal(value, **kwargs)
        except KeyError:
            obj = IntrinsicLiteral(value, **kwargs)

        # And attach our own meta-data
        if hasattr(obj, 'kind'):
            obj.kind = kwargs.get('kind', None)
        return obj


class LiteralList(StrCompareMixin, pmbl.AlgebraicLeaf):
    """
    A list of constant literals, e.g., as used in Array Initialization Lists.
    """

    init_arg_names = ('values', 'dtype')

    def __init__(self, values, dtype=None, **kwargs):
        self.elements = values
        self.dtype = dtype
        super().__init__(**kwargs)

    mapper_method = intern('map_literal_list')

    def __getinitargs__(self):
        return (self.elements, self.dtype)
