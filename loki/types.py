"""
Collection of classes to represent type information for symbols used
throughout Loki's :ref:`internal_representation:internal representation`
"""

from enum import IntEnum
from collections import OrderedDict
from loki.tools import flatten, as_tuple


__all__ = ['DataType', 'BasicType', 'DerivedType', 'ProcedureType', 'SymbolAttributes']


class DataType:
    """
    Base class for data types a symbol may have
    """

    def __init__(self, *args): # pylint:disable=unused-argument
        # Make sure we always instantiate one of the subclasses
        # Note that we cannot use ABC for that as this would cause a
        # metaclass clash with IntEnum, which is used to represent
        # intrinsic types in BasicType
        assert self.__class__ is not DataType


class BasicType(DataType, IntEnum):
    """
    Representation of intrinsic data types, names taken from the FORTRAN convention.

    Currently, there are

    - :any:`LOGICAL`
    - :any:`INTEGER`
    - :any:`REAL`
    - :any:`CHARACTER`
    - :any:`COMPLEX`

    and, to indicate an undefined data type (e.g., for imported
    symbols whose definition is not available), :any:`DEFERRED`.

    For convenience, string representations of FORTRAN and C99 types can be
    heuristically converted.
    """

    DEFERRED = -1
    LOGICAL = 1
    INTEGER = 2
    REAL = 3
    CHARACTER = 4
    COMPLEX = 5

    @classmethod
    def from_str(cls, value):
        """
        Try to convert the given string using one of the `from_*` methods.
        """
        lookup_methods = (cls.from_name, cls.from_fortran_type, cls.from_c99_type)
        for meth in lookup_methods:
            try:
                return meth(value)
            except KeyError:
                pass
        return ValueError('Unknown data type: %s' % value)

    @classmethod
    def from_name(cls, value):
        """
        Convert the given string representation of the :any:`BasicType`.
        """
        return {t.name: t for t in cls}[value]

    @classmethod
    def from_fortran_type(cls, value):
        """
        Convert the given string representation of a FORTRAN type.
        """
        type_map = {'logical': cls.LOGICAL, 'integer': cls.INTEGER, 'real': cls.REAL,
                    'double precision': cls.REAL, 'double complex': cls.COMPLEX,
                    'character': cls.CHARACTER, 'complex': cls.COMPLEX}
        return type_map[value.lower()]

    @classmethod
    def from_c99_type(cls, value):
        """
        Convert the given string representation of a C99 type.
        """
        logical_types = ['bool', '_Bool']
        integer_types = ['short', 'int', 'long', 'long long']
        integer_types += flatten([('signed %s' % t, 'unsigned %s' % t) for t in integer_types])
        real_types = ['float', 'double', 'long double']
        character_types = ['char']
        complex_types = ['float _Complex', 'double _Complex', 'long double _Complex']

        type_map = {t: cls.LOGICAL for t in logical_types}
        type_map.update({t: cls.INTEGER for t in integer_types})
        type_map.update({t: cls.REAL for t in real_types})
        type_map.update({t: cls.CHARACTER for t in character_types})
        type_map.update({t: cls.COMPLEX for t in complex_types})

        return type_map[value]


class DerivedType(DataType):
    """
    Representation of derived data types that may have an associated `TypeDef`.

    Please note that the typedef attribute may be of `ir.TypeDef` or `BasicType.DEFERRED`,
    depending on the scope of the derived type declaration.
    """

    def __init__(self, name=None, typedef=None):
        super().__init__()
        assert name or typedef
        self._name = name
        self.typedef = typedef if typedef is not None else BasicType.DEFERRED

        # This is intentionally left blank, as the parent variable
        # generation will populate this, if the typedef is known.
        self.variables = tuple()

    @property
    def name(self):
        return self._name if self.typedef is BasicType.DEFERRED else self.typedef.name

    @property
    def variable_map(self):
        return OrderedDict([(v.basename, v) for v in self.variables])

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<DerivedType {}>'.format(self.name)


class ProcedureType(DataType):
    """
    Representation of a function or subroutine type definition.
    """

    def __init__(self, name=None, is_function=False, procedure=None):
        super().__init__()
        assert name or procedure
        self._name = name
        self._is_function = is_function
        self.procedure = procedure if procedure is not None else BasicType.DEFERRED

    @property
    def name(self):
        return self._name if self.procedure is BasicType.DEFERRED else self.procedure.name

    @property
    def parameters(self):
        if self.procedure is BasicType.DEFERRED:
            return tuple()
        return self.procedure.arguments

    @property
    def is_function(self):
        if self.procedure is BasicType.DEFERRED:
            return self._is_function
        return self.procedure.is_function

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<ProcedureType {}>'.format(self.name)


class SymbolAttributes:
    """
    Representation of a symbol's attributes, such as data type and declared
    properties

    It has a fixed :any:`DataType` associated with it, available as property
    :attr:`SymbolAttributes.dtype`.

    Any other properties can be attached on-the-fly, thus allowing to store
    arbitrary metadata for a symbol, e.g., declaration attributes such as
    ``POINTER``, ``ALLOCATABLE``, or the shape of an array, or structural
    information, e.g., whether a variable is a loop index, argument, etc.

    There is no need to check for the presence of attributes, undefined
    attributes can be queried and default to `None`.

    Parameters
    ----------
    dtype : :any:`DataType`
        The data type associated with the symbol
    **kwargs : optional
        Any attributes that should be stored as properties
    """

    def __init__(self, dtype, **kwargs):
        if isinstance(dtype, DataType):
            self.dtype = dtype
        else:
            self.dtype = BasicType.from_str(dtype)

        for k, v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)

    def __setattr__(self, name, value):
        if value is None and name in dir(self):
            delattr(self, name)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name not in dir(self):
            return None
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        object.__delattr__(self, name)

    def __repr__(self):
        parameters = [str(self.dtype)]
        for k, v in self.__dict__.items():
            if k in ['dtype', 'source']:
                continue
            if isinstance(v, bool):
                if v:
                    parameters += [str(k)]
            elif k == 'parent' and v is not None:
                parameters += ['parent=%s(%s)' % ('Type' if isinstance(v, SymbolAttributes)
                                                  else 'Variable', v.name)]
            else:
                parameters += ['%s=%s' % (k, str(v))]
        return '<{} {}>'.format(self.__class__.__name__, ', '.join(parameters))

    def __getinitargs__(self):
        args = [self.dtype]
        for k, v in self.__dict__.items():
            if k in ['dtype', 'source']:
                continue
            args += [(k, v)]
        return tuple(args)

    def clone(self, **kwargs):
        """
        Clone the :any:`SymbolAttributes`, optionally overwriting any attributes

        Attributes that should be removed should simply be given as `None`.
        """
        args = self.__dict__.copy()
        args.update(kwargs)
        dtype = args.pop('dtype')
        return self.__class__(dtype, **args)

    def compare(self, other, ignore=None):
        """
        Compare :any:`SymbolAttributes` objects while ignoring a set of select attributes.

        Parameters
        ----------
        other : :any:`SymbolAttributes`
            The object to compare with
        ignore : iterable, optional
            Names of attributes to ignore while comparing.

        Returns
        -------
        bool
        """
        ignore_attrs = as_tuple(ignore)
        keys = set(as_tuple(self.__dict__.keys()) + as_tuple(other.__dict__.keys()))
        return all(self.__dict__.get(k) == other.__dict__.get(k)
                   for k in keys if k not in ignore_attrs)
