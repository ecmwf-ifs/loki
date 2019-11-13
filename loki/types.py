from enum import IntEnum
from loki.tools import flatten


__all__ = ['DataType', 'SymbolType', 'SymbolTable']


class DataType(IntEnum):
    """
    Representation of intrinsic data types, names taken from the FORTRAN convention.

    Currently, there are
    - `LOGICAL`
    - `INTEGER`
    - `REAL`
    - `CHARACTER`
    - `COMPLEX`
    - `DERIVED_TYPE`

    For convenience, string representations of FORTRAN and C99 types can be
    heuristically converted.
    """

    LOGICAL = 1
    INTEGER = 2
    REAL = 3
    CHARACTER = 4
    COMPLEX = 5
    DERIVED_TYPE = 6

    @classmethod
    def from_str(cls, value):
        """
        Try to convert the given string using any of the `from_*_type` methods.
        """
        lookup_methods = (cls.from_fortran_type, cls.from_c99_type)
        for meth in lookup_methods:
            try:
                return meth(value)
            except KeyError:
                pass
        return ValueError('Unknown data type: %s' % value)

    @classmethod
    def from_fortran_type(cls, value):
        """
        Convert the given string representation of a FORTRAN type.
        """
        type_map = {'logical': cls.LOGICAL, 'integer': cls.INTEGER, 'real': cls.REAL,
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


class SymbolType(object):
    """
    Representation of a symbols type.

    It has a fixed class:``DataType`` associated, available as the property `DataType.dtype`.

    Any other properties can be attached on-the-fly, thus allowing to store arbitrary metadata
    for a symbol, e.g., declaration attributes such as `POINTER`, `ALLOCATABLE` or structural
    information, e.g., whether a variable is a loop index, argument, etc.

    There is no need to check for the presence of attributes, undefined attributes can be queried
    and default to `None`.
    """

    def __init__(self, dtype, **kwargs):
        self.dtype = dtype if isinstance(dtype, DataType) else DataType.from_str(dtype)

        for k, v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)

    def __setattr__(self, name, value):
        if value is None:
            delattr(self, name)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return object.__getattr__(self, name) if name in dir(self) else None

    def __delattr__(self, name):
        object.__delattr__(self, name)

    def __repr__(self):
        parameters = [str(self.dtype)]
        for k, v in self.__dict__.items():
            if k in ['dtype', 'source']:
                continue
            elif isinstance(v, bool):
                if v:
                    parameters += [str(k)]
            else:
                parameters += ['%s=%s' % (k, str(v))]
        return '<Type %s>' % ', '.join(parameters)

    def __getinitargs__(self):
        args = [self.dtype]
        for k, v in self.__dict__.items():
            if k in ['dtype', 'source']:
                continue
            else:
                args += [(k, v)]
        return tuple(args)


class SymbolTable(dict):
    """
    Lookup table for the type of symbols within a scope.
    """

    def __init__(self, scope, **kwargs):
        super(SymbolTable, self).__init__(**kwargs)

        self.scope = scope

