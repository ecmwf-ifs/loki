import weakref
from enum import IntEnum
from loki.tools import flatten


__all__ = ['DataType', 'SymbolType', 'TypeTable']


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
    and, to mark symbols without a known type, `DEFERRED` (e.g., for members of an externally
    defined derived type on use).

    For convenience, string representations of FORTRAN and C99 types can be
    heuristically converted.
    """

    DEFERRED = -1
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


class SymbolType:
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
        if value is None and name in dir(self):
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
            elif k == 'parent' and v is not None:
                parameters += ['parent=%s(%s)' % ('Type' if isinstance(v, SymbolType)
                                                  else 'Variable', v.name)]
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

    def clone(self, **kwargs):
        args = {k: v for k, v in self.__dict__.items()}
        args.update(kwargs)
        dtype = args.pop('dtype')
        return self.__class__(dtype, **args)


class TypeTable(dict):
    """
    Lookup table for types that essentially behaves like a class:``dict``.

    Used to store types for symbols or derived types within a scope.
    For derived types, no separate entries for the declared variables within a type
    are added. Instead, lookup methods (such as ``get``, ``__getitem__``, ``lookup`` etc.)
    disect the name and take care of chasing the information chain automagically.

    :param parent: class:``TypeTable`` instance of the parent scope to allow
                   for recursive lookups.
    :param case_sensitive: Treat names of variables to be case sensitive.
    """

    def __init__(self, parent=None, case_sensitive=False, **kwargs):
        super(TypeTable, self).__init__(**kwargs)
        self._parent = weakref.ref(parent) if parent is not None else None
        self._case_sensitive = case_sensitive

    @property
    def parent(self):
        return self._parent() if self._parent is not None else None

    def format_lookup_name(self, name):
        if not self._case_sensitive:
            name = name.lower()
        name = name.partition('(')[0]  # Remove any dimension parameters
        return name

    def _lookup(self, name, recursive):
        """
        Recursively look for a symbol in the table.
        """
        value = super(TypeTable, self).get(name, None)
        if value is None and recursive and self.parent is not None:
            return self.parent._lookup(name, recursive)
        else:
            return value

    def lookup(self, name, recursive=True):
        """
        Lookup a symbol in the type table and return the type or `None` if not found.

        :param name: Name of the type or symbol.
        :param recursive: If no entry by that name is found, try to find it in the
                          table of the parent scope.
        """
        name_parts = self.format_lookup_name(name)
        value = self._lookup(name_parts, recursive)
        return value

    def __getitem__(self, key):
        value = self.lookup(key, recursive=False)
        if value is None:
            raise KeyError(key)
        return value

    def get(self, key, default=None):
        value = self.lookup(key, recursive=False)
        return value if value is not None else default

    def __setitem__(self, key, value):
        name_parts = self.format_lookup_name(key)
        super(TypeTable, self).__setitem__(name_parts, value)

    def __hash__(self):
        return hash(tuple(self.keys()))

    def __repr__(self):
        return '<loki.types.TypeTable object at %s>' % hex(id(self))

    def setdefault(self, key, default=None):
        super(TypeTable, self).setdefault(self.format_lookup_name(key), default)
