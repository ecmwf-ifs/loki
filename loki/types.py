r"""
Collection of classes to represent type information for symbols used
throughout :ref:`internal_representation:Loki's internal representation`

The key ideas of Loki's type system are:

* Every symbol (:any:`TypedSymbol`, such as :any:`Scalar`,
  :any:`Array`, :any:`ProcedureSymbol`) has a type (represented by a
  :any:`DataType`) and, possibly, other attributes associated with it.
  Type and attributes are stored together in a :any:`SymbolAttributes`
  object, which is essentially a `dict`.

  .. note::
     An array variable ``VAR`` may be declared in Fortran as a subroutine
     argument in the following way:

     .. code-block:: none

        INTEGER(4), INTENT(INOUT) :: VAR(10)

     This variable has type :any:`BasicType.INTEGER` and the following
     additional attributes:

     * ``KIND=4``
     * ``INTENT=INOUT``
     * ``SHAPE=(10,)``

     The corresponding :any:`SymbolAttributes` object can be created as

     .. code-block::

        SymbolAttributes(BasicType.INTEGER, kind=Literal(4), intent='inout', shape=(Literal(10),))

* The :any:`SymbolAttributes` object is stored in the relevant :any:`SymbolTable`
  and queried from there by all expression nodes that represent use of the
  associated symbol. This means, changing the declared attributes of a symbol
  applies this change for all instances of this symbol.

  .. warning::
     Changing symbol attributes can lead to invalid states.

     For example, removing the ``shape`` property from the :any:`SymbolAttributes`
     object in a symbol table converts the corresponding :any:`Array` to
     a :any:`Scalar` variable. But at this point all expression tree nodes will
     still be :any:`Array`, possibly also with subscript operations (represented
     by the ``dimensions`` property).

     For plain :any:`Array` nodes (without subscript), rebuilding the IR will
     automatically take care of instantiating these objects as :any:`Scalar` but
     removing ``dimensions`` properties must be done explicitly.

* Every object that defines a new scope (e.g., :any:`Subroutine`,
  :any:`Module`, implementing :any:`Scope`) has an associated symbol table
  (:any:`SymbolTable`). The :any:`SymbolAttributes` of a symbol declared or
  imported in a scope are stored in the symbol table of that scope.
* The symbol tables/scopes are organized in a hierarchical fashion, i.e., they
  are aware of their enclosing scope and allow to recursively look-up entries.
* The overall schematics of the scope and type representation are depicted in the
  following diagram:

  .. code-block:: none

        Subroutine | Module | TypeDef | ...
                \      |      /
                 \     |     /   <is>
                  \    |    /
                     Scope
                       |
                       | <has>
                       |
                  SymbolTable  - - - - - - - - - - - - TypedSymbol
                       |
                       |  <has entries>
                       |
                SymbolAttributes
             /     |       |      \
            /      |       |       \  <has properties>
           /       |       |        \
     DataType | (kind) | (intent) | (...)

* The :any:`DataType` of a symbol can be one of

  * :any:`BasicType`: intrinsic types, such as ``INTEGER``, ``REAL``, etc.
  * :any:`DerivedType`: derived types defined somewhere
  * :any:`ProcedureType`: any subroutines or functions declared or imported

  Note that this is different from the understanding of types in the Fortran
  standard, where only intrinsic types and derived types are considered a
  type. Treating also procedures as types allows us to treat them uniformly
  when considering external subprograms, procedure pointers and type bound
  procedures.

  .. code-block:: none

     BasicType | DerivedType | ProcedureType
              \       |       /
               \      |      /    <implements>
                \     |     /
                   DataType

* Derived type definitions (via :any:`TypeDef`) also create entries in the
  symbol table to make the type definition available to declarations.
* For imported symbols (via :any:`Import`) the source module may not be
  available and thus no information about the symbol. This is indicated by
  :any:`BasicType.DEFERRED`.
"""

import weakref
from enum import IntEnum
from collections import OrderedDict
from loki.tools import flatten, as_tuple


__all__ = [
    'DataType', 'BasicType', 'DerivedType', 'ProcedureType',
    'SymbolAttributes', 'SymbolTable', 'Scope'
]


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

    and, to indicate a currently not defined data type (e.g., for imported
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


class SymbolTable(dict):
    """
    Lookup table for symbol types that maps symbol names to :any:`SymbolAttributes`

    It is used to store types for declared variables, defined types or imported
    symbols within their respective scope. If its associated scope is nested
    into an enclosing scope, it allows to perform recursive look-ups in parent
    scopes.

    The interface of this table behaves like a :class:`dict`.

    Parameters
    ----------
    parent : :any:`SymbolTable`, optional
        The symbol table of the parent scope for recursive look-ups.
    case_sensitive : bool, optional
        Respect the case of symbol names in lookups (default: `False`).
    """

    def __init__(self, parent=None, case_sensitive=False, **kwargs):
        super().__init__(**kwargs)
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
        value = super().get(name, None)
        if value is None and recursive and self.parent is not None:
            return self.parent._lookup(name, recursive)
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

    def __contains__(self, key):
        return super().__contains__(self.format_lookup_name(key))

    def __getitem__(self, key):
        value = self.lookup(key, recursive=False)
        if value is None:
            raise KeyError(key)
        return value

    def get(self, key, default=None):
        value = self.lookup(key, recursive=False)
        return value if value is not None else default

    def __setitem__(self, key, value):
        assert isinstance(value, SymbolAttributes)
        name_parts = self.format_lookup_name(key)
        super().__setitem__(name_parts, value)

    def __hash__(self):
        return hash(tuple(self.keys()))

    def __repr__(self):
        return '<loki.types.SymbolTable object at %s>' % hex(id(self))

    def setdefault(self, key, default=None):
        super().setdefault(self.format_lookup_name(key), default)

    def update(self, other):
        if isinstance(other, dict):
            other = {self.format_lookup_name(k): v for k, v in other.items()}
        else:
            other = {self.format_lookup_name(k): v for k, v in other}
        super().update(other)


class Scope:
    """
    Scoping object that manages type caching and derivation for typed symbols.

    The :any:`Scope` provides a symbol table that uniquely maps a symbol's name
    to its :any:`SymbolAttributes` or, for a derived type definition, directly
    to its :any:`DerivedType`.

    See :any:`SymbolTable` for more details on how to look-up symbols.

    Parameters
    ----------
    parent : :any:`Scope`, optional
        The enclosing scope, thus allowing recursive look-ups
    """

    def __init__(self, parent=None):
        self._parent = weakref.ref(parent) if parent is not None else None

        parent_symbols = self.parent.symbols if self.parent is not None else None
        self.symbols = SymbolTable(parent=parent_symbols)

        # Potential link-back to the owner that can be used to
        # traverse the dependency chain upwards.
        self._defined_by = None

    @property
    def parent(self):
        """
        Access the enclosing scope.
        """
        return self._parent() if self._parent is not None else None

    @property
    def defined_by(self):
        """
        Object that owns this :any:`Scope` and defines the types and symbols it connects
        """
        return self._defined_by() if self._defined_by is not None else None

    @defined_by.setter
    def defined_by(self, value):
        """
        Ensure we only ever store a weakref to the defining object.
        """
        self._defined_by = weakref.ref(value)

    def __repr__(self):
        # pylint: disable=no-member
        if self.defined_by is not None and self.defined_by.name:
            return 'Scope::{}'.format(self.defined_by.name)
        return 'Scope::'
