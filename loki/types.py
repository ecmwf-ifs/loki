"""
Collection of classes to represent type information for symbols used throughout
:doc:`internal_representation`

"""

import weakref
from enum import IntEnum
from loki.tools import flatten, as_tuple, LazyNodeLookup


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
        raise ValueError(f'Unknown data type: {value}')

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
        integer_types += flatten([(f'signed {t}', f'unsigned {t}') for t in integer_types])
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
    Representation of derived data types that may have an associated :any:`TypeDef`

    Please note that the typedef attribute may be of :any:`TypeDef` or
    :any:`BasicType.DEFERRED`, if the associated type definition is not available.

    Parameters
    ----------
    name : str, optional
        The name of the derived type. Can be omitted if :data:`typedef` is provided
    typedef : :any:`TypeDef`, optional
        The definition of the derived type. Takes precedence over :data:`name`
    """

    def __init__(self, name=None, typedef=None):
        super().__init__()
        assert name or typedef
        self._name = name
        self.typedef = typedef if typedef is not None else BasicType.DEFERRED

    @property
    def name(self):
        return self._name if self.typedef is BasicType.DEFERRED else self.typedef.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<DerivedType {self.name}>'


class ProcedureType(DataType):
    """
    Representation of a function or subroutine type definition.

    This serves also as the cross-link between the use of a procedure (e.g. in a
    :any:`CallStatement`) to the :any:`Subroutine` object that is the target of
    a call. If the corresponding object is not yet available when the
    :any:`ProcedureType` object is created, or its definition is transient and
    subject to IR rebuilds (e.g. :any:`StatementFunction`), the :any:`LazyNodeLookup`
    utility can be used to defer the actual instantiation. In that situation,
    :data:`name` should be provided in addition.

    Parameters
    ----------
    name : str, optional
        The name of the function or subroutine. Can be skipped if :data:`procedure`
        is provided (not in the form of a :any:`LazyNodeLookup`)
    is_function : bool, optional
        Indicate that this is a function
    is_generic : bool, optional
        Indicate that this is a generic function
    procedure : :any:`Subroutine` or :any:`StatementFunction` or :any:`LazyNodeLookup`, optional
        The procedure this type represents
    """

    def __init__(self, name=None, is_function=None, is_generic=False, procedure=None, return_type=None):
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel
        super().__init__()
        assert name or isinstance(procedure, Subroutine)
        assert isinstance(return_type, SymbolAttributes) or procedure or not is_function
        self.is_generic = is_generic
        if procedure is None or isinstance(procedure, LazyNodeLookup):
            self._procedure = procedure
            self._name = name
            self._is_function = is_function or False
            self._return_type = return_type
            # NB: not applying an assert on the procedure name for LazyNodeLookup as
            # the point of the lazy lookup is that we might not have the the procedure
            # definition available at type instantiation time
        else:
            self._procedure = weakref.ref(procedure)
            # Cache all properties for when procedure link becomes inactive
            assert name is None or name.lower() == self.procedure.name.lower()
            self._name = self.procedure.name
            assert is_function is None or is_function == self.procedure.is_function
            self._is_function = self.procedure.is_function
            # TODO: compare return type once type comparison is more robust
            self._return_type = self.procedure.return_type

    @property
    def name(self):
        """
        The name of the procedure

        This looks up the name in the linked :attr:`procedure` if available, otherwise
        returns the name stored during instanation of the :any:`ProcedureType` object.
        """
        return self._name if self.procedure is BasicType.DEFERRED else self.procedure.name

    @property
    def procedure(self):
        """
        The :any:`Subroutine` object of the procedure

        If not provided during instantiation or if the underlying :any:`weakref` is dead,
        this returns :any:`BasicType.DEFERRED`.
        """
        if self._procedure is None:
            return BasicType.DEFERRED
        if self._procedure() is None:
            return BasicType.DEFERRED
        return self._procedure()

    @property
    def parameters(self):
        """
        The tuple of procedure arguments, if :attr:`procedure` is available
        """
        if self.procedure is BasicType.DEFERRED:
            return tuple()
        return self.procedure.arguments

    @property
    def is_function(self):
        """
        Return `True` if the procedure is a function, otherwise `False`
        """
        if self.procedure is BasicType.DEFERRED:
            return self._is_function
        return self.procedure.is_function

    @property
    def return_type(self):
        """
        The return type of the function (or `None`)
        """
        if self.procedure is BasicType.DEFERRED:
            return self._return_type
        return self.procedure.return_type

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<ProcedureType {self.name}>'


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
                typename = 'Type' if isinstance(v, SymbolAttributes) else 'Variable'
                parameters += [f'parent={typename}({v.name})']
            else:
                parameters += [f'{k}={str(v)}']
        return f'<{self.__class__.__name__} {", ".join(parameters)}>'

    def __getinitargs__(self):
        args = [self.dtype]
        for k, v in self.__dict__.items():
            if k in ['dtype', 'source']:
                continue
            args += [(k, v)]
        return tuple(args)

    def __eq__(self, other):
        """
        Compare :any:`SymbolAttributes` via internal comparison but without execptions.
        """
        return self.compare(other, ignore=None)

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
