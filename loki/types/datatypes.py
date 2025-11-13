# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of classes to represent type information for symbols used throughout
:doc:`internal_representation`
"""

import weakref
from enum import Enum

from loki.tools import flatten, LazyNodeLookup


__all__ = ['DataType', 'BasicType', 'ProcedureType', 'ModuleType']


class DataType:
    """
    Base class for data types a symbol may have
    """


class BasicType(DataType, int, Enum):
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
    is_intrinsic : bool, optional
        Indicate that this is an intrinsic function
    procedure : :any:`Subroutine` or :any:`StatementFunction` or :any:`LazyNodeLookup`, optional
        The procedure this type represents
    """

    def __init__(
            self, name=None, is_function=None, is_generic=False,
            is_intrinsic=False, procedure=None, return_type=None
    ):
        # pylint: disable=import-outside-toplevel,cyclic-import
        from loki.subroutine import Subroutine
        from loki.types.symbol_table import SymbolAttributes

        super().__init__()
        assert name or isinstance(procedure, Subroutine)
        assert isinstance(return_type, SymbolAttributes) or procedure or not is_function or is_intrinsic
        self.is_generic = is_generic
        self.is_intrinsic = is_intrinsic
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
            self._return_type = self.procedure.return_type if self.procedure.is_function else None

    @property
    def _canonical(self):
        return (self._name, self._procedure, self.is_function, self.is_generic, self.return_type)

    def __eq__(self, other):
        if isinstance(other, ProcedureType):
            return self._canonical == other._canonical
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._canonical)

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
    def is_elemental(self):
        """
        Return ``True`` if the procedure has the ``elemental`` prefix, otherwise ``False``
        """
        if self.procedure is BasicType.DEFERRED:
            return False
        if not hasattr(self.procedure, 'prefix'):
            # StatementFunction objects have no prefix!
            # This will be fixed once procedures are unified
            return False
        return 'elemental'.lower() in tuple(pre.lower() for pre in self.procedure.prefix)

    @property
    def return_type(self):
        """
        The return type of the function (or `None`)
        """
        if not self.is_function:
            return None
        if self.procedure is BasicType.DEFERRED:
            return self._return_type
        return self.procedure.return_type

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<ProcedureType {self.name}>'

    def __getstate__(self):
        _ignore = ('_procedure', )
        return dict((k, v) for k, v in self.__dict__.items() if k not in _ignore)

    def __setstate__(self, s):
        self.__dict__.update(s)

        self._procedure = None


class ModuleType(DataType):
    """
    Representation of a module definition.

    This serves as a caching mechanism for module definitions in symbol tables.

    Parameters
    ----------
    name : str, optional
        The name of the module. Can be skipped if :data:`module`
        is provided (not in the form of a :any:`LazyNodeLookup`)
    module : :any:`Module` :any:`LazyNodeLookup`, optional
        The procedure this type represents
    """

    def __init__(self, name=None, module=None):
        from loki.module import Module  # pylint: disable=import-outside-toplevel,cyclic-import
        super().__init__()
        assert name or isinstance(module, Module)
        if module is None or isinstance(module, LazyNodeLookup):
            self._module = module
            self._name = name
        else:
            self._module = weakref.ref(module)
            # Cache all properties for when module link becomes inactive
            assert name is None or name.lower() == self.module.name.lower()
            self._name = self.module.name

    @property
    def name(self):
        """
        The name of the module

        This looks up the name in the linked :attr:`module` if available, otherwise
        returns the name stored during instantiation of the :any:`ModuleType` object.
        """
        return self._name if self.module is BasicType.DEFERRED else self.module.name

    @property
    def module(self):
        """
        The :any:`Module` object represented by this type

        If not provided during instantiation or if the underlying :any:`weakref` is dead,
        this returns :any:`BasicType.DEFERRED`.
        """
        if self._module is None:
            return BasicType.DEFERRED
        if self._module() is None:
            return BasicType.DEFERRED
        return self._module()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<ModuleType {self.name}>'
