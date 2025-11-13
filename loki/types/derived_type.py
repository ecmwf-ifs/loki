# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Representation of a derived type with local symbol table. """

from loki.types.datatypes import BasicType, DataType
from loki.types.scope import Scope


__all__ = ['DerivedType']


class DerivedType(DataType, Scope):
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

    def __init__(self, name=None, typedef=None, scope=None):
        super().__init__(parent=scope)
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

    @property
    def _canonical(self):
        return (self._name, self.typedef)

    def __eq__(self, other):
        if isinstance(other, DerivedType):
            return self._canonical == other._canonical
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._canonical)
