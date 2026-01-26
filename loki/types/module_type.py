# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Representation of a module type for module definitions. """

import weakref

from loki.tools import LazyNodeLookup
from loki.types.datatypes import BasicType, DataType


__all__ = ['ModuleType']


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
