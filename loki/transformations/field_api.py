# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A set utility classes for dealing with FIELD API boilerplate in
parallel kernels and offload regions.
"""

from enum import Enum
from itertools import chain

from loki.expression import symbols as sym
from loki.ir import nodes as ir
from loki.scope import Scope


__all__ = [
    'FieldAPITransferType', 'FieldPointerMap', 'get_field_type',
    'field_get_device_data', 'field_sync_host'
]


class FieldAPITransferType(Enum):
    READ_ONLY = 1
    READ_WRITE = 2
    WRITE_ONLY = 3


class FieldPointerMap:
    """
    Helper class to map FIELD API pointers to intents and access descriptors.

    This utility is used to store arrays passed to target kernel calls
    and the corresponding device pointers added by the transformation.

    The pointer/array variable pairs are exposed through the class
    properties, based on the intent of the kernel argument.
    """
    def __init__(self, inargs, inoutargs, outargs, scope, ptr_prefix='loki_devptr_'):
        self.inargs = inargs
        self.inoutargs = inoutargs
        self.outargs = outargs

        self.scope = scope

        self.ptr_prefix = ptr_prefix

    def dataptr_from_array(self, a: sym.Array):
        """
        Returns a contiguous pointer :any:`Variable` with types matching the array :data:`a`.
        """
        shape = (sym.RangeIndex((None, None)),) * (len(a.shape)+1)
        devptr_type = a.type.clone(pointer=True, contiguous=True, shape=shape, intent=None)
        base_name = a.name if a.parent is None else '_'.join(a.name.split('%'))
        return sym.Variable(name=self.ptr_prefix + base_name, type=devptr_type, dimensions=shape)

    @staticmethod
    def field_ptr_from_view(field_view):
        """
        Returns a symbol for the pointer to the corresponding Field object.
        """
        type_chain = field_view.name.split('%')
        field_type_name = 'F_' + type_chain[-1]
        return field_view.parent.get_derived_type_member(field_type_name)

    @property
    def dataptrs(self):
        """ Create a list of contiguous data pointer symbols """
        return tuple(
            self.dataptr_from_array(a)
            for a in chain(*(self.inargs, self.inoutargs, self.outargs))
        )

    @property
    def host_to_device_calls(self):
        """
        Returns a tuple of :any:`CallStatement` for host-to-device transfers on fields.
        """
        READ_ONLY, READ_WRITE = FieldAPITransferType.READ_ONLY, FieldAPITransferType.READ_WRITE

        host_to_device = tuple(field_get_device_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_ONLY, scope=self.scope
        ) for arg in self.inargs)
        host_to_device += tuple(field_get_device_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, scope=self.scope
        ) for arg in self.inoutargs)
        host_to_device += tuple(field_get_device_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, scope=self.scope
        ) for arg in self.outargs)

        return host_to_device

    @property
    def sync_host_calls(self):
        """
        Returns a tuple of :any:`CallStatement` for host-synchronization transfers on fields.
        """
        sync_host = tuple(
            field_sync_host(self.field_ptr_from_view(arg), scope=self.scope) for arg in self.inoutargs
        )
        sync_host += tuple(
            field_sync_host(self.field_ptr_from_view(arg), scope=self.scope) for arg in self.outargs
        )
        return sync_host


def get_field_type(a: sym.Array) -> sym.DerivedType:
    """
    Returns the corresponding FIELD API type for an array.

    This function is IFS specific and assumes that the
    type is an array declared with one of the IFS type specifiers, e.g. KIND=JPRB
    """
    type_map = ["jprb",
                "jpit",
                "jpis",
                "jpim",
                "jpib",
                "jpia",
                "jprt",
                "jprs",
                "jprm",
                "jprd",
                "jplm"]
    type_name = a.type.kind.name

    assert type_name.lower() in type_map, ('Error array type kind is: '
                                           f'"{type_name}" which is not a valid IFS type specifier')
    rank = len(a.shape)
    field_type = sym.DerivedType(name="field_" + str(rank) + type_name[2:4].lower())
    return field_type


def field_get_device_data(field_ptr, dev_ptr, transfer_type: FieldAPITransferType, scope: Scope):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``GET_DEVICE_DATA`` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``GET_DEVICE_DATA`` from.
    dev_ptr: :any:`Array`
        Device pointer array
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which ``GET_DEVICE_DATA`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    """
    if not isinstance(transfer_type, FieldAPITransferType):
        raise TypeError(f"transfer_type must be of type FieldAPITransferType, but is of type {type(transfer_type)}")
    if transfer_type == FieldAPITransferType.READ_ONLY:
        suffix = 'RDONLY'
    elif transfer_type == FieldAPITransferType.READ_WRITE:
        suffix = 'RDWR'
    elif transfer_type == FieldAPITransferType.WRITE_ONLY:
        suffix = 'WRONLY'
    else:
        suffix = ''
    procedure_name = 'GET_DEVICE_DATA_' + suffix
    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope),
                            arguments=(dev_ptr.clone(dimensions=None),), )


def field_sync_host(field_ptr, scope):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``SYNC_HOST`` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``SYNC_HOST`` from.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    """

    procedure_name = 'SYNC_HOST_RDWR'
    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope), arguments=())
