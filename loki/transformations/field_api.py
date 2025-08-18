# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A set of utility classes for dealing with FIELD API boilerplate in
parallel kernels and offload regions.
"""

from enum import Enum
from itertools import chain

from loki.expression import symbols as sym
from loki.ir import nodes as ir
from loki.scope import Scope


__all__ = [
    'FieldAPITransferType', 'FieldAPIDestination', 'FieldPointerMap',
    'get_field_type', 'field_sync_host',
    'field_get_host_data', 'field_delete_device_data'
]


class FieldAPITransferType(Enum):
    READ_ONLY = 1
    READ_WRITE = 2
    WRITE_ONLY = 3


class FieldAPIDestination(Enum):
    HOST = 1
    DEVICE = 2


class FieldPointerMap:
    """
    Helper class to map FIELD API pointers to intents and access descriptors.

    This utility is used to store arrays passed to target kernel calls
    and easily access corresponding device pointers added by the transformation.
    """
    def __init__(
            self, inargs, inoutargs, outargs, scope, ptr_prefix='loki_ptr_', arr_prefix=''
    ):
        # Ensure no duplication between in/inout/out args
        inoutargs += tuple(v for v in inargs if v in outargs)
        inargs = tuple(v for v in inargs if v not in inoutargs)
        outargs = tuple(v for v in outargs if v not in inoutargs)

        # Filter out duplicates and return as tuple
        self.inargs = tuple(dict.fromkeys(a.clone(dimensions=None) for a in inargs))
        self.inoutargs = tuple(dict.fromkeys(a.clone(dimensions=None) for a in inoutargs))
        self.outargs = tuple(dict.fromkeys(a.clone(dimensions=None) for a in outargs))

        # Simplistic sort of pointer variables
        self.inargs = sorted(self.inargs, key=str)
        self.inoutargs = sorted(self.inoutargs, key=str)
        self.outargs = sorted(self.outargs, key=str)

        # Filter out duplicates across argument tuples
        self.inargs = tuple(a for a in self.inargs if a not in self.inoutargs)

        self.scope = scope

        self.ptr_prefix = ptr_prefix
        self.arr_prefix = arr_prefix

    def dataptr_from_array(self, a: sym.Array):
        """
        Returns a contiguous pointer :any:`Variable` with types matching the array :data:`a`.
        """
        shape = (sym.RangeIndex((None, None)),) * (len(a.shape)+1)
        dataptr_type = a.type.clone(pointer=True, contiguous=True, shape=shape, intent=None)
        base_name = a.name if a.parent is None else '_'.join(a.name.split('%'))
        return sym.Variable(name=self.ptr_prefix + base_name, type=dataptr_type, dimensions=shape)

    def field_ptr_from_view(self, field_view):
        """
        Returns a symbol for the pointer to the corresponding Field object.
        """
        basename = field_view.basename
        if self.arr_prefix and basename.startswith(self.arr_prefix):
            basename = basename[len(self.arr_prefix):]
        field_type_name = 'F_' + basename
        return field_view.parent.get_derived_type_member(field_type_name)

    @property
    def args(self):
        """ A tuple of all argument symbols, concatanating in/inout/out arguments """
        return tuple(chain(*(self.inargs, self.inoutargs, self.outargs)))

    @property
    def dataptrs(self):
        """ Create a list of contiguous data pointer symbols """
        return tuple(dict.fromkeys(self.dataptr_from_array(a) for a in self.args))

    @property
    def host_to_device_calls(self):
        """
        Returns a tuple of :any:`CallStatement` for host-to-device transfers on fields.
        """
        READ_ONLY, READ_WRITE = FieldAPITransferType.READ_ONLY, FieldAPITransferType.READ_WRITE
        DEVICE = FieldAPIDestination.DEVICE

        host_to_device = tuple(field_get_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_ONLY, DEVICE, scope=self.scope
        ) for arg in self.inargs)
        host_to_device += tuple(field_get_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, DEVICE, scope=self.scope
        ) for arg in self.inoutargs)
        host_to_device += tuple(field_get_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, DEVICE, scope=self.scope
        ) for arg in self.outargs)

        return tuple(dict.fromkeys(host_to_device))

    @property
    def device_to_host_calls(self):
        """
        Returns a tuple of :any:`CallStatement` for device-to-device transfers on fields.
        """
        READ_ONLY, READ_WRITE = FieldAPITransferType.READ_ONLY, FieldAPITransferType.READ_WRITE
        HOST = FieldAPIDestination.HOST

        host_to_device = tuple(field_get_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_ONLY, HOST, scope=self.scope
        ) for arg in self.inargs)
        host_to_device += tuple(field_get_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, HOST, scope=self.scope
        ) for arg in self.inoutargs)
        host_to_device += tuple(field_get_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, HOST, scope=self.scope
        ) for arg in self.outargs)

        return tuple(dict.fromkeys(host_to_device))

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
        return tuple(dict.fromkeys(sync_host))


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


def field_get_data(
        field_ptr, dataptr, transfer_type: FieldAPITransferType,
        dest: FieldAPIDestination, scope: Scope
):
    """
    Utility function to generate a :any:`CallStatement` corresponding
    to a Field API ``GET_DEVICE_DATA`` or ``GET_HOST_DATA`` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``GET_DEVICE_DATA`` from.
    dataptr: :any:`Array`
        Device or host pointer symbol
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which ``GET_DEVICE_DATA`` method to call.
    transfer_dest: :any:`FieldAPIDestination`
        Transfer destination, either ``HOST`` or ``DEVICE``
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    """
    assert isinstance(dest, FieldAPIDestination)

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
    procedure_dest = 'HOST' if dest == FieldAPIDestination.HOST else 'DEVICE'
    procedure_name = f'GET_{procedure_dest}_DATA_' + suffix
    return ir.CallStatement(
        name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope),
        arguments=(dataptr.clone(dimensions=None),),
    )


def field_get_host_data(field_ptr, host_ptr, transfer_type: FieldAPITransferType, scope: Scope):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``GET_HOST_DATA`` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``GET_HOST_DATA`` from.
    host_ptr: :any:`Array`
        Host pointer array
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which ``GET_HOST_DATA`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    """
    if not isinstance(transfer_type, FieldAPITransferType):
        raise TypeError(f"transfer_type must be of type FieldAPITransferType, but is of type {type(transfer_type)}")
    if transfer_type == FieldAPITransferType.READ_ONLY:
        suffix = 'RDONLY'
    elif transfer_type == FieldAPITransferType.READ_WRITE:
        suffix = 'RDWR'
    else:
        suffix = ''
    procedure_name = 'GET_HOST_DATA_' + suffix
    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope),
                            arguments=(host_ptr.clone(dimensions=None),), )


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


def field_delete_device_data(field_ptr, scope):
    """
    Utility unction to generate a :any:`CallStatement` corresponding to a Field API
    `DELETE_DEVICE_DATA` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``DELETE_DEVICE_DATA`` from.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    """

    procedure_name = 'DELETE_DEVICE_DATA'
    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope), arguments=())
