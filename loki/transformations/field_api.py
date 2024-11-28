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
    def __init__(self, devptrs, inargs, inoutargs, outargs):
        self.inargs = inargs
        self.inoutargs = inoutargs
        self.outargs = outargs
        self.devptrs = devptrs


    @property
    def in_pairs(self):
        """
        Iterator that yields array/pointer pairs for kernel arguments of intent(in).

        Yields
        ______
        :any:`Array`
            Original kernel call argument
        :any:`Array`
            Corresponding device pointer added by the transformation.
        """
        for i, inarg in enumerate(self.inargs):
            yield inarg, self.devptrs[i]

    @property
    def inout_pairs(self):
        """
        Iterator that yields array/pointer pairs for arguments with intent(inout).

        Yields
        ______
        :any:`Array`
            Original kernel call argument
        :any:`Array`
            Corresponding device pointer added by the transformation.
        """
        start = len(self.inargs)
        for i, inoutarg in enumerate(self.inoutargs):
            yield inoutarg, self.devptrs[i+start]

    @property
    def out_pairs(self):
        """
        Iterator that yields array/pointer pairs for arguments with intent(out)

        Yields
        ______
        :any:`Array`
            Original kernel call argument
        :any:`Array`
            Corresponding device pointer added by the transformation.
        """

        start = len(self.inargs)+len(self.inoutargs)
        for i, outarg in enumerate(self.outargs):
            yield outarg, self.devptrs[i+start]


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
