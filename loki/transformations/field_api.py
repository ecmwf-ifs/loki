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
from loki.types import Scope


__all__ = [
    'FieldAPITransferType', 'FieldPointerMap', 'get_field_type', 'field_get_device_data',
    'field_get_host_data', 'field_sync_device', 'field_sync_host', 'field_create_device_data',
    'field_delete_device_data', 'field_wait_for_async_queue', 'FieldAPIAccessorType'
]


class FieldAPITransferType(Enum):
    READ_ONLY = 'RDONLY'
    READ_WRITE = 'RDWR'
    WRITE_ONLY = 'WRONLY'
    FORCE = 'FORCE'

    @property
    def suffix(self):
        return self.value


class FieldAPITransferDirection(Enum):
    DEVICE_TO_HOST = 'HOST'
    HOST_TO_DEVICE = 'DEVICE'

    @property
    def suffix(self):
        return self.value


class FieldAPIAccessorType(Enum):

    """
    Create FIELD_API data access calls using the native type-bound methods e.g.
    CALL FIELD%GET_HOST/DEVICE_DATA_...()
    """
    TYPE_BOUND = 'GET'

    """
    Create FIELD_API data access calls using a generic interface that
    takes a FIELD as an argument e.g.
    CALL SGET_HOST/DEVICE_DATA_...(..., FIELD)

    This mode offers additional safety for uninitialised and zero-sized fields.
    """
    GENERIC = 'SGET'

    def __str__(self):
        return self.value


class FieldPointerMap:
    """
    Helper class to map FIELD API pointers to intents and access descriptors.

    This utility is used to store arrays passed to target kernel calls
    and easily access corresponding device pointers added by the transformation.
    """

    def __init__(self, inargs, inoutargs, outargs, scope, ptr_prefix='loki_ptr_'):
        # Ensure no duplication between in/inout/out args
        inoutargs += tuple(v for v in inargs if v in outargs)
        inargs = tuple(v for v in inargs if v not in inoutargs)
        outargs = tuple(v for v in outargs if v not in inoutargs)

        # Filter out duplicates and return as tuple
        self.inargs = tuple(dict.fromkeys(a.clone(dimensions=None) for a in inargs))
        self.inoutargs = tuple(dict.fromkeys(a.clone(dimensions=None) for a in inoutargs))
        self.outargs = tuple(dict.fromkeys(a.clone(dimensions=None) for a in outargs))

        # Filter out duplicates across argument tuples
        self.inargs = tuple(a for a in self.inargs if a not in self.inoutargs)

        self.scope = scope

        self.ptr_prefix = ptr_prefix

    def dataptr_from_array(self, a: sym.Array):
        """
        Returns a contiguous pointer :any:`Variable` with types matching the array :data:`a`.
        """
        shape = (sym.RangeIndex((None, None)),) * (len(a.shape)+1)
        dataptr_type = a.type.clone(pointer=True, contiguous=True, shape=shape, intent=None)
        base_name = a.name if a.parent is None else '_'.join(a.name.split('%'))
        return sym.Variable(name=self.ptr_prefix + base_name, type=dataptr_type, dimensions=shape)

    @staticmethod
    def field_ptr_from_view(field_view):
        """
        Returns a symbol for the pointer to the corresponding Field object.
        """
        type_chain = field_view.name.split('%')
        field_type_name = 'F_' + type_chain[-1]
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

        host_to_device = tuple(field_get_device_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_ONLY, scope=self.scope
        ) for arg in self.inargs)
        host_to_device += tuple(field_get_device_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, scope=self.scope
        ) for arg in self.inoutargs)
        host_to_device += tuple(field_get_device_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), READ_WRITE, scope=self.scope
        ) for arg in self.outargs)

        return tuple(dict.fromkeys(host_to_device))

    @property
    def sync_host_calls(self):
        """
        Returns a tuple of :any:`CallStatement` for host-synchronization transfers on fields.
        """
        READ_WRITE = FieldAPITransferType.READ_WRITE

        sync_host = tuple(
            field_sync_host(self.field_ptr_from_view(arg), transfer_type=READ_WRITE, scope=self.scope)
            for arg in self.inoutargs
        )
        sync_host += tuple(
            field_sync_host(self.field_ptr_from_view(arg), transfer_type=READ_WRITE, scope=self.scope)
            for arg in self.outargs
        )
        return tuple(dict.fromkeys(sync_host))

    def host_to_device_force_calls(self, queue=None, blk_bounds=None, offset=None):
        """
        Returns a tuple of :any:`CallStatement` for host-to-device force transfers on fields.
        """
        FORCE = FieldAPITransferType.FORCE
        host_to_device = tuple(field_get_device_data(
            self.field_ptr_from_view(arg), self.dataptr_from_array(arg), transfer_type=FORCE,
            scope=self.scope, queue=queue, blk_bounds=blk_bounds, offset=offset)
                               for arg in chain(self.inargs, self.inoutargs, self.outargs))
        return tuple(dict.fromkeys(host_to_device))


    def sync_host_force_calls(self, queue=None, blk_bounds=None, offset=None):
        """
        Returns a tuple of :any:`CallStatement` for host-synchronization transfers on fields.
        """
        FORCE = FieldAPITransferType.FORCE

        sync_host = tuple(field_sync_host(
            self.field_ptr_from_view(arg), transfer_type=FORCE, scope=self.scope,
            queue=queue, blk_bounds=blk_bounds, offset=offset)
                          for arg in chain(self.inoutargs, self.outargs))
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


def _field_get_data(field_ptr, dev_ptr, transfer_type: FieldAPITransferType,
                    transfer_direction: FieldAPITransferDirection,
                    scope: Scope, queue=None, blk_bounds=None, offset=None,
                    accessor_type: FieldAPIAccessorType=FieldAPIAccessorType.TYPE_BOUND):
    """
    Internal function to generate FIELD API ``GET DATA`` calls.

    .. note::
        This routine is not meant to be called from any code outisde `field_api.py`, then the
        corresponding :any:`field_get_device_data` or :any:`field_get_host_data` functions
        should be called instead.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``GET_DEVICE_DATA`` from.
    dev_ptr: :any:`Array`
        Device pointer array
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which type of ``GET DATA`` method to call.
    transfer_direction: :any:`FieldAPITransferDirection`
        Field API transfer direction to determine which type of ``GET DATA`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    queue: integer
       ``QUEUE`` optional  argument
    blk_bounds: integer dimension(2) array
        ``BLK_BOUNDS`` optional argument
    offset: integer
        ``OFFSET`` optional argument
    accessor_type: :any:`FieldAPIAccessorType`
        Type of accessor to be used, e.g., 'get_' type bound method or 'sget'
    """
    if not isinstance(transfer_type, FieldAPITransferType):
        raise TypeError("transfer_type must be of type FieldAPITransferType, " +
                        f"but is of type {type(transfer_type)}")
    if not isinstance(transfer_direction, FieldAPITransferDirection):
        raise TypeError("transfer_direction must be of type FieldAPITransferDirection, " +
                        f"but is of type {type(transfer_direction)}")

    if transfer_type != FieldAPITransferType.FORCE and (queue is not None or blk_bounds is not None):
        raise ValueError("Only force copy methods can have non-None type queue or blk_bounds")
    if (transfer_type == FieldAPITransferType.WRITE_ONLY and
        transfer_direction == FieldAPITransferDirection.DEVICE_TO_HOST
    ):
        raise TypeError("incorrect transfer_type (WRITE_ONLY) for Field-API get method")

    procedure_name = f'{accessor_type}_' + transfer_direction.suffix + '_DATA_' + transfer_type.suffix

    kwargs = []
    if queue is not None:
        kwargs.append(('queue', queue))
    if blk_bounds is not None:
        kwargs.append(('blk_bounds', blk_bounds))
    if offset is not None:
        kwargs.append(('offset', offset))

    kwargs = tuple(kwargs) if len(kwargs) > 0 else None

    if accessor_type == FieldAPIAccessorType.TYPE_BOUND:
        return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope),
                                arguments=(dev_ptr.clone(dimensions=None),), kwarguments=kwargs)
    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, scope=scope),
            arguments=(dev_ptr.clone(dimensions=None), field_ptr), kwarguments=kwargs)


def field_get_device_data(field_ptr, dev_ptr, transfer_type: FieldAPITransferType, scope: Scope,
                          queue=None, blk_bounds=None, offset=None,
                          accessor_type: FieldAPIAccessorType=FieldAPIAccessorType.TYPE_BOUND):
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
    queue: integer
       ``QUEUE`` optional  argument
    blk_bounds: integer dimension(2) array
        ``BLK_BOUNDS`` optional argument
    offset: integer
        ``OFFSET`` optional argument
    accessor_type: :any:`FieldAPIAccessorType`
        Type of accessor to be used, e.g., 'get_' type bound method or 'sget'
    """
    return _field_get_data(field_ptr, dev_ptr, transfer_type, FieldAPITransferDirection.HOST_TO_DEVICE,
                           scope, queue=queue, blk_bounds=blk_bounds, offset=offset, accessor_type=accessor_type)


def field_get_host_data(field_ptr, dev_ptr, transfer_type: FieldAPITransferType, scope: Scope,
                        queue=None, blk_bounds=None, offset=None,
                        accessor_type: FieldAPIAccessorType=FieldAPIAccessorType.TYPE_BOUND):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``GET_HOST_DATA`` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``GET_DEVICE_DATA`` from.
    dev_ptr: :any:`Array`
        Device pointer array
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which ``GET_HOST_DATA`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    queue: integer
       ``QUEUE`` optional  argument
    blk_bounds: integer dimension(2) array
        ``BLK_BOUNDS`` optional argument
    offset: integer
        ``OFFSET`` optional argument
    accessor_type: :any:`FieldAPIAccessorType`
        Type of accessor to be used, e.g., 'get_' type bound method or 'sget'
    """
    return _field_get_data(field_ptr, dev_ptr, transfer_type, FieldAPITransferDirection.DEVICE_TO_HOST,
                           scope, queue=queue, blk_bounds=blk_bounds, offset=offset, accessor_type=accessor_type)


def _field_sync(field_ptr, transfer_type: FieldAPITransferType,
                transfer_direction: FieldAPITransferDirection,
                scope: Scope, queue=None, blk_bounds=None, offset=None):
    """
    Internal function to generate FIELD API ``SYNC`` calls.

    .. note::
        This routine is not meant to be called from any code outisde `field_api.py`, then the
        corresponding :any:`field_sync_host` or :any:`field_sync_device` functions should be
        called instead.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``GET_DEVICE_DATA`` from.
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which type of ``GET DATA`` method to call.
    transfer_direction: :any:`FieldAPITransferDirection`
        Field API transfer direction to determine which type of ``GET DATA`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    queue: integer
       ``QUEUE`` optional  argument
    blk_bounds: integer dimension(2) array
        ``BLK_BOUNDS`` optional argument
    offset: integer
        ``OFFSET`` optional argument
    """

    if not isinstance(transfer_type, FieldAPITransferType):
        raise TypeError("transfer_type must be of type FieldAPITransferType, " +
                        f"but is of type {type(transfer_type)}")
    if not isinstance(transfer_direction, FieldAPITransferDirection):
        raise TypeError("transfer_direction must be of type FieldAPITransferDirection, " +
                        f"but is of type {type(transfer_direction)}")

    if transfer_type != FieldAPITransferType.FORCE and (queue is not None or blk_bounds is not None):
        raise ValueError("Only force copy methods can have non-None type queue or blk_bounds")

    if (
        transfer_type == FieldAPITransferType.WRITE_ONLY and
        transfer_direction == FieldAPITransferDirection.DEVICE_TO_HOST
    ):
        raise TypeError("incorrect transfer_type for Field-API sync method")

    procedure_name = 'SYNC_' + transfer_direction.suffix + '_' + transfer_type.suffix

    kwargs = []
    if queue is not None:
        kwargs.append(('queue', queue))
    if blk_bounds is not None:
        kwargs.append(('blk_bounds', blk_bounds))
    if offset is not None:
        kwargs.append(('offset', offset))
    kwargs = tuple(kwargs) if len(kwargs) > 0 else None

    return ir.CallStatement(name=sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope),
                            kwarguments=kwargs)


def field_sync_device(field_ptr, transfer_type: FieldAPITransferType, scope: Scope,
                      queue=None, blk_bounds=None, offset=None):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``SYNC_DEVICE`` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``SYNC_HOST`` from.
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which ``SYNC_DEVICE`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    queue: integer
       ``QUEUE`` optional  argument
    blk_bounds: integer dimension(2) array
        ``BLK_BOUNDS`` optional argument
    offset: integer
        ``OFFSET`` optional argument
    """

    return _field_sync(field_ptr, transfer_type, FieldAPITransferDirection.HOST_TO_DEVICE,
                       scope, queue=queue, blk_bounds=blk_bounds, offset=offset)


def field_sync_host(field_ptr, transfer_type: FieldAPITransferType, scope: Scope,
                    queue=None, blk_bounds=None, offset=None):
    """
    Utility function to generate a :any:`CallStatement` corresponding to a Field API
    ``SYNC_HOST`` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``SYNC_HOST`` from.
    transfer_type: :any:`FieldAPITransferType`
        Field API transfer type to determine which ``SYNC_HOST`` method to call.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    queue: integer
       ``QUEUE`` optional  argument
    blk_bounds: integer dimension(2) array
        ``BLK_BOUNDS`` optional argument
    offset: integer
        ``OFFSET`` optional argument
    """

    return _field_sync(field_ptr, transfer_type, FieldAPITransferDirection.DEVICE_TO_HOST,
                       scope, queue=queue, blk_bounds=blk_bounds, offset=offset)


def field_create_device_data(field_ptr, scope: Scope, blk_bounds=None):
    """
    Utility unction to generate a :any:`CallStatement` corresponding to a Field API
    `CREATE_DEVICE_DATA` call.

    Parameters
    ----------
    field_ptr: pointer to field object
        Pointer to the field to call ``DELETE_DEVICE_DATA`` from.
    scope: :any:`Scope`
        Scope of the created :any:`CallStatement`
    blk_bounds: integer dimension(2) array
        ``BLK_BOUNDS`` optional argument
    """
    kwargs = (('blk_bounds', blk_bounds),) if blk_bounds else None
    return ir.CallStatement(name=sym.ProcedureSymbol('CREATE_DEVICE_DATA', parent=field_ptr, scope=scope),
                            kwarguments=kwargs)


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


def field_wait_for_async_queue(queue, scope: Scope):
    return ir.CallStatement(name=sym.ProcedureSymbol('WAIT_FOR_ASYNC_QUEUE', scope=scope),
                            arguments=(queue,))
