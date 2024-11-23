# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from itertools import chain

from loki.batch import Transformation
from loki.expression import Array, symbols as sym
from loki.ir import (
    FindNodes, PragmaRegion, CallStatement,
    Transformer, pragma_regions_attached,
    SubstituteExpressions
)
from loki.logging import warning, error
from loki.tools import as_tuple
from loki.types import BasicType

from loki.transformations.data_offload.offload import DataOffloadTransformation
from loki.transformations.parallel import (
    FieldAPITransferType, field_get_device_data, field_sync_host,
    remove_field_api_view_updates
)


__all__ = ['FieldOffloadTransformation', 'FieldPointerMap']


def find_target_calls(region, targets):
    """
    Returns a list of all calls to targets inside the region.

    Parameters
    ----------
    :region: :any:`PragmaRegion`
    :targets: collection of :any:`Subroutine`
        Iterable object of subroutines or functions called
    :returns: list of :any:`CallStatement`
    """
    calls = FindNodes(CallStatement).visit(region)
    calls = [c for c in calls if str(c.name).lower() in targets]
    return calls


class FieldOffloadTransformation(Transformation):
    """

    Transformation to offload arrays owned by Field API fields to the device. **This transformation is IFS specific.**

    The transformation assumes that fields are wrapped in derived types specified in
    ``field_group_types`` and will only offload arrays that are members of such derived types.
    In the process this transformation removes calls to Field API ``update_view`` and adds
    declarations for the device pointers to the driver subroutine.

    The transformation acts on ``!$loki data`` regions and offloads all :any:`Array`
    symbols that satisfy the following conditions:

    1. The array is a member of an object that is of type specified in ``field_group_types``.
    
    2. The array is passed as a parameter to at least one of the kernel targets passed to ``transform_subroutine``.

    Parameters
    ----------
    devptr_prefix: str, optional
        The prefix of device pointers added by this transformation (defaults to ``'loki_devptr_'``).
    field_group_types: list or tuple of str, optional
        Names of the field group types with members that may be offloaded (defaults to ``['']``).
    offload_index: str, optional
        Names of index variable to inject in the outmost dimension of offloaded arrays in the kernel
        calls (defaults to ``'IBL'``).
    """

    def __init__(self, devptr_prefix=None, field_group_types=None, offload_index=None):
        self.deviceptr_prefix = 'loki_devptr_' if devptr_prefix is None else devptr_prefix
        field_group_types = [''] if field_group_types is None else field_group_types
        self.field_group_types = tuple(typename.lower() for typename in field_group_types)
        self.offload_index = 'IBL' if offload_index is None else offload_index

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs['role']
        targets = as_tuple(kwargs.get('targets'), (None))
        if role == 'driver':
            self.process_driver(routine, targets)

    def process_driver(self, driver, targets):
        remove_field_api_view_updates(driver, self.field_group_types)
        with pragma_regions_attached(driver):
            for region in FindNodes(PragmaRegion).visit(driver.body):
                # Only work on active `!$loki data` regions
                if not DataOffloadTransformation._is_active_loki_data_region(region, targets):
                    continue
                kernel_calls = find_target_calls(region, targets)
                offload_variables = self.find_offload_variables(driver, kernel_calls)
                device_ptrs = self._declare_device_ptrs(driver, offload_variables)
                offload_map = FieldPointerMap(device_ptrs, *offload_variables)
                self._add_field_offload_calls(driver, region, offload_map)
                self._replace_kernel_args(driver, kernel_calls, offload_map)

    def find_offload_variables(self, driver, calls):
        inargs = ()
        inoutargs = ()
        outargs = ()

        for call in calls:
            if call.routine is BasicType.DEFERRED:
                error(f'[Loki] Data offload: Routine {driver.name} has not been enriched ' +
                        f'in {str(call.name).lower()}')
                raise RuntimeError
            for param, arg in call.arg_iter():
                if not isinstance(param, Array):
                    continue
                try:
                    parent = arg.parent
                    if parent.type.dtype.name.lower() not in self.field_group_types:
                        warning(f'[Loki] Data offload: The parent object {parent.name} of type ' +
                                f'{parent.type.dtype} is not in the list of field wrapper types')
                        continue
                except AttributeError:
                    warning(f'[Loki] Data offload: Raw array object {arg.name} encountered in'
                            + f' {driver.name} that is not wrapped by a Field API object')
                    continue

                if param.type.intent.lower() == 'in':
                    inargs += (arg, )
                if param.type.intent.lower() == 'inout':
                    inoutargs += (arg, )
                if param.type.intent.lower() == 'out':
                    outargs += (arg, )

        inoutargs += tuple(v for v in inargs if v in outargs)
        inargs = tuple(v for v in inargs if v not in inoutargs)
        outargs = tuple(v for v in outargs if v not in inoutargs)

        # Filter out duplicates and return as tuple
        inargs = tuple(dict.fromkeys(inargs))
        inoutargs = tuple(dict.fromkeys(inoutargs))
        outargs = tuple(dict.fromkeys(outargs))

        return inargs, inoutargs, outargs

    def _declare_device_ptrs(self, driver, offload_variables):
        device_ptrs = tuple(self._devptr_from_array(driver, a) for a in chain(*offload_variables))
        driver.variables += device_ptrs
        return device_ptrs

    def _devptr_from_array(self, driver, a: sym.Array):
        """
        Returns a contiguous pointer :any:`Variable` with types matching the array a
        """
        shape = (sym.RangeIndex((None, None)),) * (len(a.shape)+1)
        devptr_type = a.type.clone(pointer=True, contiguous=True, shape=shape, intent=None)
        base_name = a.name if a.parent is None else '_'.join(a.name.split('%'))
        devptr_name = self.deviceptr_prefix + base_name
        if devptr_name in driver.variable_map:
            warning(f'[Loki] Data offload: The routine {driver.name} already has a ' +
                    f'variable named {devptr_name}')
        devptr = sym.Variable(name=devptr_name, type=devptr_type, dimensions=shape)
        return devptr

    def _add_field_offload_calls(self, driver, region, offload_map):
        host_to_device = tuple(field_get_device_data(self._get_field_ptr_from_view(inarg), devptr,
                               FieldAPITransferType.READ_ONLY, driver) for inarg, devptr in offload_map.in_pairs)
        host_to_device += tuple(field_get_device_data(self._get_field_ptr_from_view(inarg), devptr,
                                FieldAPITransferType.READ_WRITE, driver) for inarg, devptr in offload_map.inout_pairs)
        host_to_device += tuple(field_get_device_data(self._get_field_ptr_from_view(inarg), devptr,
                                FieldAPITransferType.READ_WRITE, driver) for inarg, devptr in offload_map.out_pairs)
        device_to_host = tuple(field_sync_host(self._get_field_ptr_from_view(inarg), driver)
                               for inarg, _ in chain(offload_map.inout_pairs, offload_map.out_pairs))
        update_map = {region: host_to_device + (region,) + device_to_host}
        Transformer(update_map, inplace=True).visit(driver.body)

    def _get_field_ptr_from_view(self, field_view):
        type_chain = field_view.name.split('%')
        field_type_name = 'F_' + type_chain[-1]
        return field_view.parent.get_derived_type_member(field_type_name)

    def _replace_kernel_args(self, driver, kernel_calls, offload_map):
        change_map = {}
        offload_idx_expr = driver.variable_map[self.offload_index]
        for arg, devptr in chain(offload_map.in_pairs, offload_map.inout_pairs, offload_map.out_pairs):
            if len(arg.dimensions) != 0:
                dims = arg.dimensions + (offload_idx_expr,)
            else:
                dims = (sym.RangeIndex((None, None)),) * (len(devptr.shape)-1) + (offload_idx_expr,)
            change_map[arg] = devptr.clone(dimensions=dims)

        arg_transformer = SubstituteExpressions(change_map, inplace=True)
        for call in kernel_calls:
            arg_transformer.visit(call)


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
