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
    SubstituteExpressions, FindVariables
)
from loki.logging import warning, error
from loki.tools import as_tuple
from loki.types import BasicType

from loki.transformations.data_offload.offload import DataOffloadTransformation
from loki.transformations.field_api import FieldPointerMap
from loki.transformations.parallel import remove_field_api_view_updates



__all__ = [
    'FieldOffloadTransformation', 'find_target_calls',
    'find_offload_variables', 'add_field_offload_calls',
    'replace_kernel_args'
]


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
                offload_variables = find_offload_variables(driver, kernel_calls, self.field_group_types)
                offload_map = FieldPointerMap(
                    *offload_variables, scope=driver, ptr_prefix=self.deviceptr_prefix
                )
                declare_device_ptrs(driver, deviceptrs=offload_map.dataptrs)
                add_field_offload_calls(driver, region, offload_map)
                replace_kernel_args(driver, offload_map, self.offload_index)


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


def find_offload_variables(driver, calls, field_group_types):
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
                if parent.type.dtype.name.lower() not in field_group_types:
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

    return inargs, inoutargs, outargs


def declare_device_ptrs(driver, deviceptrs):
    """
    Add a set of data pointer declarations to a given :any:`Subroutine`
    """
    for devptr in deviceptrs:
        if devptr.name in driver.variable_map:
            warning(f'[Loki] Data offload: The routine {driver.name} already has a ' +
                    f'variable named {devptr.name}')

    driver.variables += deviceptrs


def add_field_offload_calls(driver, region, offload_map):

    update_map = {
        region: offload_map.host_to_device_calls + (region,) + offload_map.sync_host_calls
    }
    Transformer(update_map, inplace=True).visit(driver.body)


def replace_kernel_args(driver, offload_map, offload_index):
    change_map = {}
    offload_idx_expr = driver.variable_map[offload_index]

    args = tuple(chain(offload_map.inargs, offload_map.inoutargs, offload_map.outargs))
    for arg in FindVariables().visit(driver.body):
        if not arg.name in args:
            continue

        devptr = offload_map.dataptr_from_array(arg)
        if len(arg.dimensions) != 0:
            dims = arg.dimensions + (offload_idx_expr,)
        else:
            dims = (sym.RangeIndex((None, None)),) * (len(devptr.shape)-1) + (offload_idx_expr,)
        change_map[arg] = devptr.clone(dimensions=dims)

    driver.body = SubstituteExpressions(change_map, inplace=True).visit(driver.body)
