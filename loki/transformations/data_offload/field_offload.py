# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation
from loki.expression import Array, symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer,
    SubstituteExpressions, pragma_regions_attached, is_loki_pragma
)
from loki.logging import warning
from loki.types import BasicType

from loki.transformations.field_api import FieldPointerMap
from loki.transformations.parallel import do_remove_field_api_view_updates


__all__ = [
    'FieldOffloadTransformation', 'find_offload_variables',
    'add_field_offload_calls', 'replace_kernel_args'
]


class FieldOffloadTransformation(Transformation):
    """

    Transformation to offload arrays owned by Field API fields to the device.

    **This transformation is IFS specific.**

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
        if role == 'driver':
            self.process_driver(routine)

    def process_driver(self, driver):

        # Remove the Field-API view-pointer boilerplate
        do_remove_field_api_view_updates(driver, self.field_group_types)

        with pragma_regions_attached(driver):
            with dataflow_analysis_attached(driver):
                for region in FindNodes(ir.PragmaRegion).visit(driver.body):
                    # Only work on active `!$loki data` regions
                    if not region.pragma or not is_loki_pragma(region.pragma, starts_with='data'):
                        continue

                    # Determine the array variables for generating Field API offload
                    offload_variables = find_offload_variables(driver, region, self.field_group_types)
                    offload_map = FieldPointerMap(
                        *offload_variables, scope=driver, ptr_prefix=self.deviceptr_prefix
                    )

                    # Inject declarations and offload API calls into driver region
                    declare_device_ptrs(driver, deviceptrs=offload_map.dataptrs)
                    add_field_offload_calls(driver, region, offload_map)
                    replace_kernel_args(driver, offload_map, self.offload_index)


def find_offload_variables(driver, region, field_group_types):
    """
    Finds the sets of array variable symbols for which we can generate
    Field API offload code.

    Note
    ----
    This method requires Loki's dataflow analysis to be run on the
    :data:`region` via :meth:`dataflow_analysis_attached`.

    Parameters
    ----------
    region : :any:`PragmaRegion`
        Code region object for which to determine offload variables
    field_group_types : list or tuple of str, optional
        Names of the field group types with members that may be offloaded (defaults to ``['']``).

    Returns
    -------
    (inargs, inoutargs, outargs) : (tuple, tuple, tuple)
        The sets of array symbols split into three tuples according to access type.
    """

    # Use dataflow analysis to find in, out and inout variables to that region
    inargs = region.uses_symbols - region.defines_symbols
    inoutargs = region.uses_symbols & region.defines_symbols
    outargs = region.defines_symbols - region.uses_symbols

    # Filter out relevant array symbols
    inargs = tuple(a for a in inargs if isinstance(a, sym.Array) and a.parent)
    inoutargs = tuple(a for a in inoutargs if isinstance(a, sym.Array) and a.parent)
    outargs = tuple(a for a in outargs if isinstance(a, sym.Array) and a.parent)

    # Do some sanity checking and warning for enclosed calls
    for call in FindNodes(ir.CallStatement).visit(region):
        if call.routine is BasicType.DEFERRED:
            warning(f'[Loki] Data offload: Routine {driver.name} has not been enriched ' +
                    f'in {str(call.name).lower()}')
            continue
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

    args = offload_map.args
    for arg in FindVariables().visit(driver.body):
        if not arg.name in args:
            continue

        dataptr = offload_map.dataptr_from_array(arg)
        if len(arg.dimensions) != 0:
            dims = arg.dimensions + (offload_idx_expr,)
        else:
            dims = (sym.RangeIndex((None, None)),) * (len(dataptr.shape)-1) + (offload_idx_expr,)
        change_map[arg] = dataptr.clone(dimensions=dims)

    driver.body = SubstituteExpressions(change_map, inplace=True).visit(driver.body)
