# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.analyse import dataflow_analysis_attached
from loki.batch import Transformation
from loki.expression import Array, symbols as sym, parse_expr
from loki.types import BasicType, SymbolAttributes
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer,
    SubstituteExpressions, pragma_regions_attached, is_loki_pragma, pragmas_attached
)
from loki.logging import warning, error

from loki.transformations.loop_blocking import split_loop
from loki.transformations.utilities import find_driver_loops
from loki.transformations.field_api import (
    FieldPointerMap, field_create_device_data, field_wait_for_async_queue
)
from loki.transformations.parallel import remove_field_api_view_updates


__all__ = [
    'FieldOffloadTransformation', 'FieldOffloadBlockedTransformation', 'find_offload_variables',
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
    deviceptr_prefix: str, optional
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
        remove_field_api_view_updates(driver, self.field_group_types)

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


class FieldOffloadBlockedTransformation(Transformation):
    """

    Transformation to perform blocked offload of arrays owned by Field API fields to the device.

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
    block_size: int
        Number of blocks per chunk (specified in the final dimension of the field).
    deviceptr_prefix: str, optional
        The prefix of device pointers added by this transformation (defaults to ``'loki_devptr_'``).
    field_group_types: list or tuple of str, optional
        Names of the field group types with members that may be offloaded (defaults to ``['']``).
    offload_index: str, optional
        Names of index variable to inject in the outmost dimension of offloaded arrays in the kernel
        calls (defaults to ``'IBL'``).
    asynchronous: bool, optional
        Perform asynchronous blocked offload (defaults to ``False``)
    num_queues: int, optional
        Number of queues for asynchronous offload should be set to 2 or higher for asynchronous
        offload (defaults to 1).
    """

    def __init__(self, block_size, devptr_prefix=None, field_group_types=None,
                 offload_index=None, asynchronous=False, num_queues=1):
        self.block_size = block_size
        self.deviceptr_prefix = 'loki_devptr_' if devptr_prefix is None else devptr_prefix
        field_group_types = [''] if field_group_types is None else field_group_types
        self.field_group_types = tuple(typename.lower() for typename in field_group_types)
        self.offload_index = 'IBL' if offload_index is None else offload_index

        if not isinstance(asynchronous, bool):
            warning('[Loki] FieldOffloadBlockedTransformation: asynchronous kwarg must be a bool' +
                    ' asynchronous set to False')
            self.asynchronous = False
        else:
            self.asynchronous = asynchronous
        self.num_queues = num_queues

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs['role']
        if role == 'driver':
            self.process_driver(routine)

    def process_driver(self, driver):

        # Remove the Field-API view-pointer boilerplate
        remove_field_api_view_updates(driver, self.field_group_types)

        with pragma_regions_attached(driver):
            with dataflow_analysis_attached(driver):
                for region in FindNodes(ir.PragmaRegion).visit(driver.body):
                    # Only work on active `!$loki data` regions
                    if not region.pragma or not is_loki_pragma(region.pragma, starts_with='data'):
                        continue

                    offload_variables = find_offload_variables(driver, region, self.field_group_types)
                    offload_map = FieldPointerMap(
                        *offload_variables, scope=driver, ptr_prefix=self.deviceptr_prefix
                    )
                    # inject declarations and offload API calls into driver region
                    declare_device_ptrs(driver, deviceptrs=offload_map.dataptrs)
                    # blocks all loops inside the region and places them inside one
                    splitting_vars, _, block_loop = block_driver_loops(driver, region, self.block_size)

                    if self.asynchronous and self.num_queues > 1:
                        add_device_field_allocations(driver, block_loop, offload_map,
                                                     self.block_size, self.num_queues)
                        queue, offset = add_async_blocking_vars(driver, block_loop, self.num_queues, splitting_vars)
                        add_blocked_field_offload_calls(driver, block_loop, region, offload_map,
                                                        splitting_vars, queue, offset)
                        add_wait_calls(driver, block_loop, queue, self.num_queues)
                    else:
                        add_blocked_field_offload_calls(driver, block_loop, region, offload_map, splitting_vars)
                    replace_kernel_args(driver, offload_map, self.offload_index)
        if self.asynchronous and self.num_queues >1:
            add_async_queue_to_pragmas(block_loop, queue)


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
                warning(f'[Loki] Data offload: Raw array object {arg.name} encountered in' +
                        f' {driver.name} that is not wrapped by a Field API object')
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


def add_blocked_field_offload_calls(driver, block_loop, region, offload_map, splitting_variables,
                                    queue=None, offset=None):
    """
    Adds blocked Field API data transfer calls to a region inside a block loop.


    Parameters
    ----------
    driver : :any:`Subroutine`
        Driver subroutine IR node
    block_loop : :any:`ir.Loop`
        Block loop containing the region to be offloaded.
    region : :any:`PragmaRegion`
        Code region to prepend and append data transfer calls.
    offload_map : :any:`FieldPointerMap`
        FieldPointerMap with variables to be offloaded.
    splitting_variables: :any:`LoopSplittingVariables`
        Loop splitting variables for `block_loop`
    queue : optional
        Queue parameter for Field API data transfer calls
    offset : optional
        Offset parameter for Field API data transfer calls.
    """

    host_to_device = offload_map.host_to_device_force_calls(blk_bounds=sym.LiteralList(values=(
                                                                splitting_variables.block_start,
                                                                splitting_variables.block_end)
                                                                                      ),
                                                            queue=queue,
                                                            offset=offset
                                                            )

    device_to_host = offload_map.sync_host_force_calls(blk_bounds=sym.LiteralList(values=(
                                                            splitting_variables.block_start,
                                                            splitting_variables.block_end)
                                                                                 ),
                                                       queue=queue,
                                                       offset=offset
                                                       )
    with pragmas_attached(driver, ir.Loop):
        update_map = {region: host_to_device+(region,)+device_to_host}
        Transformer(update_map, inplace=True).visit(block_loop)


def add_device_field_allocations(driver, block_loop, offload_map, block_size, num_queues):
    blk_bounds = sym.LiteralList(values=(sym.IntLiteral(1), block_size*num_queues))
    create_device_data_calls = tuple(field_create_device_data(field_ptr=offload_map.field_ptr_from_view(arg),
                                                              scope=driver,
                                                              blk_bounds=blk_bounds)
                                     for arg in offload_map.args)
    create_device_data_calls = tuple(dict.fromkeys(create_device_data_calls))
    with pragmas_attached(driver, ir.Loop):
        update_map = {
            block_loop: create_device_data_calls + (block_loop,)
        }
        Transformer(update_map, inplace=True).visit(driver.body)



def add_async_blocking_vars(routine, block_loop, num_queues, splitting_vars):
    queue = sym.Variable(name='loki_block_queue', type=SymbolAttributes(BasicType.INTEGER),
                         scope=routine)
    nqueues = sym.Variable(name='loki_block_nqueues', type=SymbolAttributes(BasicType.INTEGER,
                                                                            parameter=True,
                                                                            initial=sym.IntLiteral(num_queues)),
                           scope=routine)
    offset = sym.Variable(name='loki_block_offset', type=SymbolAttributes(BasicType.INTEGER),
                          scope=routine)
    # add variables to routine variable map
    routine.variables += (queue, nqueues, offset)

    # set queue and offset in loop
    async_blocking_body = (
        ir.Assignment(queue,
                      sym.Sum(children=(sym.InlineCall(sym.DeferredTypeSymbol('MODULO', scope=routine),
                                                       parameters=(splitting_vars.block_idx,
                                                       nqueues)),
                                        sym.IntLiteral(1)))
                      ),
        ir.Assignment(offset, parse_expr(f'({queue}-1)*{splitting_vars.block_size}'))
    )
    block_loop._update(body=async_blocking_body+block_loop.body)

    return queue, offset


def replace_kernel_args(driver, offload_map, offload_index):
    change_map = {}
    offload_idx_expr = driver.variable_map[offload_index]

    args = offload_map.args
    for arg in FindVariables().visit(driver.body):
        if arg.name not in args:
            continue

        dataptr = offload_map.dataptr_from_array(arg)
        if len(arg.dimensions) != 0:
            dims = arg.dimensions + (offload_idx_expr,)
        else:
            dims = (sym.RangeIndex((None, None)),) * (len(dataptr.shape)-1) + (offload_idx_expr,)
        change_map[arg] = dataptr.clone(dimensions=dims)

    driver.body = SubstituteExpressions(change_map, inplace=True).visit(driver.body)


def block_driver_loops(driver, region, block_size):
    with pragmas_attached(driver, ir.Loop):
        driver_loops = find_driver_loops(driver.body, targets=None)
        if len(driver_loops) == 1:
            loop = driver_loops[0]
        elif len(driver_loops) > 1:
            warning('[Loki] FieldOffloadBlockedTransformation: Multiple driver loops found in ' +
                    f'{driver.name}, discarding all but first')
            loop = driver_loops[0]
        else:
            error(f'[Loki] FieldOffloadBlockedTransformation: No driver loops found in {driver.name}')

        splitting_vars, inner_loop, outer_loop = split_loop(driver, loop, block_size, data_region=region)  # pylint: disable=used-before-assignment
        # move loop pragmas to inner loop
        if outer_loop.pragma is not None:
            pragmas = []
            for pragma in outer_loop.pragma:
                content = pragma.content
                pragmas.append(pragma.clone(content=content))
            inner_loop._update(pragma=tuple(pragmas))
            outer_loop._update(pragma=None)
    return splitting_vars, inner_loop, outer_loop


def add_async_queue_to_pragmas(section, queue):
    pragmas = FindNodes(ir.Pragma).visit(section)
    async_content = f' async({queue.name})'
    for pragma in pragmas:
        if is_loki_pragma(pragma, starts_with='data') or is_loki_pragma(pragma, starts_with='driver-loop'):
            pragma._update(content=pragma.content+async_content)


def add_wait_calls(driver, block_loop, queue, num_queues):
    wait_loop = ir.Loop(variable=queue,
                        body=field_wait_for_async_queue(queue, driver),
                        bounds=sym.LoopRange((sym.IntLiteral(1), sym.IntLiteral(num_queues)))
                        )
    change_map = {block_loop: (block_loop, wait_loop)}
    Transformer(change_map, inplace=True).visit(driver.body)
