# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0 # which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from enum import Enum
from itertools import chain
from idlelib.pyparse import trans

from loki.analyse.analyse_dataflow import FindReads
from loki.batch import Transformation
from loki.ir import nodes as ir, Transformer, pragmas_attached, FindNodes
from loki.subroutine import Subroutine
from loki.expression import symbols as sym, parse_expr, FindVariables, \
    SubstituteExpressions, ceil_division, iteration_index
from loki.tools import as_tuple
from loki.transformations.utilities import find_driver_loops
from loki.analyse import dataflow_analysis_attached, read_after_write_vars

__all__ = ['split_loop', 'block_loop_arrays']


class LoopSplittingVariables:
    """
    This class holds the loop splitting variables, e.g. outer loop block sizes and iteration
    bounds. It also holds the original loop variable of the inner loop.
    """

    def __init__(self, loop_var: sym.Variable, block_size):
        self._loop_var = loop_var
        # self._splitting_vars = splitting_vars
        self._splitting_vars = (loop_var.clone(name=loop_var.name + "_loop_block_size",
                                               type=loop_var.type.clone(parameter=True,
                                                                        initial=sym.IntLiteral(
                                                                            block_size))),
                                loop_var.clone(name=loop_var.name + "_loop_num_blocks"),
                                loop_var.clone(name=loop_var.name + "_loop_block_idx"),
                                loop_var.clone(name=loop_var.name + "_loop_local"),
                                loop_var.clone(name=loop_var.name + "_loop_iter_num"),
                                loop_var.clone(name=loop_var.name + "_loop_block_start"),
                                loop_var.clone(name=loop_var.name + "_loop_block_end")
                                )

    @property
    def loop_var(self):
        return self._loop_var

    @property
    def block_size(self):
        return self._splitting_vars[0]

    @property
    def num_blocks(self):
        return self._splitting_vars[1]

    @property
    def block_idx(self):
        return self._splitting_vars[2]

    @property
    def inner_loop_var(self):
        return self._splitting_vars[3]

    @property
    def iter_num(self):
        return self._splitting_vars[4]

    @property
    def block_start(self):
        return self._splitting_vars[5]

    @property
    def block_end(self):
        return self._splitting_vars[6]

    @property
    def splitting_vars(self):
        return self._splitting_vars


@dataclass
class LoopSplitInfo:
    splitting_vars: LoopSplittingVariables
    inner_loop: ir.Loop
    outer_loop: ir.Loop
    # def __init__(self, splitting_vars: LoopSplittingVariables, inner_loop: ir.Loop, outer_loop: ir.Loop):


def split_loop(routine: Subroutine, loop: ir.Loop, block_size: int):
    """
    Blocks a loop by splitting it into an outer loop and inner loop of size `block_size`.

    Parameters
    ----------
    routine: :any:`Subroutine`
        Subroutine object containing the loop. New variables introduced in the
        loop splitting will be declared in the body of routine.
    loop: :any:`Loop`
        Loop to be split.
    block_size: int
        inner loop size (size of blocking blocks)
    """

    # loop splitting variable declarations
    splitting_vars = LoopSplittingVariables(loop.variable, block_size)
    routine.variables += splitting_vars.splitting_vars

    # block index calculations
    blocking_body = (
        ir.Assignment(splitting_vars.block_start,
                      parse_expr(
                          f"({splitting_vars.block_idx} - 1) * {splitting_vars.block_size} + 1",
                          scope=routine)
                      ),
        ir.Assignment(splitting_vars.block_end,
                      sym.InlineCall(sym.DeferredTypeSymbol('MIN', scope=routine),
                                     parameters=(sym.Product(children=(
                                         splitting_vars.block_idx, splitting_vars.block_size)),
                                                 loop.bounds.upper))
                      ))

    # Outer loop blocking variable assignments
    loop_range = loop.bounds
    block_loop_inits = (
        ir.Assignment(splitting_vars.num_blocks,
                      ceil_division(loop_range.num_iterations,
                                    splitting_vars.block_size)),
    )

    # Inner loop
    iteration_nums = (
        ir.Assignment(splitting_vars.iter_num,
                      parse_expr(
                          f"{splitting_vars.block_start}+{splitting_vars.inner_loop_var}-1"),
                      scope=routine),
        ir.Assignment(loop.variable,
                      iteration_index(splitting_vars.iter_num, loop_range))
    )
    inner_loop = ir.Loop(variable=splitting_vars.inner_loop_var, body=iteration_nums + loop.body,
                         bounds=sym.LoopRange(
                             (sym.IntLiteral(1), parse_expr(
                                 f"{splitting_vars.block_end} - {splitting_vars.block_start} + 1",
                                 scope=routine))))

    #  Outer loop bounds + body
    outer_loop = ir.Loop(variable=splitting_vars.block_idx, body=blocking_body + (inner_loop,),
                         bounds=sym.LoopRange((sym.IntLiteral(1), splitting_vars.num_blocks)))
    change_map = {loop: block_loop_inits + (outer_loop,)}
    Transformer(change_map, inplace=True).visit(routine.body)
    return splitting_vars, inner_loop, outer_loop


def blocked_shape(a: sym.Array, blocking_indices, block_size):
    """
    calculates the dimensions for a blocked version of the array.
    """
    shape = tuple(
        sym.IntLiteral(block_size) if isinstance(dim, sym.Scalar) and any(
            bidx in dim for bidx in blocking_indices) else dim for dim
        in a.shape)
    return shape


def blocked_type(a: sym.Array):
    return a.type.clone(intent=None)


def replace_indices(dimensions, indices: list, replacement_index):
    """
    Returns a new dimension object with all occurences of indices changed to replacement_index.

    Parameters
    ----------
    dimensions:
        Symbolic representation of dimensions or indices.
    indices: list of `Variable`s
        that will be replaced in the new :any:`Dimension` object.
    replacement_index: :any:`Expression`
        replacement for the indices changed.

    Returns
    -------
    """
    dims = tuple(
        replacement_index if isinstance(dim, sym.Scalar) and any(
            blocking_var in dim for blocking_var in indices) else dim for dim
        in dimensions)
    return dims


def block_loop_arrays(routine: Subroutine, splitting_vars, inner_loop: ir.Loop,
                      outer_loop: ir.Loop, blocking_indices):
    """
    Replaces arrays inside the inner loop with blocked counterparts.

    This routine declares array variables to hold the blocks of the arrays used inside
    the loop and replaces array variables inside the loop with their blocked counterparts.
    An array is blocked with the leading dimensions

    Parameters
    ----------
    routine : Subroutine
        routine in which the blocking variables should be added.
    blocking_indices: list of  :any:`Variable`
        list of the index variables that arrays inside the loop should be blocked by.
    inner_loop: :any:`Loop`
        inner loop after loop splitting
    outer_loop : :any:`Loop`
        outer loop body after loop splitting
    blocking_indices : tuple or list of str
           Variable names of the indexes that should be blocked if in array
            expressions.

    """
    # Declare Blocked arrays
    arrays = tuple(var for var in FindVariables().visit(inner_loop.body) if
                   isinstance(var, sym.Array) and any(
                       bi in var for bi in blocking_indices))
    name_map = {a.name: a.name + '_block' for a in arrays}
    block_arrays = tuple(
        a.clone(name=name_map[a.name],
                dimensions=blocked_shape(a, blocking_indices, splitting_vars.block_size),
                type=blocked_type(a)) for a in arrays)
    routine.variables += block_arrays

    # Replace arrays in loop with blocked arrays and update idx
    block_array_expr = (
        a.clone(name=name_map[a.name],
                dimensions=replace_indices(a.dimensions, blocking_indices, inner_loop.variable))
        for a in arrays
    )
    SubstituteExpressions(dict(zip(arrays, block_array_expr)), inplace=True).visit(inner_loop.body)

    # memory copies
    block_range = sym.RangeIndex((splitting_vars.block_start, splitting_vars.block_end))
    local_range = sym.RangeIndex(
        (sym.IntLiteral(1),
         parse_expr(f"{splitting_vars.block_end} - {splitting_vars.block_start} + 1",
                    scope=routine)))
    # input variables
    in_vars = (a for a in arrays if a.type.intent in ('in', 'inout'))
    copyins = tuple(
        ir.Assignment(a.clone(name=name_map[a.name],
                              dimensions=replace_indices(a.dimensions, blocking_indices,
                                                         local_range)),
                      a.clone(
                          dimensions=replace_indices(a.dimensions, blocking_indices, block_range)))
        for a in in_vars)
    # output variables
    out_vars = (a for a in arrays if a.type.intent in ('out', 'inout'))
    copyouts = tuple(
        ir.Assignment(
            a.clone(dimensions=replace_indices(a.dimensions, blocking_indices, block_range)),
            a.clone(name=name_map[a.name],
                    dimensions=replace_indices(a.dimensions, blocking_indices, local_range))
        )
        for a in out_vars)
    change_map = {inner_loop: copyins + (inner_loop,) + copyouts}
    Transformer(change_map, inplace=True).visit(outer_loop)




def find_alternate_idx(loop: ir.Loop, routine):
    loop_variable = loop.variable
    assignment_nodes = FindNodes(ir.Assignment).visit(loop.body)
    for an in assignment_nodes:
        if isinstance(an.rhs, sym.InlineCall):
            print("something")
        else:
            rhs = an.rhs
        print(type(rhs))
        if loop.variable in rhs:
            print(f'loop variable: "{loop.variable}" is in assignment expression {an.rhs}')


def get_field_type(a: sym.Array) -> sym.DerivedType:
    """
    Returns the corresponding FIELD API type for an array.

    This transformation is IFS specific and assumes that the
    type is an array declared with one of the IFS type specifiers, e.g. KIND=JPRB
    """
    type_map = ["jprb",
                "jpit", "jpis",
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
    field_type = sym.DerivedType(name="field_" + str(rank) + type_name[2:4])
    return field_type


def field_new(field_ptr, data, scope):
    return ir.CallStatement(sym.ProcedureSymbol('FIELD_NEW', scope=scope),
                            (field_ptr,), (('DATA', data),))


def field_delete(field_ptr, scope):
    return ir.CallStatement(sym.ProcedureSymbol('FIELD_DELETE', scope=scope),
                            (field_ptr,))


class FieldAPITransferType(Enum):
    READ_ONLY = 1
    READ_WRITE = 2
    WRITE_ONLY = 3


def field_get_device_data(field_ptr, dev_ptr, transfer_type: FieldAPITransferType, scope: ir.Scope):
    assert isinstance(transfer_type, FieldAPITransferType)
    if transfer_type == FieldAPITransferType.READ_ONLY:
        suffix = 'RDONLY'
    if transfer_type == FieldAPITransferType.READ_WRITE:
        suffix = 'RDWR'
    if transfer_type == FieldAPITransferType.WRITE_ONLY:
        suffix = 'WRONLY'
    procedure_name = 'GET_DEVICE_DATA_' + suffix
    return ir.CallStatement(sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope),
                            (dev_ptr,))


def field_sync_host(field_ptr, scope):
    procedure_name = 'SYNC_HOST_RDWR'
    return ir.CallStatement(sym.ProcedureSymbol(procedure_name, parent=field_ptr, scope=scope), ())


class LoopBlockFieldAPITransformation(Transformation):
    def __init__(self, block_size=40, blocking_indices=None):
        self.block_size = block_size            # TODO: this should really be a constant that can be set at compile time
        self.block_suffix = '_block_ptr'
        self.field_block_suffix = '_field_block_ptr'
        self.blocking_indices = ('jbl',) if blocking_indices is None else blocking_indices

    def transform_subroutine(self, routine, **kwargs):
        # self.splitting_vars, self.inner_loop, self.outer_loop = split_loop(routine, kwargs['loop'],
        #                                                                   self.block_size)
        role = kwargs['role']
        targets = as_tuple(kwargs.get('targets'))
        if role == 'kernel':
            self.process_kernel(routine)
        if role == 'driver':
            self.process_driver(routine, targets)

    def process_kernel(self, routine):
        pass

    def process_driver(self, routine, targets=None):
        with pragmas_attached(routine, ir.Loop):
            driver_loops = find_driver_loops(routine, targets)

        # filter and split driver loops
        splitting_loops = self.find_splitting_loops(driver_loops, routine, targets)
        split_loops = ((split_loop(routine, loop, self.block_size)) for loop in
                       splitting_loops)

        # insert Field API objects in driver
        # TODO: ADD FIELD MODULE IMPORTS TO DRIVER
        for splitting_vars, inner_loop, outer_loop in split_loops:
            field_ptr_map, block_ptr_map = self._insert_fields(routine, splitting_vars, inner_loop, outer_loop)
            self._insert_acc_data_pragmas(inner_loop, outer_loop, block_ptr_map)

    def find_splitting_loops(self, driver_loops, routine, targets):
        # some logic to filter splitting loops (e.g. if loop splitting variable is used)
        assert self.block_suffix != self.field_block_suffix, "ASSERT TO PREVENT LSP CODE CHECK WARNINGS"
        return driver_loops

    def _field_ptr_from_array(self, a: sym.Array) -> sym.Variable:
        """
        Returns a pointer :any:`Variable` pointing to a FIELD with types matching the array.
        """

        field_ptr_type = sym.SymbolAttributes(get_field_type(a), polymorphic=True, pointer=True,
                                              intent=None, initial="NULL()")
        field_ptr = sym.Variable(name=a.name + self.field_block_suffix, type=field_ptr_type)
        return field_ptr

    def _block_ptr_from_array(self, a: sym.Array) -> sym.Variable:
        """
        Returns a contiguous pointer :any:`Variable` with types matching the array a
        """
        shape = (sym.RangeIndex((None, None)),) * len(a.shape)
        block_ptr_type = a.type.clone(pointer=True, contiguous=True, shape=shape, intent=None)
        block_ptr = sym.Variable(name=a.name + self.block_suffix, type=block_ptr_type,
                                 dimensions=shape)
        return block_ptr

    def _insert_fields(self, routine, splitting_vars, inner_loop, outer_loop):
        """
        Replaces arrays inside the inner loop with FIELD api fields

        This routine declares field object pointers to hold the blocks of the arrays used inside
        the loop and replaces array variables inside the loop with their blocked counterparts.
        An array is blocked with the leading dimensions

        """

        # Field API pointer and device pointer variables
        blocking_arrays = tuple(var for var in FindVariables().visit(inner_loop.body) if
                                isinstance(var, sym.Array) and any(
                                    bi in var for bi in self.blocking_indices))

        field_pointers = tuple(self._field_ptr_from_array(a) for a in blocking_arrays)
        routine.variables += field_pointers
        block_pointers = tuple(self._block_ptr_from_array(a) for a in blocking_arrays)
        routine.variables += block_pointers

        field_ptr_map = dict(zip(blocking_arrays, field_pointers))
        block_ptr_map = dict(zip(blocking_arrays, block_pointers))

        block_range = sym.RangeIndex(
            (splitting_vars.block_start, splitting_vars.block_end))

        # Field creation/updates
        field_updates = tuple(field_new(block_ptr_map[a],
                                        a.clone(dimensions=replace_indices(a.dimensions,
                                                                           self.blocking_indices,
                                                                           block_range)),
                                        routine) for a in blocking_arrays)

        # # memory copies
        # should be replaced by a proper data flow analysis in the future
        in_vars = tuple(a for a in blocking_arrays if a.type.intent == 'in')
        inout_vars = tuple(a for a in blocking_arrays if a.type.intent == 'inout')
        out_vars = tuple(a for a in blocking_arrays if a.type.intent == 'out')

        # FIELD API host to device transfers
        host_to_device = tuple(field_get_device_data(field_ptr_map[var], block_ptr_map[var],
                                                     FieldAPITransferType.READ_ONLY, routine)
                               for var in in_vars)
        host_to_device += tuple(field_get_device_data(field_ptr_map[var], block_ptr_map[var],
                                                      FieldAPITransferType.READ_WRITE, routine)
                                for var in inout_vars)
        host_to_device += tuple(field_get_device_data(field_ptr_map[var], block_ptr_map[var],
                                                      FieldAPITransferType.WRITE_ONLY, routine)
                                for var in out_vars)

        device_to_host = tuple(
            field_sync_host(field_ptr_map[var], routine) for var in chain(inout_vars, out_vars))
        field_deletes = tuple(field_delete(field_ptr_map[var], routine) for var in blocking_arrays)

        change_map = {inner_loop: field_updates + host_to_device + (
            inner_loop,) + device_to_host + field_deletes}
        Transformer(change_map, inplace=True).visit(outer_loop)

        # Replace arrays in loop with blocked arrays and update idx
        block_array_expr = (
            a.clone(name=block_ptr_map[a].name,
                    dimensions=replace_indices(a.dimensions, self.blocking_indices,
                                               inner_loop.variable))
            for a in blocking_arrays
        )
        SubstituteExpressions(dict(zip(blocking_arrays, block_array_expr)), inplace=True).visit(
            inner_loop.body)
        
        return field_ptr_map, block_ptr_map

    def _insert_acc_data_pragmas(self, inner_loop, outer_loop, block_ptr_map):
        non_blocked_vars = tuple(
                var for var in FindVariables().visit(inner_loop.body) if var is not block_ptr_map)

        acc_copyins = ir.Pragma(keyword='acc',
                                content=f'data copyin({", ".join(v.name for v in non_blocked_vars)})')
        acc_present = ir.Pragma(keyword='acc',
                                content=f'data present({", ".join(v.name for v in block_ptr_map.values())})')
        # acc_data_start = (ir.Pragma(keyword='acc', content='data'), acc_copyins, acc_present)
        acc_data_start = (acc_copyins, acc_present)
        acc_data_end = (ir.Pragma(keyword='acc', content='data end'),)
        
        change_map = {inner_loop: acc_data_start + (inner_loop,) + acc_data_end}
        Transformer(change_map, inplace=True).visit(outer_loop)

