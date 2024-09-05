# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from numpy.core.shape_base import block

from loki.ir import nodes as ir, Transformer
from loki.subroutine import Subroutine
from loki.expression import symbols as sym, parse_expr, FindVariables, \
    SubstituteExpressions, ceil_division, iteration_index


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
                          f"({splitting_vars.block_idx} - 1) * {splitting_vars.block_size} + 1")
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
                          f"{splitting_vars.block_start}+{splitting_vars.inner_loop_var}-1")),
        ir.Assignment(loop.variable,
                      iteration_index(splitting_vars.iter_num, loop_range))
    )
    inner_loop = ir.Loop(variable=splitting_vars.inner_loop_var, body=iteration_nums + loop.body,
                         bounds=sym.LoopRange(
                             (sym.IntLiteral(1), parse_expr(
                                 f"{splitting_vars.block_end} - {splitting_vars.block_start} + 1"))))

    #  Outer loop bounds + body
    outer_loop = ir.Loop(variable=splitting_vars.block_idx, body=blocking_body + (inner_loop,),
                         bounds=sym.LoopRange((sym.IntLiteral(1), splitting_vars.num_blocks)))
    change_map = {loop: block_loop_inits + (outer_loop,)}
    Transformer(change_map, inplace=True).visit(routine.ir)
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
    dimensions: Symbolic representation of dimensions or indices.
    indices: list of `Variable`s that will be replaced in the new :any:`Dimension` object.
    replacement_index: :any:`Expression` replacement for the indices changed.

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
         parse_expr(f"{splitting_vars.block_end} - {splitting_vars.block_start} + 1")))
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
