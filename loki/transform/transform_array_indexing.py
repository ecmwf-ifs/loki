# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collection of utility routines to deal with common array indexing conversions.

"""
from collections import defaultdict
from itertools import count
import operator as op

from loki.logging import info
from loki.analyse import dataflow_analysis_attached
from loki.expression import (
    symbols as sym, simplify, symbolic_op, FindVariables, SubstituteExpressions
)
from loki.ir import Assignment, Loop, VariableDeclaration
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import SymbolAttributes, BasicType
from loki.visitors import FindNodes, Transformer


__all__ = [
    'shift_to_zero_indexing', 'invert_array_indices',
    'resolve_vector_notation', 'normalize_range_indexing',
    'promote_variables', 'promote_nonmatching_variables',
    'promotion_dimensions_from_loop_nest', 'demote_variables',
    'flatten_arrays', 'normalize_array_shape_and_access'
]


def shift_to_zero_indexing(routine):
    """
    Shift all array indices to adjust to 0-based indexing conventions (eg. for C or Python)
    """
    vmap = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if isinstance(v, sym.Array):
            new_dims = []
            for d in v.dimensions:
                if isinstance(d, sym.RangeIndex):
                    start = d.start - sym.Literal(1) if d.start is not None else None
                    # no shift for stop because Python ranges are [start, stop)
                    new_dims += [sym.RangeIndex((start, d.stop, d.step))]
                else:
                    new_dims += [d - sym.Literal(1)]
            vmap[v] = v.clone(dimensions=as_tuple(new_dims))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


def invert_array_indices(routine):
    """
    Invert data/loop accesses from column to row-major

    TODO: Take care of the indexing shift between C and Fortran.
    Basically, we are relying on the CGen to shift the iteration
    indices and dearly hope that nobody uses the index's value.
    """
    # Invert array indices in the routine body
    vmap = {}
    for v in FindVariables(unique=True).visit(routine.body):
        if isinstance(v, sym.Array):
            rdim = as_tuple(reversed(v.dimensions))
            vmap[v] = v.clone(dimensions=rdim)
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Invert variable and argument dimensions for the automatic cast generation
    for v in routine.variables:
        if isinstance(v, sym.Array):
            rdim = as_tuple(reversed(v.dimensions))
            if v.shape:
                rshape = as_tuple(reversed(v.shape))
                vmap[v] = v.clone(dimensions=rdim, type=v.type.clone(shape=rshape))
            else:
                vmap[v] = v.clone(dimensions=rdim)
    routine.variables = [vmap.get(v, v) for v in routine.variables]


def resolve_vector_notation(routine):
    """
    Resolve implicit vector notation by inserting explicit loops
    """
    loop_map = {}
    index_vars = set()
    vmap = {}
    for stmt in FindNodes(Assignment).visit(routine.body):
        # Loop over all variables and replace them with loop indices
        vdims = []
        shape_index_map = {}
        index_range_map = {}
        for v in FindVariables(unique=False).visit(stmt):
            if not isinstance(v, sym.Array):
                continue

            # Skip if the entire array is used implicitly
            if not v.dimensions:
                continue

            ivar_basename = f'i_{stmt.lhs.basename}'
            for i, dim, s in zip(count(), v.dimensions, as_tuple(v.shape)):
                if isinstance(dim, sym.RangeIndex):
                    # Create new index variable
                    vtype = SymbolAttributes(BasicType.INTEGER)
                    ivar = sym.Variable(name=f'{ivar_basename}_{i}', type=vtype, scope=routine)
                    shape_index_map[s] = ivar
                    index_range_map[ivar] = s

                    if ivar not in vdims:
                        vdims.append(ivar)

            # Add index variable to range replacement
            new_dims = as_tuple(shape_index_map.get(s, d)
                                for d, s in zip(v.dimensions, as_tuple(v.shape)))
            vmap[v] = v.clone(dimensions=new_dims)

        index_vars.update(list(vdims))

        # Recursively build new loop nest over all implicit dims
        if len(vdims) > 0:
            loop = None
            body = stmt
            for ivar in vdims:
                irange = index_range_map[ivar]
                if isinstance(irange, sym.RangeIndex):
                    bounds = sym.LoopRange(irange.children)
                else:
                    bounds = sym.LoopRange((sym.Literal(1), irange, sym.Literal(1)))
                loop = Loop(variable=ivar, body=as_tuple(body), bounds=bounds)
                body = loop

            loop_map[stmt] = loop

    if len(loop_map) > 0:
        routine.body = Transformer(loop_map).visit(routine.body)
    routine.variables += tuple(set(index_vars))

    # Apply variable substitution
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


def normalize_range_indexing(routine):
    """
    Replace the ``(1:size)`` indexing in array sizes that OMNI introduces.
    """
    def is_one_index(dim):
        return isinstance(dim, sym.RangeIndex) and dim.lower == 1 and dim.step is None

    vmap = {}
    for v in routine.variables:
        if isinstance(v, sym.Array):
            new_dims = [d.upper if is_one_index(d) else d for d in v.dimensions]
            new_shape = [d.upper if is_one_index(d) else d for d in v.shape]
            new_type = v.type.clone(shape=as_tuple(new_shape))
            vmap[v] = v.clone(dimensions=as_tuple(new_dims), type=new_type)
    routine.variables = [vmap.get(v, v) for v in routine.variables]


def promote_variables(routine, variable_names, pos, index=None, size=None):
    """
    Promote a list of variables by inserting new array dimensions of given size
    and updating all uses of these variables with a given index expression.

    When providing only :data:`size` or :data:`index`, promotion is restricted
    to updating only variable declarations or their use, respectively, and the
    other is left unchanged.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which the variables should be promoted.
    variable_names : list of str
        The names of variables to be promoted. Matching of variables against
        names is case-insensitive.
    pos : int
        The position of the new array dimension using Python indexing
        convention (i.e., count from 0 and use negative values to count from
        the end).
    index : :py:class:`pymbolic.primitives.Expression`, optional
        The indexing expression (or a tuple for multi-dimension promotion)
        to use for the promotion dimension(s), e.g., loop variables. Usage of
        variables is only updated if `index` is provided. When the index
        expression is not live at the variable use, ``:`` is used instead.
    size : :py:class:`pymbolic.Expression`, optional
        The size of the dimension (or tuple for multi-dimension promotion) to
        insert at `pos`. When this is provided, the declaration of variables
        is updated accordingly.
    """
    variable_names = {name.lower() for name in variable_names}

    if not variable_names:
        return

    # Insert new index dimension
    if index is not None:
        index = as_tuple(index)
        index_vars = [set(FindVariables().visit(i)) for i in index]

        # Create a copy of the tree and apply promotion in-place
        routine.body = Transformer().visit(routine.body)

        with dataflow_analysis_attached(routine):
            for node, var_list in FindVariables(unique=False, with_ir_node=True).visit(routine.body):
                # All the variables marked for promotion that appear in this IR node
                var_list = [v for v in var_list if v.name.lower() in variable_names]

                if not var_list:
                    continue

                # We use the given index expression in this node if all
                # variables therein are defined, otherwise we use `:`
                node_index = tuple(i if v <= node.live_symbols else sym.RangeIndex((None, None))
                                   for i, v in zip(index, index_vars))

                var_map = {}
                for var in var_list:
                    # If the position is given relative to the end we convert it to
                    # a positive index
                    if hasattr(var, 'dimensions'):
                        var_dim = var.dimensions
                    else:
                        var_dim = ()
                    if pos < 0:
                        var_pos = len(var_dim) - pos + 1
                    else:
                        var_pos = pos

                    dimensions = as_tuple(var_dim[:var_pos] + node_index + var_dim[var_pos:])
                    var_map[var] = var.clone(dimensions=dimensions)

                # need to apply update immediately because identical variable use
                # in other nodes might yield same hash but different substitution
                SubstituteExpressions(var_map, inplace=True).visit(node)

    # Apply shape promotion
    if size is not None:
        size = as_tuple(size)

        var_list = [var for decl in FindNodes(VariableDeclaration).visit(routine.spec)
                    for var in decl.symbols if var.name.lower() in variable_names]

        var_shapes = [getattr(var, 'shape', ()) for var in var_list]
        if pos < 0:
            var_pos = [len(shape) - pos + 1 for shape in var_shapes]
        else:
            var_pos = [pos] * len(var_shapes)
        var_shapes = [d[:p] + size + d[p:] for d, p in zip(var_shapes, var_pos)]

        var_map = {v: v.clone(type=v.type.clone(shape=shape), dimensions=shape)
                   for v, shape in zip(var_list, var_shapes)}
        routine.spec = SubstituteExpressions(var_map).visit(routine.spec)


def promotion_dimensions_from_loop_nest(var_names, loops, promotion_vars_dims, promotion_vars_index):
    """
    Determine promotion dimensions corresponding to the iteration space of a loop nest.

    Parameters
    ----------
    var_names : list of str
        The names of variables to consider for promotion.
    loops : list of :any:`Loop`
        The list of nested loops, sorted from outermost to innermost.
    promotion_vars_dims : dict((str, tuple))
        The mapping of variable names to promotion dimensions. When determining
        promotion dimensions for the variables in :data:`var_names` this dict is
        checked for already existing promotion dimensions and, if not matching,
        the maximum of both is taken for each dimension.
    promotion_vars_index : dict((str, tuple))
        The mapping of variable names to subscript expressions. These expressions
        are later inserted for every variable use. When the indexing expression
        for the loop nest does not match the existing expression in this dict,
        a :any:`RuntimeError` is raised.

    Returns
    -------
    (:data:`promotion_vars_dims`, :data:`promotion_vars_dims`) : tuple of dict
        The updated mappings :data:`promotion_vars_dims` and :data:`promotion_vars_index`.

    """
    # TODO: Would be nice to be able to promote this to the smallest possible dimension
    #       (in a loop var=start,end this is (end-start+1) with subscript index (var-start+1))
    #       but it requires being able to decide whether this yields a constant dimension,
    #       thus we need to stick to the upper bound for the moment as this is constant
    #       in our use cases.
    loop_lengths = [simplify(loop.bounds.stop) for loop in reversed(loops)]
    loop_index = [loop.variable for loop in reversed(loops)]

    def _merge_dims_and_index(dims_a, index_a, dims_b, index_b, var_name):
        """
        Helper routine that takes two pairs of promotion dimensions and indices
        (let's call them a and b) and tries to merge them to form the promotion
        configuration that accomodates both.
        """
        # Let's assume we have the same or more promotion dimensions in b than in a
        if len(dims_b) < len(dims_a):
            return _merge_dims_and_index(dims_b, index_b, dims_a, index_a, var_name)  # pylint: disable=arguments-out-of-order

        # We identify each dimension by the index expression; therefore, we have
        # to merge them first
        new_index = []
        ptr_a, ptr_b = 0, 0
        while ptr_a < len(index_a) and ptr_b < len(index_b):
            # Let's see if the next index in a can be found somewhere in b
            try:
                a_in_b = index_b.index(index_a[ptr_a], ptr_b)
            except ValueError:
                a_in_b = None

            if a_in_b is None:
                # It's not in there, so just add it to the new index
                # and go to the next
                new_index += [index_a[ptr_a]]
                ptr_a += 1
            else:
                # Found a in b: add it and anything before from b
                new_index += index_b[ptr_b:a_in_b+1]
                ptr_a += 1
                ptr_b = a_in_b + 1

            # Skip any indices we have already dealt with
            while ptr_a < len(index_a) and index_a[ptr_a] in new_index:
                ptr_a += 1
            while ptr_b < len(index_b) and index_b[ptr_b] in new_index:
                ptr_b += 1

        # Add any remaining indices in a and b
        if ptr_a < len(index_a):
            assert ptr_b == len(index_b)
            new_index += index_a[ptr_a:]
        else:
            assert ptr_a == len(index_a)
            new_index += index_b[ptr_b:]

        # With the merged index in place, we need to go through each corresponding
        # dimension from a and b and pick the larger
        new_dims = []
        for idx in new_index:
            # Look for position of that index in a and b
            try:
                ptr_a = index_a.index(idx)
            except ValueError:
                ptr_a = None
            try:
                ptr_b = index_b.index(idx)
            except ValueError:
                ptr_b = None

            if ptr_a is None:
                # exists only in b
                new_dims += [dims_b[ptr_b]]
            elif ptr_b is None:
                # exists only in a
                new_dims += [dims_a[ptr_a]]
            else:
                # exists in both: pick the larger
                if symbolic_op(dims_a[ptr_a], op.lt, dims_b[ptr_b]):
                    new_dims += [dims_b[ptr_b]]
                else:
                    new_dims += [dims_a[ptr_a]]

        # ... and we're done: return the new dimensions and index
        return new_dims, new_index

    for var_name in var_names:
        # Check if we have already marked this variable for promotion: let's make sure the added
        # dimensions are large enough for this loop (nest)
        if var_name not in promotion_vars_dims:
            promotion_vars_dims[var_name] = loop_lengths
            promotion_vars_index[var_name] = loop_index
        else:
            promotion_vars_dims[var_name], promotion_vars_index[var_name] = \
                _merge_dims_and_index(promotion_vars_dims[var_name], promotion_vars_index[var_name],
                                      loop_lengths, loop_index, var_name)

    return promotion_vars_dims, promotion_vars_index


def promote_nonmatching_variables(routine, promotion_vars_dims, promotion_vars_index):
    """
    Promote multiple variables with potentially non-matching promotion
    dimensions or index expressions.

    This is a convenience routine for using :meth:`promote_variables` that
    groups variables by indexing expression and promotion dimensions to
    reduce the number of calls to :meth:`promote_variables`.

    Parameters
    ----------
    routine : any:`Subroutine`
        The subroutine to be modified.
    promotion_vars_dims : dict
        The mapping of variable names to promotion dimensions. The variables'
        shapes are expanded where necessary to have at least these dimensions.
    promotion_vars_index : dict
        The mapping of variable names to subscript expressions to be used
        whenever reading/writing the variable.
    """
    if not promotion_vars_dims:
        return

    variable_map = routine.variable_map

    # First, let's find out what dimensions we actually need to add
    for var_name in promotion_vars_dims:
        shape = variable_map[var_name].type.shape
        if shape is None:
            continue

        # Eliminate 1:n declared shapes (mostly thanks to OMNI)
        shape = [s.stop if isinstance(s, sym.Range) and s.start == 1 else s for s in shape]

        dims = []
        index = []
        for dim, idx in zip(promotion_vars_dims[var_name], promotion_vars_index[var_name]):
            if not any(symbolic_op(dim, op.eq, d) for d in shape):
                dims += [dim]
                index += [idx]
        promotion_vars_dims[var_name] = dims
        promotion_vars_index[var_name] = index

    # Group promotion variables by index and size to reduce number of traversals for promotion
    index_size_var_map = defaultdict(list)
    for var_name, size in promotion_vars_dims.items():
        index_size_var_map[(as_tuple(promotion_vars_index[var_name]), as_tuple(size))] += [var_name]
    for (index, size), var_names in index_size_var_map.items():
        promote_variables(routine, var_names, -1, index=index, size=size)
    info('%s: promoted variable(s): %s', routine.name, ', '.join(promotion_vars_dims.keys()))


def demote_variables(routine, variable_names, dimensions):
    """
    Demote a list of array variables by removing any occurence of a
    provided set of dimension symbols.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which the variables should be promoted.
    variable_names : list of str
        The names of variables to be promoted. Matching of variables against
        names is case-insensitive.
    dimensions : :py:class:`pymbolic.Expression` or tuple
        Symbol name or tuple of symbol names representing the dimension
        to remove from all occurances of the named variables.
    """
    dimensions = as_tuple(dimensions)

    # Compare lower-case only, since we're not comparing symbols
    vnames = tuple(name.lower() for name in variable_names)

    variables = FindVariables(unique=False).visit(routine.ir)
    variables = tuple(v for v in variables if v.name.lower() in vnames)
    variables = tuple(v for v in variables if hasattr(v, 'shape'))

    if not variables:
        return

    # Record original array shapes
    shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})

    # Remove shape and dimension entries from each variable in the list
    vmap = {}
    for v in variables:
        old_shape = shape_map[v.name]
        new_shape = tuple(s for s in old_shape if s not in dimensions)
        new_dims = tuple(d for d, s in zip(v.dimensions, old_shape) if s in new_shape)

        new_type = v.type.clone(shape=new_shape or None)
        vmap[v] = v.clone(dimensions=new_dims or None, type=new_type)

    # Propagate the new dimensions to declarations and routine bodys
    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)

    # Ensure all declarations with `DIMENSION` keywords are modified too!
    decls = tuple(
        d for d in FindNodes(VariableDeclaration).visit(routine.spec)
        if d.dimensions and any(s.name.lower() in vnames for s in d.symbols)
    )
    decl_map = {}
    for decl in decls:
        # If all symbols have the same shape (after demotion)
        sym_shape = tuple(s.shape if isinstance(s, sym.Array) else None for s in decl.symbols)
        if all(d == sym_shape[0] for d in sym_shape):
            dimensions = decl.symbols[0].shape if isinstance(decl.symbols[0], sym.Array) else None
            decl_map[decl] = decl.clone(dimensions=dimensions)
        else:
            # If not, split into multiple declarations
            sdims = tuple(s.shape if isinstance(s, sym.Array) else None for s in decl.symbols)
            decl_map[decl] = tuple(
                decl.clone(symbols=(s,), dimensions=d) for s, d in zip(decl.symbols, sdims)
            )
    routine.spec = Transformer(decl_map).visit(routine.spec)

    info(f'[Loki::Transform] Demoted variables in {routine.name}: {", ".join(variable_names)}')

def normalize_array_shape_and_access(routine):
    """
    Shift all arrays to start counting at "1"
    """
    def is_range_index(dim):
        return isinstance(dim, sym.RangeIndex) and not dim.lower == 1

    vmap = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if isinstance(v, sym.Array):
            new_dims = []
            for i, d in enumerate(v.shape):
                if isinstance(d, sym.RangeIndex):
                    if isinstance(v.dimensions[i], sym.RangeIndex):
                        start = simplify(v.dimensions[i].start - d.start + 1) if d.start is not None else None
                        stop = simplify(v.dimensions[i].stop - d.start + 1) if d.stop is not None else None
                        new_dims += [sym.RangeIndex((start, stop, d.step))]
                    else:
                        start = simplify(v.dimensions[i] - d.start + 1) if d.start is not None else None
                        new_dims += [start]
                else:
                    new_dims += [v.dimensions[i]]
            vmap[v] = v.clone(dimensions=as_tuple(new_dims))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    vmap = {}
    for v in routine.variables:
        if isinstance(v, sym.Array):
            new_dims = [sym.RangeIndex((1, simplify(d.upper - d.lower + 1)))
                if is_range_index(d) else d for d in v.dimensions]
            new_shape = [sym.RangeIndex((1, simplify(d.upper - d.lower + 1)))
                if is_range_index(d) else d for d in v.shape]
            new_type = v.type.clone(shape=as_tuple(new_shape))
            vmap[v] = v.clone(dimensions=as_tuple(new_dims), type=new_type)
    routine.variables = [vmap.get(v, v) for v in routine.variables]
    normalize_range_indexing(routine)


def flatten_arrays(routine, order='F', start_index=1):
    """
    Flatten arrays, converting multi-dimensional arrays to
    one-dimensional arrays.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which the variables should be promoted.
    order : str
        Assume Fortran (F) vs. C memory/array order.
    start_index : int
        Assume array indexing starts with `start_index`.
    """
    def new_dims(dim, shape):
        if len(dim) > 1:
            if isinstance(shape[-2], sym.RangeIndex):
                raise TypeError(f'Resolve shapes being of type RangeIndex, e.g., "{shape[-2]}" before flattening!')
            _dim = (sym.Sum((dim[-2], sym.Product((shape[-2], dim[-1] - start_index)))),)
            new_dim = dim[:-2]
            new_dim += _dim
            return new_dims(new_dim, shape[:-1])
        return dim

    if order == 'C':
        array_map = {
            var: var.clone(dimensions=new_dims(var.dimensions[::-1], var.shape[::-1]))
            for var in FindVariables().visit(routine.body)
            if isinstance(var, sym.Array) and var.shape and len(var.shape)
        }
    elif order == 'F':
        array_map = {
            var: var.clone(dimensions=new_dims(var.dimensions, var.shape))
            for var in FindVariables().visit(routine.body)
            if isinstance(var, sym.Array) and var.shape and len(var.shape)
        }
    else:
        raise ValueError(f'Unsupported array order "{order}"')

    routine.body = SubstituteExpressions(array_map).visit(routine.body)

    routine.variables = [v.clone(dimensions=as_tuple(sym.Product(v.shape)),
                                 type=v.type.clone(shape=as_tuple(sym.Product(v.shape))))
                         if isinstance(v, sym.Array) else v for v in routine.variables]
