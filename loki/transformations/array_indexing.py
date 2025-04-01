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

from loki.batch import Transformation, ProcedureItem
from loki.logging import info
from loki.analyse import dataflow_analysis_attached
from loki.expression import symbols as sym, simplify, symbolic_op, is_constant
from loki.ir import (
    nodes as ir, Assignment, Loop, VariableDeclaration, FindNodes,
    Transformer, FindVariables, SubstituteExpressions, FindInlineCalls
)
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import SymbolAttributes, BasicType
from loki.transformations.inline import inline_constant_parameters


__all__ = [
    'remove_explicit_array_dimensions', 'add_explicit_array_dimensions',
    'shift_to_zero_indexing', 'invert_array_indices',
    'resolve_vector_notation', 'normalize_range_indexing',
    'promote_variables', 'promote_nonmatching_variables',
    'promotion_dimensions_from_loop_nest', 'demote_variables',
    'flatten_arrays', 'normalize_array_shape_and_access',
    'LowerConstantArrayIndices'
]

def remove_explicit_array_dimensions(routine, calls_only=False):
    """
    Remove colon notation from array dimensions within :any:`Subroutine` ``routine``.
    E.g., convert two-dimensional array ``arr2d(:,:)`` to ``arr2d`` or
    ``arr3d(:,:,:)`` to ``arr3d``, but NOT e.g., ``arr(1,:,:)``.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to check
    calls_only: bool
        Whether to remove colon notation from array dimensions only
        from arrays within (inline) calls or all arrays (default: False)
    """
    if calls_only:
        # handle calls (to subroutines) and inline calls (to functions)
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        inline_calls = FindInlineCalls().visit(routine.body)
        inline_call_map = {}
        for call in as_tuple(calls) + as_tuple(inline_calls):
            # handle arguments
            arguments = ()
            for arg in call.arguments:
                if isinstance(arg, sym.Array) and all(dim == sym.RangeIndex((None, None)) for dim in arg.dimensions):
                    new_dimensions = None
                    arguments += (arg.clone(dimensions=new_dimensions),)
                else:
                    arguments += (arg,)
            # handle kwargs
            kwarguments = ()
            for (kwarg_name, kwarg) in call.kwarguments:
                if isinstance(kwarg, sym.Array) and all(dim==sym.RangeIndex((None, None)) for dim in kwarg.dimensions):
                    kwarguments += ((kwarg_name, kwarg.clone(dimensions=None)),)
                else:
                    kwarguments += ((kwarg_name, kwarg),)
            # distinguish calls and inline calls
            if isinstance(call, sym.InlineCall):
                inline_call_map[call] = call.clone(parameters=arguments, kw_parameters=kwarguments)
            else:
                # directly update calls
                call._update(arguments=arguments, kwarguments=kwarguments)
        if inline_call_map:
            # update inline calls via expression substitution
            routine.body = SubstituteExpressions(inline_call_map).visit(routine.body)
    else:
        arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
        array_map = {}
        for array in arrays:
            if all(dim == sym.RangeIndex((None, None)) for dim in array.dimensions):
                new_dimensions = None
                array_map[array] = array.clone(dimensions=new_dimensions)
        routine.body = SubstituteExpressions(array_map).visit(routine.body)


def add_explicit_array_dimensions(routine):
    """
    Make dimensions of arrays explicit within :any:`Subroutine` ``routine``.
    E.g., convert two-dimensional array ``arr2d`` to ``arr2d(:,:)`` or
    ``arr3d`` to ``arr3d(:,:,:)``.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to check
    """
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
    array_map = {}
    for array in arrays:
        if not array.dimensions:
            new_dimensions = (sym.RangeIndex((None, None)),) * len(array.shape)
            array_map[array] = array.clone(dimensions=new_dimensions)
    routine.body = SubstituteExpressions(array_map).visit(routine.body)


def shift_to_zero_indexing(routine, ignore=None):
    """
    Shift all array indices to adjust to 0-based indexing conventions (eg. for C or Python)

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which the array dimensions should be shifted
    ignore : list of str
        List of variable names for which, if found in the dimension expression
        of an array subscript, that dimension is not shifted to zero.
    """
    ignore = as_tuple(ignore)
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
                    if ignore and any(var in ignore for var in FindVariables().visit(d)):
                        new_dims += [d]
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
    # find available loops and create map {(lower, upper, step): loop_variable}
    loops = FindNodes(Loop).visit(routine.body)
    loop_map = {(loop.bounds.lower, loop.bounds.upper, loop.bounds.step or 1):
                loop.variable for loop in loops}
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
                    # use the shape for e.g., `ARR(:)`, but use the dimension for e.g., `ARR(2:5)`
                    _s = dim if dim.lower is not None else s
                    # create tuple to test whether an appropriate loop is already available
                    test_range = (sym.IntLiteral(1), _s, 1) if not isinstance(_s, sym.RangeIndex)\
                            else (_s.lower, _s.upper, 1)
                    # actually test for it
                    if test_range in loop_map:
                        # Use index variable of available matching loop
                        ivar = loop_map[test_range]
                    else:
                        # Create new index variable
                        vtype = SymbolAttributes(BasicType.INTEGER)
                        ivar = sym.Variable(name=f'{ivar_basename}_{i}', type=vtype, scope=routine)
                    shape_index_map[(i, s)] = ivar
                    index_range_map[ivar] = _s

                    if ivar not in vdims:
                        vdims.append(ivar)

            # Add index variable to range replacement
            new_dims = as_tuple(shape_index_map.get((i, s), d)
                                for i, d, s in zip(count(), v.dimensions, as_tuple(v.shape)))
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
    def is_explicit_range_index(dim):
        # return False if assumed sized array or lower dimension equals to 1
        # return (isinstance(dim, sym.RangeIndex) and not dim.lower == 1 and not dim is None
        #             and not dim.lower is None and not dim.upper is None)
        return (isinstance(dim, sym.RangeIndex)
                and not (dim.lower == 1 or dim.lower is None or dim.upper is None))

    vmap = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if isinstance(v, sym.Array):
            # skip if e.g., `array(len)`, passed as `call routine(array)`
            if not v.dimensions:
                continue
            new_dims = []
            for i, d in enumerate(v.shape):
                if is_explicit_range_index(d):
                    if isinstance(v.dimensions[i], sym.RangeIndex):
                        start = simplify(v.dimensions[i].start - d.start + 1) if d.start is not None else None
                        stop = simplify(v.dimensions[i].stop - d.start + 1) if d.stop is not None else None
                        new_dims += [sym.RangeIndex((start, stop, d.step))]
                    else:
                        start = simplify(v.dimensions[i] - d.start + 1) if d.start is not None else None
                        new_dims += [start]
                else:
                    new_dims += [v.dimensions[i]]
            if new_dims:
                vmap[v] = v.clone(dimensions=as_tuple(new_dims))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    vmap = {}
    for v in routine.variables:
        if isinstance(v, sym.Array):
            new_dims = [sym.RangeIndex((1, simplify(d.upper - d.lower + 1)))
                if is_explicit_range_index(d) else d for d in v.dimensions]
            new_shape = [sym.RangeIndex((1, simplify(d.upper - d.lower + 1)))
                if is_explicit_range_index(d) else d for d in v.shape]
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
        if all(_dim == sym.RangeIndex((None, None)) for _dim in dim):
            return None
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

class LowerConstantArrayIndices(Transformation):
    """
    A transformation to pass/lower constant array indices down the call tree. 

    For example, the following code:

    .. code-block:: fortran

      subroutine driver(...)
        real, intent(inout) :: var(nlon,nlev,5,nb)
        do ibl=1,10
          call kernel(var(:, :, 1, ibl), var(:, :, 2:5, ibl))
        end do
      end subroutine driver

      subroutine kernel(var1, var2)
        real, intent(inout) :: var1(nlon, nlev)
        real, intent(inout) :: var2(nlon, nlev, 4)
        var1(:, :) = ...
        do jk=1,nlev
          do jl=1,nlon
            var1(jl, jk) = ...
            do jt=1,4
              var2(jl, jk, jt) = ...
            enddo
          enddo
        enddo
      end subroutine kernel

    is transformed to:

    .. code-block:: fortran

      subroutine driver(...)
        real, intent(inout) :: var(nlon,nlev,5,nb)
        do ibl=1,10
          call kernel(var(:, :, :, ibl), var(:, :, :, ibl))
        end do
      end subroutine driver

      subroutine kernel(var1, var2)
        real, intent(inout) :: var1(nlon, nlev, 5)
        real, intent(inout) :: var2(nlon, nlev, 5)
        var1(:, :, 1) = ...
        do jk=1,nlev
          do jl=1,nlon
            var1(jl, jk, 1) = ...
            do jt=1,4
              var2(jl, jk, jt + 2 + -1) = ...
            enddo
          enddo
        enddo
      end subroutine kernel

    Parameters
    ----------
    recurse_to_kernels: bool
        Recurse to kernels, thus lower constant array indices below the driver level for nested
        kernel calls (default: `True`).
    inline_external_only: bool
        Inline only external constant expressions or all of them (default: `False`)
    """

    # This trafo only operates on procedures
    item_filter = (ProcedureItem,)

    def __init__(self, recurse_to_kernels=True, inline_external_only=True):
        self.recurse_to_kernels = recurse_to_kernels
        self.inline_external_only = inline_external_only

    @staticmethod
    def explicit_dimensions(routine):
        """
        Make dimensions of arrays explicit within :any:`Subroutine` ``routine``.
        E.g., convert two-dimensional array ``arr2d`` to ``arr2d(:,:)`` or
        ``arr3d`` to ``arr3d(:,:,:)``.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to check
        """
        arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
        array_map = {}
        for array in arrays:
            if not array.dimensions:
                new_dimensions = (sym.RangeIndex((None, None)),) * len(array.shape)
                array_map[array] = array.clone(dimensions=new_dimensions)
        routine.body = SubstituteExpressions(array_map).visit(routine.body)

    @staticmethod
    def is_constant_dim(dim):
        """
        Check whether dimension dim is constant, thus, either a constant
        value or a constant range index.

        Parameters
        ----------
        dim: :py:class:`pymbolic.primitives.Expression`
        """
        if is_constant(dim):
            return True
        if isinstance(dim, sym.RangeIndex)\
                and all(child is not None and is_constant(child) for child in dim.children[:-1]):
            return True
        return False

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs['role']
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets', None)))
        if role == 'driver' or self.recurse_to_kernels:
            inline_constant_parameters(routine, external_only=self.inline_external_only)
            self.process(routine, targets)

    def process(self, routine, targets):
        """
        Process the driver and possibly kernels
        """
        dispatched_routines = ()
        offset_map = {}
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if str(call.name).lower() not in targets:
                continue
            # skip already dispatched routines but still update the call signature
            if call.routine in dispatched_routines:
                self.update_call_signature(call)
                continue
            # explicit array dimensions for the callee
            self.explicit_dimensions(call.routine)
            dispatched_routines += (call.routine,)
            # create the offset map and apply to call and callee
            offset_map[call.routine.name.lower()] = self.create_offset_map(call)
            self.process_callee(call.routine, offset_map[call.routine.name.lower()])
            self.update_call_signature(call)

    def update_call_signature(self, call):
        """
        Replace constant indices for call arguments being arrays with ':' and update the call.
        """
        new_args = [arg.clone(dimensions=\
                tuple(sym.RangeIndex((None, None)) if self.is_constant_dim(d) else d for d in arg.dimensions))\
                if isinstance(arg, sym.Array) else arg for arg in call.arguments]
        new_kwargs = [(kw[0], kw[1].clone(dimensions=\
                tuple(sym.RangeIndex((None, None)) if self.is_constant_dim(d) else d for d in kw[1].dimensions)))\
                if isinstance(kw[1], sym.Array) else kw for kw in call.kwarguments]
        call._update(arguments=as_tuple(new_args), kwarguments=as_tuple(new_kwargs))

    def create_offset_map(self, call):
        """
        Create map/dictionary for arguments with constant array indices.
        
        For, e.g., 

        integer :: arg(len1, len2, len3, len4)
        call kernel(..., arg(:, 2, 4:6, i), ...)

        offset_map[arg] = {
            0: (0, None, None),  # same index as before, no offset
            1: (None, 1, len2),  # New index, offset 1, size of the dimension is len2
            2: (1, 4, len3),     # Used to be position 1, offset 4, size of the dimension is len3
            3: (-1, None, None), # disregard as this is neither constant nor passed to callee
        }
        """
        offset_map = {}
        for routine_arg, call_arg in call.arg_iter():
            if not isinstance(routine_arg, sym.Array):
                continue
            offset_map[routine_arg.name] = {}
            current_index = 0
            for i, dim in enumerate(call_arg.dimensions):
                if self.is_constant_dim(dim):
                    if isinstance(dim, sym.RangeIndex):
                        # constant array index is e.g. '1:3' or '5:10'
                        offset_map[routine_arg.name][i] = (current_index, dim.children[0], call_arg.shape[i])
                    else:
                        # constant array index is e.g., '1' or '42'
                        offset_map[routine_arg.name][i] = (None, dim, call_arg.shape[i])
                        current_index -= 1
                else:
                    if not isinstance(dim, sym.RangeIndex):
                        # non constant array index is a variable e.g. 'jl'
                        offset_map[routine_arg.name][i] = (-1, None, None)
                        current_index -= 1
                    else:
                        # non constant array index is ':'
                        offset_map[routine_arg.name][i] = (current_index, None, None)
                current_index += 1
        return offset_map

    def process_callee(self, routine, offset_map):
        """
        Process/adapt the callee according to information in `offset_map`.

        Adapt the variable declarations and usage/indexing.
        """
        # adapt variable declarations, thus adapt the dimension and shape of the corresponding arguments
        vmap = {}
        variable_map = routine.variable_map
        for var_name in offset_map:
            var = variable_map[var_name]
            new_dims = ()
            for i in range(max(k for k, v in offset_map[var.name].items() if v != 0) + 1):
                original_index = offset_map[var_name][i][0]
                offset = offset_map[var_name][i][1]
                size = offset_map[var_name][i][2]
                if not (original_index is None or 0 <= original_index < len(var.dimensions)):
                    continue
                if offset is not None:
                    new_dims += (size,)
                else:
                    new_dims += (var.shape[original_index],)
            vmap[var] = var.clone(dimensions=new_dims, type=var.type.clone(shape=new_dims))
        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        # adapt the variable usage, thus the indexing/dimension
        vmap = {}
        for var in FindVariables(unique=False).visit(routine.body):
            if var.name in offset_map and var.dimensions is not None and var.dimensions:
                new_dims = ()
                for i in range(max(k for k, v in offset_map[var.name].items() if v != 0) + 1):
                    original_index = offset_map[var.name][i][0]
                    offset = offset_map[var.name][i][1]
                    if not (original_index is None or 0 <= original_index < len(var.dimensions)):
                        continue
                    if offset is not None:
                        if original_index is None:
                            new_dims += (offset,)
                        else:
                            new_dims += (var.dimensions[original_index] + offset - 1,)
                    else:
                        new_dims += (var.dimensions[original_index],)
                vmap[var] = var.clone(dimensions=new_dims)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)
