"""
Collection of utility routines to deal with common array indexing conversions.

"""
from collections import defaultdict
from itertools import count
import operator as op

from loki import info
from loki.analyse import defined_symbols_attached
from loki.expression import (
    symbols as sym, simplify, symbolic_op, FindVariables, SubstituteExpressions
)
from loki.ir import Assignment, Loop, Declaration
from loki.tools import as_tuple
from loki.types import SymbolType, BasicType
from loki.visitors import FindNodes, Transformer


__all__ = [
    'shift_to_zero_indexing', 'invert_array_indices',
    'resolve_vector_notation', 'normalize_range_indexing',
    'promote_variables', 'promote_nonmatching_variables',
    'promotion_dimensions_from_loop_nest'
]


def shift_to_zero_indexing(routine):
    """
    Shift all array indices to adjust to 0-based indexing conventions (eg. for C or Python)
    """
    vmap = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if isinstance(v, sym.Array):
            new_dims = []
            for d in v.dimensions.index_tuple:
                if isinstance(d, sym.RangeIndex):
                    start = d.start - sym.Literal(1) if d.start is not None else None
                    # no shift for stop because Python ranges are [start, stop)
                    new_dims += [sym.RangeIndex((start, d.stop, d.step))]
                else:
                    new_dims += [d - sym.Literal(1)]
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(as_tuple(new_dims)))
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
            rdim = as_tuple(reversed(v.dimensions.index_tuple))
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(rdim))
    routine.body = SubstituteExpressions(vmap).visit(routine.body)

    # Invert variable and argument dimensions for the automatic cast generation
    for v in routine.variables:
        if isinstance(v, sym.Array):
            rdim = as_tuple(reversed(v.dimensions.index_tuple))
            if v.shape:
                rshape = as_tuple(reversed(v.shape))
                vmap[v] = v.clone(dimensions=sym.ArraySubscript(rdim),
                                  type=v.type.clone(shape=rshape))
            else:
                vmap[v] = v.clone(dimensions=sym.ArraySubscript(rdim))
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

            ivar_basename = 'i_%s' % stmt.lhs.basename
            for i, dim, s in zip(count(), v.dimensions.index_tuple, as_tuple(v.shape)):
                if isinstance(dim, sym.RangeIndex):
                    # Create new index variable
                    vtype = SymbolType(BasicType.INTEGER)
                    ivar = sym.Variable(name='%s_%s' % (ivar_basename, i), type=vtype,
                                        scope=routine.scope)
                    shape_index_map[s] = ivar
                    index_range_map[ivar] = s

                    if ivar not in vdims:
                        vdims.append(ivar)

            # Add index variable to range replacement
            new_dims = as_tuple(shape_index_map.get(s, d)
                                for d, s in zip(v.dimensions.index_tuple, as_tuple(v.shape)))
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(new_dims))

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
            new_dims = [d.upper if is_one_index(d) else d for d in v.dimensions.index_tuple]
            new_shape = [d.upper if is_one_index(d) else d for d in v.shape]
            new_type = v.type.clone(shape=as_tuple(new_shape))
            vmap[v] = v.clone(dimensions=sym.ArraySubscript(as_tuple(new_dims)), type=new_type)
    routine.variables = [vmap.get(v, v) for v in routine.variables]


def promote_variables(routine, variable_names, pos, index=None, size=None):
    """
    Promote the given variables by inserting new array dimensions of given size.

    :param :class:``Subroutine`` routine:
            the subroutine to be modified.
    :param list variable_names:
            the list of (case-insensitive) variable names to be promoted.
    :param int pos:
            the position of the new array dimension using Python indexing
            convention (i.e., negative values count from end).
    :param index:
            (optional) the index expression to use for the new dimension when
            accessing/writing any of the variables.
    :type index:
            :class:``pymbolic.Expression`` or tuple of expressions
    :param size:
            (optional) the size of the new array dimension. If specified the
            given size is inserted into the variable shape and, as a
            consequence, variable declarations are updated accordingly.
    :type size:
            :class:``pymbolic.Expression`` or tuple of expressions

    NB: When specifying only ``index`` the declaration and declared shape of
        variables is not changed. Similarly, when specifying only ``size`` the
        use of variables is left unchanged.

    """
    variable_names = {name.lower() for name in variable_names}

    if not variable_names:
        return

    # Insert new index dimension
    if index is not None:
        index = as_tuple(index)
        index_vars = [{var.name.lower() for var in FindVariables().visit(i)} for i in index]

        # Create a copy of the tree and apply promotion in-place
        routine.body = Transformer().visit(routine.body)

        with defined_symbols_attached(routine):
            for node, var_list in FindVariables(unique=False, with_ir_node=True).visit(routine.body):

                var_list = [v for v in var_list if v.name.lower() in variable_names]

                if not var_list:
                    continue

                # We use the given index expression in this node if all
                # variables therein are defined, otherwise we use `:`
                node_index = tuple(i if v <= node.defined_symbols else sym.RangeIndex((None, None))
                                   for i, v in zip(index, index_vars))

                var_map = {}
                for var in var_list:
                    # If the position is given relative to the end we convert it to
                    # a positive index
                    var_dim = getattr(var, 'dimensions', sym.ArraySubscript(())).index
                    if pos < 0:
                        var_pos = len(var_dim) - pos + 1
                    else:
                        var_pos = pos

                    dimensions = sym.ArraySubscript(var_dim[:var_pos] + node_index + var_dim[var_pos:])
                    var_map[var] = var.clone(dimensions=dimensions)

                # need to apply update immediately because identical variable use
                # in other nodes might yield same hash but different substitution
                SubstituteExpressions(var_map, inplace=True).visit(node)

    # Apply shape promotion
    if size is not None:
        size = as_tuple(size)

        var_list = [var for decl in FindNodes(Declaration).visit(routine.spec)
                    for var in decl.variables if var.name.lower() in variable_names]

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

    :param list var_names:
        the names of the variables to consider for promotion.
    :param list loops:
        the list of nested loops, sorted from outermost to innermost.
    :param dict promotion_vars_dims:
        the mapping of variable names to promotion dimensions. When determining
        promotion dimensions for the variables in ``var_names`` this dict is
        checked for already existing promotion dimensions and, if not matching,
        the maximum of both is taken for each dimension.
    :param dict promotion_vars_index:
        the mapping of variable names to subscript expressions. These expressions
        are later inserted for every variable use. When the indexing expression
        for the loop nest does not match the existing expression in this dict,
        a ``RuntimeError`` is raised.

    :return: the updated mappings ``(promotion_vars_dims, promotion_vars_index)``.

    """
    # TODO: Would be nice to be able to promote this to the smallest possible dimension
    #       (in a loop var=start,end this is (end-start+1) with subscript index (var-start+1))
    #       but it requires being able to decide whether this yields a constant dimension,
    #       thus we need to stick to the upper bound for the moment as this is constant
    #       in our use cases.
    loop_lengths = [simplify(loop.bounds.stop) for loop in reversed(loops)]
    loop_index = [loop.variable for loop in reversed(loops)]

    for var_name in var_names:
        # Check if we have already marked this variable for promotion: let's make sure the added
        # dimensions are large enough for this loop (nest)
        if var_name not in promotion_vars_dims:
            promotion_vars_dims[var_name] = loop_lengths
            promotion_vars_index[var_name] = loop_index
        else:
            if len(promotion_vars_dims[var_name]) != len(loop_lengths):
                raise RuntimeError('Conflicting promotion dimensions for "{}"'.format(var_name))
            for i, (loop_length, index) in enumerate(zip(loop_lengths, loop_index)):
                if index != promotion_vars_index[var_name][i]:
                    raise RuntimeError('Loop variable "{}" does not match previous index "{}" for "{}"'.format(
                        str(index), str(promotion_vars_index[var_name][i]), var_name))
                if symbolic_op(promotion_vars_dims[var_name][i], op.lt, loop_length):
                    promotion_vars_dims[var_name][i] = loop_length

    return promotion_vars_dims, promotion_vars_index


def promote_nonmatching_variables(routine, promotion_vars_dims, promotion_vars_index, pos=-1):
    """
    Promote multiple variables with potentially non-matching promotion
    dimensions or index expressions.

    :param :class:``Subroutine`` routine:
        the subroutine to be modified.
    :param dict promotion_vars_dims:
        the mapping of variable names to promotion dimensions. The variables'
        shapes are expanded by these at position ``pos``.
    :param dict promotion_vars_index:
        the mapping of variable names to subscript expressions to be used whenever
        reading/writing the variable.

    This is a convenience routine for using ``promote_variables`` that groups variables by
    indexing expression and promotion dimensions to reduce the number of calls to
    ``promote_variables``.

    """
    if not promotion_vars_dims:
        return
    # Group promotion variables by index and size to reduce number of traversals for promotion
    index_size_var_map = defaultdict(list)
    for var_name, size in promotion_vars_dims.items():
        index_size_var_map[(as_tuple(promotion_vars_index[var_name]), as_tuple(size))] += [var_name]
    for (index, size), var_names in index_size_var_map.items():
        promote_variables(routine, var_names, pos, index=index, size=size)
    info('%s: promoted variable(s): %s', routine.name, ', '.join(promotion_vars_dims.keys()))
