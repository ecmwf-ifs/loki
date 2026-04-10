# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Utilities to manipulate vector notation in array expressions. """

from itertools import count

from loki.expression import symbols as sym, LokiIdentityMapper
from loki.frontend import HAVE_FP
from loki.ir import (
    nodes as ir, FindNodes, FindExpressions, Transformer,
    FindVariables, SubstituteExpressions, FindInlineCalls,
    FindLiteralLists
)
from loki.tools import as_tuple, dict_override, OrderedSet
from loki.types import SymbolAttributes, BasicType

from loki.transformations.utilities import (
    get_integer_variable, get_loop_bounds
)

if HAVE_FP:
    from fparser.two import Fortran2003


__all__ = [
    'remove_explicit_array_dimensions', 'add_explicit_array_dimensions',
    'resolve_vector_notation', 'resolve_vector_dimension',
    'ResolveVectorNotationTransformer'
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


def resolve_vector_notation(routine):
    """
    Resolve implicit vector notation by inserting explicit loops
    """

    # Find loops and map their range to the loop index variable
    loop_map = {
        sym.RangeIndex(loop.bounds.children): loop.variable
        for loop in FindNodes(ir.Loop).visit(routine.body)
    }

    transformer = ResolveVectorNotationTransformer(
        loop_map=loop_map, scope=routine, inplace=True,
        derive_qualified_ranges=True,
        map_unknown_ranges=True
    )
    routine.body = transformer.visit(routine.body)

    # Add declarations for all newly create loop index variables
    routine.variables += tuple(OrderedSet(transformer.index_vars))


def get_loop_bounds_test(routine, lower, upper):
    """
    Check loop bounds for a particular :any:`Dimension` in a
    :any:`Subroutine`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        Subroutine to perform checks on.
    dimension : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions
        used to define the data dimension and iteration space.
    """
    def get_valid(elem, variable_map):
        if isinstance(elem, str) and elem.isnumeric():
            return sym.Literal(int(elem))
        if elem in variable_map:
            return variable_map[elem]
        if elem.split('%', maxsplit=1)[0] in variable_map:
            return routine.resolve_typebound_var(elem, variable_map)
        return None

    bounds = ()
    variable_map = routine.variable_map
    # valid_lower = [get_valid(_lower, variable_map) for _lower in lower]
    # valid_lower = [_ for _ in valid_lower if _ is not None]
    # valid_upper = [get_valid(_upper, variable_map) for _upper in upper]
    # valid_upper = [_ for _ in valid_upper if _ is not None]
    valid_lower = []
    for _lower in lower:
        try:
            valid_lower.append(get_valid(_lower, variable_map))
        except Exception as e:
            pass
    valid_upper = []
    for _upper in upper:
        try:
            valid_upper.append(get_valid(_upper, variable_map))
        except Exception as e:
            pass
    valid_lower = [_ for _ in valid_lower if _ is not None]
    valid_upper = [_ for _ in valid_upper if _ is not None]
   
    for _lower in valid_lower:
        for _upper in valid_upper:
            bounds += ((_lower, _upper),)
    return bounds

def resolve_vector_dimension(routine, dimension, derive_qualified_ranges=False):
    """
    Resolve vector notation for a given dimension only. The dimension
    is defined by a loop variable and the bounds of the given range.

    Unliked the related :meth:`resolve_vector_notation` utility, this
    will only resolve the defined dimension according to ``bounds``
    and ``loop_variable``.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to resolve vector notation usage.
    dimension : :any:`Dimension`
        Dimension object that defines the dimension to resolve
    derive_qualified_ranges : bool
        Flag to enable the derivation of (all) range bounds from
        shape information.
    """
    # Find the iteration index variable and bound variables
    index = get_integer_variable(routine, name=dimension.index)
    
    _lower = as_tuple(dimension.lower) + ('1',)
    _upper = as_tuple(dimension.upper) + as_tuple(dimension.sizes)
    bounds = get_loop_bounds_test(routine, lower=_lower, upper=_upper)

    # Map any range indices to the given loop index variable
    loop_map = {sym.RangeIndex(_bounds): index for _bounds in bounds}

    transformer = ResolveVectorNotationTransformer(
        loop_map=loop_map, scope=routine, inplace=True,
        derive_qualified_ranges=derive_qualified_ranges,
        map_unknown_ranges=False
    )
    routine.body = transformer.visit(routine.body)

    # Add declarations for all newly create loop index variables
    routine.variables += tuple(OrderedSet(transformer.index_vars))


class IterationRangeShapeMapper(LokiIdentityMapper):
    """
    A mapper that derives the fully qualified iteration dimension for
    unbounded :any:`RangeIndex` indices in array expressions.
    """

    @staticmethod
    def _shape_to_range(s):
        return sym.RangeIndex(
            (s.lower, s.upper, s.step) if isinstance(s, sym.Range) else (sym.IntLiteral(1), s)
        )

    def map_array(self, expr, *args, **kwargs):
        """ Replace ``:`` range indices with ``1:shape`` vector indices """

        # Resolve implicit range indices if we know the shape
        if not expr.dimensions and expr.shape:
            expr = expr.clone(dimensions=tuple(sym.RangeIndex((None, None)) for _ in expr.shape))

        # Derive fully qualified bounds for ``:``
        new_dims = tuple(
            self._shape_to_range(s) if isinstance(d, sym.RangeIndex) and d == ':' else d
            for i, d, s in zip(count(), expr.dimensions, as_tuple(expr.shape))
        )
        # make sure it is not a inline call that was misread as array access ...
        if new_dims:
            return expr.clone(dimensions=new_dims)
        return expr


class IterationRangeIndexMapper(LokiIdentityMapper):
    """
    A mapper that replaces fully qualified :any:`RangeIndex` symbols
    with discrete loop indices and collects the according
    ``index_to_range_map``.

    This takes mapping of known loop indices for a set of ranges and will
    use these variables if it encounters a matching index range. If not it
    will create new index variables using the given scope and ``basename``.
    The flag ``map_unknown_ranges`` can be used to toggle the
    automatic generation of generic indices from qualified range
    symbols.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to check
    loop_map : dict of :any:`RangeIndex` to :any:`Scalar`
        Map of known loop indices for given ranges
    basename : str
        Base name string for new iteration variables
    scope : :any:`Subroutine` or :any:`Module`
        Scope in which to create potential new iteration index symbols
    map_unknown_ranges : bool
        Flag to indicate whether range indices not encountered in ``loop_map``
        should be should be remapped to generic loop indices.
    """

    def __init__(
            self, *args, loop_map=None, basename=None, scope=None,
            map_unknown_ranges=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loop_map = loop_map or {}
        self.basename = basename if basename else 'i'
        self.scope = scope
        self.map_unknown_ranges = map_unknown_ranges

        self.index_range_map = {}

    def map_array(self, expr, *args, **kwargs):

        shape_index_map = {}
        for i, dim in zip(count(), expr.dimensions):
            if isinstance(dim, sym.RangeIndex):
                # See if index variable is knwon for this loop range
                if dim in self.loop_map:
                    ivar = self.loop_map[dim]
                else:
                    # Skip if we're not supposed to create new indices
                    if not self.map_unknown_ranges or dim == sym.RangeIndex((None, None)):
                        continue
                
                    # Create new index variable
                    vtype = SymbolAttributes(BasicType.INTEGER)
                    ivar = sym.Variable(name=f'{self.basename}_{i}', type=vtype, scope=self.scope)
                shape_index_map[(i, dim)] = ivar
                self.index_range_map[ivar] = dim

        # Add index variable to range replacement
        new_dims = as_tuple(
            shape_index_map.get((i, d), d) for i, d in zip(count(), expr.dimensions)
        )
        return expr.clone(dimensions=new_dims)



class ResolveVectorNotationTransformer(Transformer):
    """
    A :any:`Transformer` that resolves implicit vector notation by
    inserting explicit loops.

    Parameters
    ----------
    loop_map : dict of tuple to :any:`Variable`
        A dict mapping the tuple ``(lower, upper, step)`` to
        a known variable symbol to use as loop index.
    scope : :any:`Subroutine` or :any:`Module`
        The scope in which to create new loop index variables
    derive_qualified_ranges : bool
        Derive explicit bounds for all unqualified index ranges
        (``:``) before resolving them with loops.
    map_unknown_ranges : bool
        Flag to indicate whether unknown, but fully qualified range
        indices are to be remapped to loops.
    """

    def __init__(
            self, *args, loop_map=None, scope=None,
            derive_qualified_ranges=True, map_unknown_ranges=True,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.scope = scope
        self.loop_map = {} if loop_map is None else loop_map
        self.index_vars = OrderedSet()

        self.map_unknown_ranges = map_unknown_ranges
        self.derive_qualified_ranges = derive_qualified_ranges
        self.infer_iteration_shape = True

    def visit_Assignment(self, stmt, **kwargs):  # pylint: disable=unused-argument
        
        # early exit since pointer assignment
        if stmt.ptr:
            return stmt

        # early exit since lhs is not an array
        if not isinstance(stmt.lhs, sym.Array):
            return stmt

        # early exit since rhs is/has and literal list
        rhs_literal_lists = FindLiteralLists().visit(stmt.rhs)
        if rhs_literal_lists:
            return stmt

        create_loops = kwargs.get('create_loops', True)

        # check for forbidden calls in the rhs
        inline_calls = [(_.name).lower() for _ in FindInlineCalls().visit(stmt.rhs)]
        forbidden_ops = ['present', 'sum']
        if any(op in inline_calls for op in forbidden_ops):
            return stmt
        if HAVE_FP:
            if any(redux_op in FindExpressions().visit(stmt.rhs)
                   for redux_op in Fortran2003.Intrinsic_Name.array_reduction_names):
                return stmt

        # replace all unbounded ranges with bounded ranges based on array shape
        if self.derive_qualified_ranges:
            shape_mapper = IterationRangeShapeMapper()
            stmt._update(lhs=shape_mapper(stmt.lhs), rhs=shape_mapper(stmt.rhs))
       
        # find all arrays in the rhs
        rhs_vars = FindVariables(unique=False).visit(stmt.rhs)
        arrays = [var for var in rhs_vars if isinstance(var, sym.Array) and any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions)]
        # get the corresponding array dimensions
        arrays_dims = [array.dimensions for array in arrays]
        # get the indices for each array dimensions being a range index, e.g., ':' or '1:n'
        arrays_dims_gri = [[i for i, dim in enumerate(dims) if isinstance(dim, sym.RangeIndex)] for dims in arrays_dims]
        # could be some nested arrays, e.g. arr1(1:n, arr2(1:m))
        if len(arrays_dims_gri) > 1 and not all(len(array_dim_gri) == len(arrays_dims_gri[0]) for array_dim_gri in arrays_dims_gri[1:]):
            return stmt
        
        # get the lhs array
        lhs_array = stmt.lhs
        # and the corresponding dimensions
        lhs_dims = lhs_array.dimensions
        # get the indices for the lhs array being a range index e.g., ':' or '1:n'
        lhs_dims_gri = [i for i, dim in enumerate(lhs_dims) if isinstance(dim, sym.RangeIndex)]
        # exclude the generic range indices: ':' 
        lhs_dims_ri = [i for i, j in enumerate(lhs_dims_gri) if lhs_dims[j] != sym.RangeIndex((None, None))]

        # allow and imply ranges for ":" on the rhs
        # TODO: make this an option
        if True:
            rel_dim_indices = lhs_dims_ri
        else:
            rel_dim_indices = [i for i, j in enumerate(lhs_dims_ri) if all(array_dims[array_dims_gri[i]] != sym.RangeIndex((None, None))
                for array_dims, array_dims_gri in zip(arrays_dims, arrays_dims_gri))]

        # nothing to do here, therefore return stmt as is
        if not rel_dim_indices:
            return stmt

        def map_dims(dims, loop_map, map_unknown_ranges=True, basename='i', scope=None):
            index_range_map = {}
            shape_index_map = {}
            for i, dim in zip(count(), dims):
                if isinstance(dim, sym.RangeIndex):
                    # See if index variable is knwon for this loop range
                    if dim in loop_map:
                        ivar = loop_map[dim]
                    else:
                        # Skip if we're not supposed to create new indices
                        if not map_unknown_ranges or dim == sym.RangeIndex((None, None)):
                            continue
                        vtype = SymbolAttributes(BasicType.INTEGER)
                        ivar = sym.Variable(name=f'{basename}_{i}', type=vtype, scope=scope)
                    shape_index_map[(i, dim)] = ivar
                    index_range_map[ivar] = dim
            # Add index variable to range replacement
            new_dims = as_tuple(
                shape_index_map.get((i, d), d) for i, d in zip(count(), dims)
            )
            return new_dims, index_range_map
        
        def _shift(lhs, lhs_range, rhs_range):
            _sum = sym.Product((-1, lhs_range.lower))
            _sum = sym.Sum((lhs, _sum, rhs_range.lower))
            return _sum # TODO: call simplify

        # get the relevant dimensions from the lhs
        rel_lhs_dims = [lhs_dims[lhs_dims_gri[i]] for i in rel_dim_indices]
        # derive new relevant lhs dims
        new_lhs_dims, index_range_map = map_dims(rel_lhs_dims, self.loop_map, scope=self.scope, basename=f'i_{stmt.lhs.basename}')
        # map this to the rhs arrays
        rel_arrays_rhs_dims = [[array_dims[array_dims_gri[i]] for i in rel_dim_indices] for array_dims, array_dims_gri in zip(arrays_dims, arrays_dims_gri)]
        new_rel_arrays_rhs_dims = []
        for array, rel_rhs_dims in zip(arrays, rel_arrays_rhs_dims): 
            new_rel_rhs_dims = []
            for i, lhs_dim, new_lhs_dim, rhs_dim in zip(count(), rel_lhs_dims, new_lhs_dims, rel_rhs_dims):
                if lhs_dim == rhs_dim or rhs_dim == sym.RangeIndex((None, None)) or isinstance(rhs_dim, sym.RangeIndex) and rhs_dim.lower == 1:
                    new_rel_rhs_dims.append(new_lhs_dim)
                else:
                    new_rel_rhs_dims.append(_shift(new_lhs_dim, lhs_dim, rhs_dim))
            new_rel_arrays_rhs_dims.append(new_rel_rhs_dims)

        # create new array for lhs
        _new_lhs_arr_dims = list(lhs_dims)
        for i, d in enumerate(new_lhs_dims):
            _new_lhs_arr_dims[lhs_dims_gri[rel_dim_indices[i]]] = d
        new_lhs_arr = lhs_array.clone(dimensions=as_tuple(_new_lhs_arr_dims))

        # create new array(s) for rhs
        new_arrays = []
        for i_arr, _array in enumerate(arrays):
            _new_arr_dims = list(arrays_dims[i_arr])
            for i, d in enumerate(new_rel_arrays_rhs_dims[i_arr]):
                # _new_arr_dims[array_dims_gri[rel_dim_indices[i]]] = d
                _new_arr_dims[arrays_dims_gri[i_arr][rel_dim_indices[i]]] = d
            new_arrays.append(_array.clone(dimensions=as_tuple(_new_arr_dims)))

        # FINALLY: update the statement
        stmt._update(lhs=new_lhs_arr, rhs=SubstituteExpressions({old: new for old, new in zip(arrays, new_arrays)}).visit(stmt.rhs))

        # Record all newly create loop index variables,
        # so that we can declare them in the outer context
        self.index_vars.update(list(index_range_map.keys()))

        # Recursively build new loop nest over all implicit dims
        if create_loops and len(index_range_map):
            loop = None
            body = stmt
            for ivar, irange in index_range_map.items():
                if isinstance(irange, sym.RangeIndex):
                    bounds = sym.LoopRange(irange.children)
                else:
                    bounds = sym.LoopRange((sym.Literal(1), irange, sym.Literal(1)))
                loop = ir.Loop(variable=ivar, body=as_tuple(body), bounds=bounds)
                body = loop

            return (ir.Comment('! loki resolved vector notation'), loop)

        # No vector dimensions encountered, return unchanged
        return stmt

    def visit_MaskedStatement(self, masked, **kwargs):  # pylint: disable=unused-argument
        # TODO: Currently limited to simple, single-clause WHERE stmts
        # assert len(masked.conditions) == 1 and len(masked.bodies) == 1
        if not len(masked.conditions) == 1 and len(masked.bodies) == 1:
            return masked

        # Replace all unbounded ranges with bounded ranges based on array shape
        conditions = masked.conditions
        if self.derive_qualified_ranges:
            conditions = IterationRangeShapeMapper()(conditions)

        index_mapper = IterationRangeIndexMapper(
            loop_map=self.loop_map, scope=self.scope,
            map_unknown_ranges=self.map_unknown_ranges
        )
        conditions = index_mapper(conditions)
        index_range_map = index_mapper.index_range_map

        with dict_override(kwargs, {'create_loops': False}):
            bodies = self.visit(masked.bodies, **kwargs)
            else_body = self.visit(masked.default, **kwargs)

        # Rebuild construct as an IF conditional inside a loop over the range bounds
        if not index_range_map:
            return masked

        idx_range = list(index_range_map.values())[0]
        bounds = sym.LoopRange((idx_range.start, idx_range.stop, idx_range.step))
        cond = ir.Conditional(
            condition=conditions[0], body=bodies, else_body=else_body
        )

        # Recursively build new loop nest over all implicit dims
        if len(index_range_map):
            loop = None
            body = cond
            for ivar, irange in index_range_map.items():
                if isinstance(irange, sym.RangeIndex):
                    bounds = sym.LoopRange(irange.children)
                else:
                    bounds = sym.LoopRange((sym.Literal(1), irange, sym.Literal(1)))
                loop = ir.Loop(variable=ivar, body=as_tuple(body), bounds=bounds)
                body = loop
            return loop

        return masked
