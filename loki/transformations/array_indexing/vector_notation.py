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
    FindVariables, SubstituteExpressions, FindInlineCalls
)
from loki.tools import as_tuple, dict_override
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
    )
    routine.body = transformer.visit(routine.body)

    # Add declarations for all newly create loop index variables
    routine.variables += tuple(set(transformer.index_vars))


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
    bounds = get_loop_bounds(routine, dimension=dimension)

    # Map any range indices to the given loop index variable
    loop_map = {sym.RangeIndex(bounds): index}

    transformer = ResolveVectorNotationTransformer(
        loop_map=loop_map, scope=routine, inplace=True,
        derive_qualified_ranges=derive_qualified_ranges,
        map_unknown_ranges=False
    )
    routine.body = transformer.visit(routine.body)

    # Add declarations for all newly create loop index variables
    routine.variables += tuple(set(transformer.index_vars))


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
                    if not self.map_unknown_ranges:
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
        self.index_vars = set()

        self.map_unknown_ranges = map_unknown_ranges
        self.derive_qualified_ranges = derive_qualified_ranges

    def visit_Assignment(self, stmt, **kwargs):  # pylint: disable=unused-argument
        create_loops = kwargs.get('create_loops', True)

        if HAVE_FP:
            if any(redux_op in FindExpressions().visit(stmt.rhs)
                   for redux_op in Fortran2003.Intrinsic_Name.array_reduction_names):
                return stmt

        # Replace all unbounded ranges with bounded ranges based on array shape
        if self.derive_qualified_ranges:
            shape_mapper = IterationRangeShapeMapper()
            stmt._update(lhs=shape_mapper(stmt.lhs), rhs=shape_mapper(stmt.rhs))

        # Replace all range indices with loop indices and collect the corresponding mapping
        index_mapper = IterationRangeIndexMapper(
            loop_map=self.loop_map, basename=f'i_{stmt.lhs.basename}', scope=self.scope,
            map_unknown_ranges=self.map_unknown_ranges
        )
        stmt._update(lhs=index_mapper(stmt.lhs), rhs=index_mapper(stmt.rhs))

        # Record all newly create loop index variables,
        # so that we can declare them in the outer context
        index_range_map = index_mapper.index_range_map
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

            return loop

        # No vector dimensions encountered, return unchanged
        return stmt

    def visit_MaskedStatement(self, masked, **kwargs):  # pylint: disable=unused-argument
        # TODO: Currently limited to simple, single-clause WHERE stmts
        assert len(masked.conditions) == 1 and len(masked.bodies) == 1

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
