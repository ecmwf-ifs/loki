# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

""" Utilities to manipulate vector notation in array expressions. """

from itertools import count

from loki.expression import symbols as sym, LokiIdentityMapper
from loki.expression.mappers import ExpressionRetriever
from loki.expression.symbolic import simplify
from loki.frontend import HAVE_FP
from loki.logging import warning
from loki.ir import (
    nodes as ir, FindNodes, FindExpressions, Transformer,
    FindVariables, SubstituteExpressions, FindInlineCalls,
    FindLiteralLists, ExpressionFinder
)
from loki.tools import as_tuple, dict_override, OrderedSet
from loki.types import SymbolAttributes, BasicType

from loki.transformations.utilities import get_integer_variable

if HAVE_FP:
    from fparser.two import Fortran2003


__all__ = [
    'remove_explicit_array_dimensions', 'add_explicit_array_dimensions',
    'resolve_vector_notation', 'resolve_vector_dimension',
    'ResolveVectorNotationTransformer'
]


class _OutermostVarRetriever(ExpressionRetriever):  # pylint: disable=abstract-method
    """
    Like :class:`ExpressionRetriever` but does not recurse into
    parent chains of derived-type member symbols.

    Standard :any:`FindVariables` traverses ``VariableSymbol.parent``
    links, so visiting ``ydg%yrdimv%nflevg`` yields three ``Scalar``
    nodes: ``ydg``, ``ydg%yrdimv``, and ``ydg%yrdimv%nflevg``.
    This retriever skips the parent recursion and therefore returns
    only the outermost (longest-chain) symbol for each derived-type
    access expression.
    """

    def map_variable_symbol(self, expr, *args, **kwargs):
        if not self.visit(expr):
            return
        # Do NOT recurse into expr.parent — stop at the outermost symbol.
        self.post_visit(expr, *args, **kwargs)

    map_deferred_type_symbol = map_variable_symbol


class _FindOutermostVariables(ExpressionFinder):
    """
    Like :any:`FindVariables` but returns only the outermost
    (longest-chain) variable for each derived-type member access.

    For an expression containing ``ydg%yrdimv%nflevg``, standard
    :any:`FindVariables` returns ``{ydg, ydg%yrdimv,
    ydg%yrdimv%nflevg}``; this class returns only
    ``{ydg%yrdimv%nflevg}``.
    """
    retriever = _OutermostVarRetriever(
        lambda e: isinstance(e, (sym.Scalar, sym.Array, sym.DeferredTypeSymbol))
    )


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


def resolve_vector_notation(routine, resolve_implicit_rhs_ranges=True,
                            substitute_derived_type_bounds=False,
                            insert_comments=False):
    """
    Resolve implicit vector notation by inserting explicit loops.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine in which to resolve vector notation usage.
    resolve_implicit_rhs_ranges : bool
        When ``True`` (default), resolve all LHS range dimensions even
        if the corresponding RHS arrays use bare ``:`` ranges.
    substitute_derived_type_bounds : bool
        When ``True``, replace derived-type member references in
        synthesized loop bounds with plain scalar variables.  Only
        needed for driver routines.  Defaults to ``False``.
    insert_comments : bool
        When ``True``, insert a ``! loki resolved vector notation``
        comment before each generated loop nest.  Defaults to ``False``.
    """

    # Find loops and map their range to the loop index variable
    loop_map = {
        sym.RangeIndex(loop.bounds.children): loop.variable
        for loop in FindNodes(ir.Loop).visit(routine.body)
    }

    transformer = ResolveVectorNotationTransformer(
        loop_map=loop_map, scope=routine, inplace=True,
        derive_qualified_ranges=True,
        map_unknown_ranges=True,
        resolve_implicit_rhs_ranges=resolve_implicit_rhs_ranges,
        substitute_derived_type_bounds=substitute_derived_type_bounds,
        insert_comments=insert_comments,
    )
    routine.body = transformer.visit(routine.body)

    # Prepend any scalar extraction assignments (from substitute_derived_type_bounds)
    # to the top of the routine body so they appear before any acc regions.
    if transformer.pre_body_stmts:
        routine.body.prepend(tuple(transformer.pre_body_stmts))

    # Add declarations for all newly create loop index variables
    routine.variables += tuple(OrderedSet(transformer.index_vars))


def _get_all_valid_loop_bounds(routine, lower, upper):
    """
    Find all valid combinations of loop bounds from candidate lists.

    For each candidate in ``lower`` and ``upper``, checks whether the
    variable exists in the routine's scope (or is a numeric literal).
    Returns the cross-product of all valid lower/upper pairs.

    Parameters
    ----------
    routine : :any:`Subroutine`
        Subroutine whose variable scope is used to validate bound names.
    lower : tuple of str
        Candidate lower bound variable names or numeric strings.
    upper : tuple of str
        Candidate upper bound variable names or numeric strings.

    Returns
    -------
    tuple of tuple
        Each inner tuple is ``(lower_expr, upper_expr)`` as resolved
        expression nodes.
    """
    variable_map = routine.variable_map
    def get_valid(elem):
        if isinstance(elem, str) and elem.isnumeric():
            return sym.Literal(int(elem))
        if elem.split('%', maxsplit=1)[0] in variable_map:
            return routine.resolve_typebound_var(elem, variable_map)
        return None

    bounds = ()
    valid_lower = [valid for _lower in lower if (valid := get_valid(_lower)) is not None]
    valid_upper = [valid for _upper in upper if (valid := get_valid(_upper)) is not None]

    for _lower in valid_lower:
        for _upper in valid_upper:
            bounds += ((_lower, _upper),)
    return bounds

def resolve_vector_dimension(routine, dimension, derive_qualified_ranges=False,
                             resolve_implicit_rhs_ranges=True,
                             substitute_derived_type_bounds=False,
                             insert_comments=False):
    """
    Resolve vector notation for a given dimension only. The dimension
    is defined by a loop variable and the bounds of the given range.

    Unlike the related :meth:`resolve_vector_notation` utility, this
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
    resolve_implicit_rhs_ranges : bool
        When ``True`` (default), resolve all LHS range dimensions even
        if the corresponding RHS arrays use bare ``:`` ranges.
    substitute_derived_type_bounds : bool
        When ``True``, replace derived-type member references in
        synthesized loop bounds with plain scalar variables.  Only
        needed for driver routines.  Defaults to ``False``.
    insert_comments : bool
        When ``True``, insert a ``! loki resolved vector notation``
        comment before each generated loop nest.  Defaults to ``False``.
    """
    # Find the iteration index variable and bound variables
    index = get_integer_variable(routine, name=dimension.index)

    _lower = as_tuple(dimension.lower) + ('1',)
    _upper = as_tuple(dimension.upper) + as_tuple(dimension.sizes)
    bounds = _get_all_valid_loop_bounds(routine, lower=_lower, upper=_upper)

    if not bounds:
        warning(
            f'[resolve_vector_dimension] No valid loop bounds found for dimension '
            f'"{dimension.name}" in routine "{routine.name}". No transformation applied.'
        )
        return

    # Map any range indices to the given loop index variable
    loop_map = {sym.RangeIndex(_bounds): index for _bounds in bounds}

    transformer = ResolveVectorNotationTransformer(
        loop_map=loop_map, scope=routine, inplace=True,
        derive_qualified_ranges=derive_qualified_ranges,
        map_unknown_ranges=False,
        resolve_implicit_rhs_ranges=resolve_implicit_rhs_ranges,
        substitute_derived_type_bounds=substitute_derived_type_bounds,
        insert_comments=insert_comments,
    )
    routine.body = transformer.visit(routine.body)

    # Prepend any scalar extraction assignments (from substitute_derived_type_bounds)
    # to the top of the routine body so they appear before any acc regions.
    if transformer.pre_body_stmts:
        routine.body.prepend(tuple(transformer.pre_body_stmts))

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

    @staticmethod
    def _shape_lower(s):
        return s.lower if isinstance(s, sym.Range) else sym.IntLiteral(1)

    @staticmethod
    def _shape_upper(s):
        return s.upper if isinstance(s, sym.Range) else s

    def map_array(self, expr, *args, **kwargs):
        """ Replace ``:`` range indices with ``1:shape`` vector indices """

        # Resolve implicit range indices if we know the shape
        if not expr.dimensions and expr.shape:
            expr = expr.clone(dimensions=tuple(sym.RangeIndex((None, None)) for _ in expr.shape))

        # Derive fully qualified bounds for ``:``
        new_dims = ()
        for d, s in zip(expr.dimensions, as_tuple(expr.shape)):
            if isinstance(d, sym.RangeIndex) and d == ':':
                new_dims += (self._shape_to_range(s),)
            elif isinstance(d, sym.RangeIndex) and d.upper is None:
                new_dims += (sym.RangeIndex(
                    (d.lower, self._shape_upper(s), d.step)
                ),)
            elif isinstance(d, sym.RangeIndex) and d.lower is None:
                new_dims += (sym.RangeIndex(
                    (self._shape_lower(s), d.upper, d.step)
                ),)
            else:
                new_dims += (d,)
        # make sure it is not a inline call that was misread as array access ...
        if new_dims:
            return expr.clone(dimensions=new_dims)
        return expr




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
    resolve_implicit_rhs_ranges : bool
        When ``True`` (default), resolve all LHS range dimensions even
        if the corresponding RHS arrays use bare ``:`` (unqualified)
        ranges. When ``False``, only resolve dimensions where all RHS
        arrays have explicit (qualified) ranges.
    substitute_derived_type_bounds : bool
        When ``True``, replace derived-type member references in
        synthesized loop bounds with existing or newly created plain
        scalar variables (see :meth:`_substitute_derived_type_bounds`).
        This is intended for **driver** routines where device-safe plain
        scalars are required.  Defaults to ``False``; kernels should
        leave derived-type bounds as-is.
    insert_comments : bool
        When ``True``, insert a ``! loki resolved vector notation``
        comment before each generated loop nest.  Defaults to ``False``.
    """

    def __init__(
            self, *args, loop_map=None, scope=None,
            derive_qualified_ranges=True, map_unknown_ranges=True,
            resolve_implicit_rhs_ranges=True,
            substitute_derived_type_bounds=False,
            insert_comments=False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.scope = scope
        self.loop_map = {} if loop_map is None else loop_map
        self.index_vars = OrderedSet()
        self.pre_body_stmts = []
        self.active_loop_vars = set()

        self.map_unknown_ranges = map_unknown_ranges
        self.derive_qualified_ranges = derive_qualified_ranges
        self.resolve_implicit_rhs_ranges = resolve_implicit_rhs_ranges
        self.substitute_derived_type_bounds_flag = substitute_derived_type_bounds
        self.insert_comments = insert_comments
        self.infer_iteration_shape = True

        # Build a lookup of existing scalar assignments of the form:
        #   SCALAR = DERIVED_TYPE_MEMBER  (e.g., KLEVS = KDIM%KLEVS)
        # Keys are canonical (lowercased) string forms of the RHS expression.
        # Used by _substitute_derived_type_bounds to replace shape-derived
        # loop bounds that reference derived-type members.
        self._scalar_assignment_map = {}
        if scope is not None:
            for assign in FindNodes(ir.Assignment).visit(scope.body):
                rhs = assign.rhs
                lhs = assign.lhs
                # Only record simple scalar = derived-type-member assignments.
                # rhs may be Scalar or DeferredTypeSymbol depending on whether
                # the derived type definition is available during parsing.
                if (isinstance(lhs, sym.Scalar) and lhs.parent is None
                        and isinstance(rhs, (sym.Scalar, sym.DeferredTypeSymbol))
                        and rhs.parent is not None):
                    self._scalar_assignment_map[str(rhs).lower().replace(' ', '')] = lhs

    @staticmethod
    def _find_range_positions(dims):
        """Return list of positions in ``dims`` that are :any:`RangeIndex`."""
        return [i for i, dim in enumerate(dims) if isinstance(dim, sym.RangeIndex)]

    @classmethod
    def _find_qualified_range_positions(cls, dims, range_positions):
        """
        Return ordinal indices into ``range_positions`` whose corresponding
        dimension is *not* a bare ``(:)`` (i.e. ``RangeIndex((None, None))``).
        """
        return cls._find_resolvable_range_positions(dims, range_positions)

    @staticmethod
    def _has_explicit_range_bounds(dim):
        return isinstance(dim, sym.RangeIndex) and dim.lower is not None and dim.upper is not None

    @classmethod
    def _is_scalarizable_bound_expr(cls, bound):
        # This intentionally keys off vector-valuedness only. Unresolved
        # derived-type members may still be scalar bounds (e.g. DIMS%KLON), so
        # rejecting DeferredTypeSymbol conservatively regresses valid cases.
        return not cls._expr_contains_vector_array(bound)

    @classmethod
    def _is_resolvable_range_dim(cls, dim):
        return (
            cls._has_explicit_range_bounds(dim)
            and cls._is_scalarizable_bound_expr(dim.lower)
            and cls._is_scalarizable_bound_expr(dim.upper)
        )

    @classmethod
    def _find_resolvable_range_positions(cls, dims, range_positions):
        """
        Return ordinal indices into ``range_positions`` that have explicit,
        scalarizable bounds.
        """
        return [
            i for i, j in enumerate(range_positions)
            if cls._is_resolvable_range_dim(dims[j])
        ]

    def _get_range_resolution_info(
            self, lhs_dims, rhs_arrays, qualification_lhs_dims=None,
            qualification_rhs_arrays=None
    ):
        """
        Collect range positions and resolvable dimension ordinals for a set of arrays.
        """
        rhs_dims_per_array = [array.dimensions for array in rhs_arrays]
        qualification_rhs_dims_per_array = None
        if qualification_rhs_arrays is not None:
            qualification_rhs_dims_per_array = [array.dimensions for array in qualification_rhs_arrays]

        lhs_range_positions, resolvable_dim_indices = self._get_resolvable_dim_indices(
            lhs_dims, rhs_dims_per_array,
            qualification_lhs_dims=qualification_lhs_dims,
            qualification_rhs_dims_per_array=qualification_rhs_dims_per_array,
        )
        rhs_range_positions_per_array = [
            self._find_range_positions(dims) for dims in rhs_dims_per_array
        ]
        return lhs_range_positions, rhs_dims_per_array, rhs_range_positions_per_array, resolvable_dim_indices

    def _get_resolvable_dim_indices(
            self, lhs_dims, rhs_dims_per_array, qualification_lhs_dims=None,
            qualification_rhs_dims_per_array=None
    ):
        """
        Return ordinal positions into the LHS range list that are safe to resolve.
        """
        lhs_range_positions = self._find_range_positions(lhs_dims)

        if self.resolve_implicit_rhs_ranges:
            lhs_qualified_positions = self._find_resolvable_range_positions(
                lhs_dims, lhs_range_positions
            )
            return lhs_range_positions, lhs_qualified_positions

        qualification_lhs_dims = lhs_dims if qualification_lhs_dims is None else qualification_lhs_dims
        qualification_lhs_positions = self._find_range_positions(qualification_lhs_dims)
        lhs_qualified_positions = self._find_resolvable_range_positions(
            qualification_lhs_dims, qualification_lhs_positions
        )

        qualification_rhs_dims_per_array = (
            rhs_dims_per_array if qualification_rhs_dims_per_array is None
            else qualification_rhs_dims_per_array
        )
        qualification_rhs_positions_per_array = [
            self._find_range_positions(dims) for dims in qualification_rhs_dims_per_array
        ]
        rhs_qualified_positions_per_array = [
            self._find_resolvable_range_positions(rhs_dims, rhs_pos)
            for rhs_dims, rhs_pos in zip(
                qualification_rhs_dims_per_array, qualification_rhs_positions_per_array
            )
        ]

        if not rhs_qualified_positions_per_array:
            return lhs_range_positions, lhs_qualified_positions

        resolvable_dim_indices = [
            j for j in lhs_qualified_positions
            if all(
                j in rhs_qualified for rhs_qualified in rhs_qualified_positions_per_array
            )
        ]
        return lhs_range_positions, resolvable_dim_indices

    @staticmethod
    def _build_loop_nest(index_range_map, body, insert_comments=False):
        """Wrap ``body`` in loops for all ranges from ``index_range_map``."""
        loop = None
        wrapped_body = body
        for ivar, irange in index_range_map.items():
            if isinstance(irange, sym.RangeIndex):
                bounds = sym.LoopRange(irange.children)
            else:
                bounds = sym.LoopRange((sym.Literal(1), irange, sym.Literal(1)))
            loop = ir.Loop(variable=ivar, body=as_tuple(wrapped_body), bounds=bounds)
            wrapped_body = loop

        if insert_comments and loop:
            return (ir.Comment('! loki resolved vector notation'), loop)
        if loop:
            return (loop,)
        return body

    def visit_Loop(self, o, **kwargs):
        self.active_loop_vars.add(o.variable)
        try:
            return self.visit_Node(o, **kwargs)
        finally:
            self.active_loop_vars.remove(o.variable)

    @staticmethod
    def _collect_range_arrays(expr):
        return [
            var for var in FindVariables(unique=False).visit(expr)
            if isinstance(var, sym.Array)
            and any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions)
        ]

    @staticmethod
    def _is_scalarizable_inline_call(call):
        """Only known elemental inline calls are safe to scalarize."""
        if call.function.type and call.function.type.is_intrinsic:
            return True
        if not call.function.type:
            return False

        def resolve_routine_from_scope():
            scope = getattr(call.function, 'scope', None)
            if scope is not None and hasattr(scope, 'subroutines'):
                return next(
                    (routine for routine in scope.subroutines
                     if routine.name.lower() == call.name.lower()),
                    None
                )
            return None

        procedure_dtype = getattr(call.function.type, 'dtype', None)
        if not procedure_dtype:
            routine = resolve_routine_from_scope()
            if routine is not None:
                return routine.procedure_type.is_elemental
            return False
        if hasattr(procedure_dtype, 'is_elemental'):
            if getattr(procedure_dtype, 'procedure', BasicType.DEFERRED) is BasicType.DEFERRED:
                routine = resolve_routine_from_scope()
                if routine is not None:
                    return routine.procedure_type.is_elemental
            return procedure_dtype.is_elemental
        procedure_type = call.procedure_type
        return procedure_type is not BasicType.DEFERRED and procedure_type.is_elemental

    def _find_scalarizable_rhs_arrays(self, expr):
        """Return RHS arrays that can be safely scalarized."""
        scalarizable_arrays = []
        unsafe_arrays = []

        inline_calls = set(FindInlineCalls().visit(expr))
        inline_call_arrays = set()
        for call in inline_calls:
            call_arrays = [
                var for var in FindVariables(unique=False).visit(call)
                if isinstance(var, sym.Array)
                and any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions)
            ]
            inline_call_arrays.update(call_arrays)
            if self._is_scalarizable_inline_call(call):
                scalarizable_arrays.extend(call_arrays)
            else:
                unsafe_arrays.extend(call_arrays)

        plain_arrays = [
            var for var in self._collect_range_arrays(expr)
            if var not in inline_call_arrays
        ]
        scalarizable_arrays.extend(plain_arrays)
        return scalarizable_arrays, unsafe_arrays

    @staticmethod
    def _expr_contains_vector_array(expr):
        """Return True if ``expr`` contains an array-valued expression."""
        for var in FindVariables(unique=False).visit(expr):
            if not isinstance(var, sym.Array):
                continue
            if not var.dimensions:
                return True
            if any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                return True
        return False

    @classmethod
    def _has_vector_valued_rhs_subscript(cls, expr):
        """
        Return True if an RHS array uses a vector-valued subscript expression.

        This conservatively rejects cases like ``arr(map, j)`` where ``map`` is
        itself an array-valued expression. A more ambitious implementation could
        scalarize that subscript elementwise with the resolved loop index.
        """
        for var in FindVariables(unique=False).visit(expr):
            if not isinstance(var, sym.Array) or not var.dimensions:
                continue
            if var.type.dtype is BasicType.DEFERRED:
                continue
            for dim in var.dimensions:
                if isinstance(dim, sym.RangeIndex):
                    continue
                if cls._expr_contains_vector_array(dim):
                    return True
        return False

    def _has_unsafe_inline_call(self, expr):
        """Return True if expr contains a non-scalarizable inline call with vector args."""
        for call in FindInlineCalls().visit(expr):
            if self._is_scalarizable_inline_call(call):
                continue
            if any(isinstance(arg, sym.Array) for arg in FindVariables(unique=False).visit(call)):
                return True
        for var in FindVariables(unique=False).visit(expr):
            if not isinstance(var, sym.Array):
                continue
            if var.type.dtype is not BasicType.DEFERRED:
                continue
            if not var.dimensions:
                continue
            if any(isinstance(arg, sym.Array) for arg in var.dimensions):
                return True
        return False

    @staticmethod
    def _rhs_array_dim_map(resolved_dim_indices, rhs_range_positions):
        """Map resolved LHS dimension ordinals to actual RHS range positions."""
        return {
            i: rhs_range_positions[i]
            for i in resolved_dim_indices
            if i < len(rhs_range_positions)
        }

    @staticmethod
    def _has_unresolved_vector_ranges(nodes):
        """Return True if any assignment still contains range notation."""
        for assign in FindNodes(ir.Assignment).visit(nodes):
            if (isinstance(assign.lhs, sym.Array)
                    and any(isinstance(dim, sym.RangeIndex) for dim in assign.lhs.dimensions)):
                return True
            if ResolveVectorNotationTransformer._collect_range_arrays(assign.rhs):
                return True
        return False

    def _resolve_mask_expr(self, expr):
        """Resolve vector mask expressions to scalar indices when supported."""
        if self.derive_qualified_ranges:
            expr = IterationRangeShapeMapper()(expr)

        cond_arrays = self._collect_range_arrays(expr)
        if not cond_arrays:
            return expr, {}, False

        lhs_range_positions, cond_dims_per_array, _, resolvable_dim_indices = self._get_range_resolution_info(
            cond_arrays[0].dimensions, cond_arrays
        )
        if not resolvable_dim_indices:
            return expr, {}, False

        resolved_ranges = [
            cond_dims_per_array[0][lhs_range_positions[i]] for i in resolvable_dim_indices
        ]
        cond_all_dims = cond_arrays[0].dimensions
        reserved_ivars = {
            d for i, d in enumerate(cond_all_dims)
            if i not in lhs_range_positions and isinstance(d, sym.Scalar) and d in self.loop_map.values()
        }
        # Reusing an active outer loop variable here would create a nested loop
        # that shadows the surrounding scalar context (e.g. an outer ``jk``).
        reserved_ivars.update(self.active_loop_vars)
        new_dims, index_range_map, _ = self._map_ranges_to_indices(
            resolved_ranges, self.loop_map, reserved_ivars=reserved_ivars,
            map_unknown_ranges=self.map_unknown_ranges,
            scope=self.scope, basename='i_mask'
        )

        actually_resolved = [
            (orig_i, new_dim) for orig_i, new_dim in zip(resolvable_dim_indices, new_dims)
            if not isinstance(new_dim, sym.RangeIndex)
        ]
        if not actually_resolved:
            return expr, {}, False

        resolved_dim_indices, new_dims = zip(*actually_resolved)
        subst_map = {}
        for array in cond_arrays:
            range_positions = self._find_range_positions(array.dimensions)
            array_dims = list(array.dimensions)
            for i, new_dim in enumerate(new_dims):
                array_dims[range_positions[resolved_dim_indices[i]]] = new_dim
            subst_map[array] = array.clone(dimensions=as_tuple(array_dims))

        return SubstituteExpressions(subst_map).visit(expr), index_range_map, True

    @staticmethod
    def _compute_shifted_index(loop_var, lhs_range, rhs_range):
        """
        Compute the RHS array index for a shifted range.

        When LHS has range ``a:b`` and RHS has range ``c:d``, the RHS index
        for loop variable ``i`` (iterating over ``a:b``) is: ``i - a + c``.

        Parameters
        ----------
        loop_var : expression
            The loop index variable replacing the LHS range.
        lhs_range : :any:`RangeIndex`
            The range on the LHS array.
        rhs_range : :any:`RangeIndex`
            The range on the RHS array.

        Returns
        -------
        expression
            The offset-adjusted index expression, simplified.
        """
        return simplify(sym.Sum((loop_var, sym.Product((-1, lhs_range.lower)), rhs_range.lower)))

    @staticmethod
    def _map_ranges_to_indices(dims, loop_map, map_unknown_ranges=True, basename='i',
                              scope=None, reserved_ivars=None):
        """
        Map :any:`RangeIndex` dimensions to loop index variables.

        For each ``RangeIndex`` in ``dims``, either reuse a known index
        from ``loop_map`` or create a new integer variable. Returns the
        new dimension tuple, a mapping from index variables to their
        corresponding ranges, and a set of the newly created (synthesized)
        index variables.

        Parameters
        ----------
        dims : tuple
            The dimension expressions to process.
        loop_map : dict
            Map of known ``RangeIndex`` to loop variables.
        map_unknown_ranges : bool
            Whether to create new indices for unknown ranges.
        basename : str
            Base name for newly created index variables.
        scope : :any:`Subroutine` or :any:`Module`
            Scope for newly created variables.
        reserved_ivars : set, optional
            Set of index variables already in use as non-range scalar
            subscripts in the same array reference.  If a loop variable
            from ``loop_map`` collides with this set, a fresh variable
            is synthesized to avoid aliasing.

        Returns
        -------
        tuple
            ``(new_dims, index_range_map, synthesized_ivars)`` where
            ``new_dims`` has loop variables replacing ranges,
            ``index_range_map`` maps each new variable to its original
            range, and ``synthesized_ivars`` is the set of index variables
            that were newly created (as opposed to reused from
            ``loop_map``).
        """
        if reserved_ivars is None:
            reserved_ivars = set()
        index_range_map = {}
        shape_index_map = {}
        synthesized_ivars = set()
        for i, dim in zip(count(), dims):
            if isinstance(dim, sym.RangeIndex):
                # See if index variable is known for this loop range
                if dim in loop_map:
                    ivar = loop_map[dim]
                    # Guard against arrays with duplicate range dimensions
                    # (e.g. arr(KLEVSN, KLEVSN)) where both positions map to
                    # the same loop variable.  If ivar is already in use for a
                    # different position or already present as a scalar
                    # subscript in the same array reference, or would shadow an
                    # active outer loop variable, create a fresh synthesized
                    # variable so that each dimension gets its own distinct
                    # loop index and surrounding scalar references keep their
                    # meaning.
                    if ivar in index_range_map or ivar in reserved_ivars:
                        if not map_unknown_ranges:
                            continue
                        vtype = SymbolAttributes(BasicType.INTEGER)
                        ivar = sym.Variable(name=f'{basename}_{i}', type=vtype, scope=scope)
                        synthesized_ivars.add(ivar)
                else:
                    # Skip if we're not supposed to create new indices
                    if not map_unknown_ranges or dim == sym.RangeIndex((None, None)):
                        continue
                    vtype = SymbolAttributes(BasicType.INTEGER)
                    ivar = sym.Variable(name=f'{basename}_{i}', type=vtype, scope=scope)
                    synthesized_ivars.add(ivar)
                shape_index_map[(i, dim)] = ivar
                index_range_map[ivar] = dim
        # Build new dimensions tuple, substituting mapped ranges
        new_dims = as_tuple(
            shape_index_map.get((i, d), d) for i, d in zip(count(), dims)
        )
        return new_dims, index_range_map, synthesized_ivars

    def _substitute_derived_type_bounds(self, index_range_map, synthesized_ivars):
        """
        For synthesized loop bounds that contain derived-type member references,
        substitute with existing scalar variables from the routine, or create
        new ones if none exist.

        Only applies to variables in ``synthesized_ivars`` (i.e. those whose
        loop range was created from array shape information, not from an
        explicit range in the source code).  Ranges from ``loop_map``
        (Case A — explicit source-code ranges) are left untouched.

        Parameters
        ----------
        index_range_map : dict
            Maps loop index variables to their :any:`RangeIndex` ranges.
        synthesized_ivars : set
            The subset of keys in ``index_range_map`` that were newly
            created by :meth:`_map_ranges_to_indices` (i.e. not reused
            from ``loop_map``).

        Returns
        -------
        tuple
            ``(new_index_range_map, pre_stmts, new_vars)`` where
            ``new_index_range_map`` has substituted bounds,
            ``pre_stmts`` is a (possibly empty) tuple of
            :any:`Assignment` nodes to prepend before the loop nest, and
            ``new_vars`` is a tuple of newly declared scalar variables.
        """
        new_index_range_map = {}
        pre_stmts = ()
        new_vars = ()

        for ivar, irange in index_range_map.items():
            # Case A: range came from loop_map (explicit source-code range) — skip
            if ivar not in synthesized_ivars:
                new_index_range_map[ivar] = irange
                continue

            # Find derived-type member variables in the range bounds.
            # _FindOutermostVariables returns only the outermost (longest-chain)
            # symbol for each derived-type access, so e.g. ydg%yrdimv%nflevg
            # is returned but not the intermediate ydg%yrdimv or the root ydg.
            # This avoids generating incorrect struct-to-scalar assignments.
            # Both Scalar and DeferredTypeSymbol are included: the latter appears
            # when the derived-type definition is not available during parsing.
            derived_members = [
                v for v in _FindOutermostVariables().visit(irange)
                if isinstance(v, (sym.Scalar, sym.DeferredTypeSymbol))
                and v.parent is not None
            ]

            if not derived_members:
                new_index_range_map[ivar] = irange
                continue

            # Build substitution map: derived-type member -> scalar variable
            subst_map = {}
            for member in derived_members:
                key = str(member).lower().replace(' ', '')
                if key in self._scalar_assignment_map:
                    # Existing scalar found — reuse it
                    subst_map[member] = self._scalar_assignment_map[key]
                else:
                    # No existing scalar — create one with the member's basename
                    scalar_name = member.basename
                    # Check for name collision in scope
                    if self.scope is not None:
                        existing = self.scope.variable_map.get(scalar_name.lower())
                        if existing is not None and existing != member:
                            # Name collision with a different variable — skip substitution
                            continue
                    vtype = SymbolAttributes(BasicType.INTEGER)
                    new_scalar = sym.Variable(
                        name=scalar_name, type=vtype, scope=self.scope
                    )
                    subst_map[member] = new_scalar
                    # Record new variable and assignment for insertion
                    new_vars += (new_scalar,)
                    pre_stmts += (ir.Assignment(lhs=new_scalar, rhs=member),)
                    # Also register in the map so subsequent dimensions reuse it
                    self._scalar_assignment_map[key] = new_scalar

            if subst_map:
                new_irange = SubstituteExpressions(subst_map).visit(irange)
                new_index_range_map[ivar] = new_irange
            else:
                new_index_range_map[ivar] = irange

        return new_index_range_map, pre_stmts, new_vars

    def _resolve_literal_list(self, stmt):
        """
        Resolve an assignment whose RHS is a literal list by *unrolling* it
        into one scalar assignment per element.

        For example ``A(1:3) = (/ 1.0, 2.0, 3.0 /)`` becomes::

            A(1) = 1.0
            A(2) = 2.0
            A(3) = 3.0

        Only the simple case of a pure literal-list RHS assigned to an LHS
        array with a single, explicitly bounded, range dimension is handled.
        Mixed RHS expressions and bare ``:`` ranges are not supported.

        Returns the unrolled scalar assignments, or ``None`` if the assignment
        cannot be unrolled (in which case a warning describing the failure mode
        has already been emitted).
        """
        scope_str = f' in routine "{self.scope.name}"' if self.scope is not None else ''

        # Only handle a pure literal-list RHS (no mixed expressions).
        if not isinstance(stmt.rhs, sym.LiteralList):
            warning(
                f'[ResolveVectorNotationTransformer] Mixed literal-list RHS of '
                f'"{stmt}"{scope_str} prevents vector notation resolution. '
            )
            return None

        # Only unroll when every element is a scalar variable or a literal,
        # so that each element is rank-0 and the RHS element count trivially
        # matches the LHS range extent without further shape introspection.
        if not all(isinstance(el, (sym.Scalar, sym._Literal)) for el in stmt.rhs.elements):
            warning(
                f'[ResolveVectorNotationTransformer] Literal-list RHS of '
                f'"{stmt}"{scope_str} has elements that are not scalar '
                f'variables or literals; literal-list unrolling not supported. '
            )
            return None

        lhs_array = stmt.lhs
        # A LiteralList RHS is always a rank-1 initialiser (multi-dimensional
        # initialisers require RESHAPE, which Loki treats as an InlineCall, not
        # a LiteralList), so the LHS has exactly one range dimension (though not
        # necessarily the leading one, e.g. foo(j, 1:3)).
        range_pos = self._find_range_positions(lhs_array.dimensions)[0]
        lhs_range = lhs_array.dimensions[range_pos]

        lower = lhs_range.lower
        step = lhs_range.step
        # A bare ``:`` (unknown lower bound) cannot be turned into concrete
        # element indices.
        if lower is None:
            warning(
                f'[ResolveVectorNotationTransformer] Unqualified ":" on LHS of '
                f'"{stmt}"{scope_str} prevents literal-list unrolling (shape unknown). '
            )
            return None
        step_expr = step if step else sym.IntLiteral(1)

        # Emit one scalar assignment per element, with LHS index ``lower + k*step``.
        assignments = []
        for k, element in enumerate(stmt.rhs.elements):
            index = simplify(sym.Sum((lower, sym.Product((sym.IntLiteral(k), step_expr)))))
            new_dims = list(lhs_array.dimensions)
            new_dims[range_pos] = index
            new_lhs = lhs_array.clone(dimensions=as_tuple(new_dims))
            assignments.append(ir.Assignment(lhs=new_lhs, rhs=element))

        return as_tuple(assignments)

    def visit_Assignment(self, stmt, **kwargs):  # pylint: disable=unused-argument

        # --- Step 1: Early exits ---

        # Pointer assignment
        if stmt.ptr:
            return stmt

        # LHS is not an array
        if not isinstance(stmt.lhs, sym.Array):
            return stmt

        create_loops = kwargs.get('create_loops', True)

        # Forbidden intrinsic calls in the RHS
        inline_calls = [(_.name).lower() for _ in FindInlineCalls().visit(stmt.rhs)]
        forbidden_ops = ['present', 'sum']
        if any(op in inline_calls for op in forbidden_ops):
            return stmt
        if HAVE_FP:
            if any(redux_op in FindExpressions().visit(stmt.rhs)
                   for redux_op in Fortran2003.Intrinsic_Name.array_reduction_names):
                return stmt
        if self._has_unsafe_inline_call(stmt.rhs):
            return stmt
        if self._has_vector_valued_rhs_subscript(stmt.rhs):
            return stmt

        # --- Step 2: Record original range usage before shape inference ---
        orig_lhs_dims = stmt.lhs.dimensions
        orig_rhs_arrays, orig_unsafe_rhs_arrays = self._find_scalarizable_rhs_arrays(stmt.rhs)
        if orig_unsafe_rhs_arrays:
            return stmt

        # --- Step 3: Derive qualified ranges from shapes ---
        if self.derive_qualified_ranges:
            shape_mapper = IterationRangeShapeMapper()
            stmt._update(lhs=shape_mapper(stmt.lhs), rhs=shape_mapper(stmt.rhs))

        # --- Resolve literal-list RHS by unrolling ---
        # This runs after Step 3 so the LHS range has explicit bounds.
        # Falls back to the previous warn-and-bail behaviour when unrolling is
        # not possible (mixed RHS, unknown ranges).
        if FindLiteralLists().visit(stmt.rhs):
            resolved = self._resolve_literal_list(stmt)
            if resolved:
                return resolved
            return stmt

        # --- Step 4: Identify range-indexed dimensions ---

        # RHS arrays that have at least one RangeIndex dimension
        rhs_arrays, unsafe_rhs_arrays = self._find_scalarizable_rhs_arrays(stmt.rhs)
        if unsafe_rhs_arrays:
            return stmt

        # LHS array dimensions
        lhs_array = stmt.lhs
        lhs_dims = lhs_array.dimensions
        lhs_range_positions, rhs_dims_per_array, rhs_range_positions_per_array, resolvable_dim_indices = (
            self._get_range_resolution_info(
                lhs_dims, rhs_arrays,
                qualification_lhs_dims=orig_lhs_dims,
                qualification_rhs_arrays=orig_rhs_arrays,
            )
        )

        # --- Step 5: Filter to resolvable dimensions ---
        # Nothing to resolve
        if not resolvable_dim_indices:
            if lhs_range_positions:
                scope_str = f' in routine "{self.scope.name}"' if self.scope is not None else ''
                warning(
                    f'[ResolveVectorNotationTransformer] Unqualified ":" on LHS of '
                    f'"{stmt}"{scope_str} could not be resolved (shape unknown). '
                )
            return stmt

        # --- Step 6: Map LHS ranges to loop index variables ---
        resolved_lhs_ranges = [
            lhs_dims[lhs_range_positions[i]] for i in resolvable_dim_indices
        ]
        # Collect loop variables already present as fixed scalar subscripts so
        # that _map_ranges_to_indices does not alias them with a resolved range.
        reserved_ivars = {
            d for i, d in enumerate(lhs_dims)
            if i not in lhs_range_positions and isinstance(d, sym.Scalar) and d in self.loop_map.values()
        }
        # Keep existing scalar subscripts and active surrounding loop indices
        # distinct from any new loop variable chosen for resolved ranges.
        reserved_ivars.update(self.active_loop_vars)
        new_lhs_dims, index_range_map, synthesized_ivars = self._map_ranges_to_indices(
            resolved_lhs_ranges, self.loop_map, reserved_ivars=reserved_ivars,
            map_unknown_ranges=self.map_unknown_ranges,
            scope=self.scope, basename=f'i_{stmt.lhs.basename}'
        )

        # Filter out dimensions that were not actually resolved to a scalar loop
        # variable (i.e. new_lhs_dim is still a RangeIndex).  This can happen
        # when map_unknown_ranges=False and the LHS range is not in loop_map.
        # Keeping such dims would corrupt RHS expressions by feeding a RangeIndex
        # into _compute_shifted_index, producing e.g. ``-1 + (1:klevsn)``.
        actually_resolved = [
            (orig_i, lhs_rng, new_dim)
            for orig_i, lhs_rng, new_dim
            in zip(resolvable_dim_indices, resolved_lhs_ranges, new_lhs_dims)
            if not isinstance(new_dim, sym.RangeIndex)
        ]
        if not actually_resolved:
            return stmt
        resolved_dim_indices, resolved_lhs_ranges, new_lhs_dims = zip(*actually_resolved)

        # --- Step 7: Compute RHS index expressions (with offset) ---
        new_rhs_dims_per_array = []
        rhs_dim_maps = [
            self._rhs_array_dim_map(resolved_dim_indices, rhs_pos)
            for rhs_pos in rhs_range_positions_per_array
        ]
        for array_dims, rhs_dim_map in zip(rhs_dims_per_array, rhs_dim_maps):
            new_rhs_dims = {}
            for i, (lhs_range, new_lhs_dim) in enumerate(zip(resolved_lhs_ranges, new_lhs_dims)):
                resolved_dim_index = resolved_dim_indices[i]
                rhs_pos = rhs_dim_map.get(resolved_dim_index)
                if rhs_pos is None:
                    continue
                rhs_range = array_dims[rhs_pos]
                is_aligned_dim = (
                    lhs_range == rhs_range or rhs_range == sym.RangeIndex((None, None))
                ) or (
                    isinstance(lhs_range, sym.RangeIndex) and isinstance(rhs_range, sym.RangeIndex) and
                    lhs_range.lower == rhs_range.lower
                )
                if is_aligned_dim:
                    new_rhs_dims[resolved_dim_index] = new_lhs_dim
                else:
                    new_rhs_dims[resolved_dim_index] = (
                        self._compute_shifted_index(new_lhs_dim, lhs_range, rhs_range)
                    )
            new_rhs_dims_per_array.append(new_rhs_dims)

        # --- Step 8: Build new array expressions ---

        # New LHS array with loop indices replacing ranges
        new_lhs_arr_dims = list(lhs_dims)
        for i, d in enumerate(new_lhs_dims):
            new_lhs_arr_dims[lhs_range_positions[resolved_dim_indices[i]]] = d
        new_lhs_arr = lhs_array.clone(dimensions=as_tuple(new_lhs_arr_dims))

        # New RHS arrays with loop indices replacing ranges
        new_rhs_array_list = []
        for i_arr, _array in enumerate(rhs_arrays):
            new_arr_dims = list(rhs_dims_per_array[i_arr])
            rhs_dim_map = rhs_dim_maps[i_arr]
            for resolved_dim_index, d in new_rhs_dims_per_array[i_arr].items():
                rhs_pos = rhs_dim_map.get(resolved_dim_index)
                if rhs_pos is None:
                    continue
                new_arr_dims[rhs_pos] = d
            new_rhs_array_list.append(_array.clone(dimensions=as_tuple(new_arr_dims)))

        # Update the statement in-place
        rhs_substitution = dict(zip(rhs_arrays, new_rhs_array_list))
        stmt._update(
            lhs=new_lhs_arr,
            rhs=SubstituteExpressions(rhs_substitution).visit(stmt.rhs)
        )

        # Record all newly created loop index variables for declaration
        self.index_vars.update(list(index_range_map.keys()))

        # --- Step 9: Substitute derived-type members in synthesized bounds ---
        # For bounds that were derived from array shapes (not from explicit
        # source-code ranges), replace any derived-type member references
        # (e.g., KDIM%KLEVS) with existing or new plain scalar variables
        # (e.g., KLEVS) so that generated loops are device-safe.
        # Only performed when substitute_derived_type_bounds_flag is True
        # (i.e. for driver routines); kernels leave derived-type bounds as-is.
        # New scalar extraction assignments are accumulated in pre_body_stmts
        # and prepended to the routine body (not inline before the loop) so
        # they land before any OpenACC data regions.
        if self.substitute_derived_type_bounds_flag:
            index_range_map, new_pre_stmts, new_vars = self._substitute_derived_type_bounds(
                index_range_map, synthesized_ivars
            )
            if new_pre_stmts:
                self.pre_body_stmts.extend(new_pre_stmts)
            if new_vars:
                self.index_vars.update(new_vars)

        # --- Step 10: Wrap in loop nest ---
        if create_loops and len(index_range_map):
            return self._build_loop_nest(index_range_map, stmt, insert_comments=self.insert_comments)

        # No vector dimensions encountered, return unchanged
        return stmt

    def visit_Conditional(self, o, **kwargs):
        """
        Visit children (which may resolve vector notation in the body),
        then demote ``inline=True`` conditionals whose body is no longer
        compatible with single-line ``IF`` formatting.

        This happens when ``visit_Assignment`` wraps a body statement in
        a loop nest (plus a comment), expanding the body beyond a single
        simple statement.
        """
        visited = self.visit_Node(o, **kwargs)

        if isinstance(visited, ir.Conditional) and visited.inline:
            body = visited.body
            is_one_liner = (
                len(body) == 1
                and not isinstance(body[0], (ir.Loop, ir.Comment, tuple))
            )
            if not is_one_liner:
                visited = visited.clone(inline=False)

        return visited

    def visit_MaskedStatement(self, masked, **kwargs):  # pylint: disable=unused-argument
        if len(masked.conditions) != 1 or len(masked.bodies) != 1:
            return masked

        condition, index_range_map, resolved = self._resolve_mask_expr(masked.conditions[0])
        if not resolved:
            return masked

        # Reuse the mask loop indices in each body assignment so the condition
        # and body refer to the same array element.
        mask_loop_map = {irange: ivar for ivar, irange in index_range_map.items()}
        with dict_override(self.loop_map, mask_loop_map):
            with dict_override(kwargs, {'create_loops': False}):
                body = self.visit(masked.bodies[0], **kwargs)
                else_body = self.visit(masked.default, **kwargs)

        body_nodes = as_tuple(body)
        else_nodes = as_tuple(else_body)

        if (FindNodes(ir.MaskedStatement).visit(body_nodes)
                or FindNodes(ir.MaskedStatement).visit(else_nodes)
                or self._has_unresolved_vector_ranges(body_nodes)
                or self._has_unresolved_vector_ranges(else_nodes)):
            return masked

        # Record all newly created loop index variables for declaration
        self.index_vars.update(list(index_range_map.keys()))

        cond = ir.Conditional(condition=condition, body=body_nodes, else_body=else_nodes)
        return self._build_loop_nest(index_range_map, cond, insert_comments=self.insert_comments)
