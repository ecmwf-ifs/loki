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
                            substitute_derived_type_bounds=False):
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
                             substitute_derived_type_bounds=False):
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
    """

    def __init__(
            self, *args, loop_map=None, scope=None,
            derive_qualified_ranges=True, map_unknown_ranges=True,
            resolve_implicit_rhs_ranges=True,
            substitute_derived_type_bounds=False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.scope = scope
        self.loop_map = {} if loop_map is None else loop_map
        self.index_vars = OrderedSet()
        self.pre_body_stmts = []

        self.map_unknown_ranges = map_unknown_ranges
        self.derive_qualified_ranges = derive_qualified_ranges
        self.resolve_implicit_rhs_ranges = resolve_implicit_rhs_ranges
        self.substitute_derived_type_bounds_flag = substitute_derived_type_bounds
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

    @staticmethod
    def _find_qualified_range_positions(dims, range_positions):
        """
        Return ordinal indices into ``range_positions`` whose corresponding
        dimension is *not* a bare ``(:)`` (i.e. ``RangeIndex((None, None))``).
        """
        return [
            i for i, j in enumerate(range_positions)
            if dims[j] != sym.RangeIndex((None, None))
        ]

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
    def _map_ranges_to_indices(dims, loop_map, map_unknown_ranges=True, basename='i', scope=None):
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
                    # different position, create a fresh synthesized variable so
                    # that each dimension gets its own distinct loop index.
                    if ivar in index_range_map:
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

    def visit_Assignment(self, stmt, **kwargs):  # pylint: disable=unused-argument

        # --- Step 1: Early exits ---

        # Pointer assignment
        if stmt.ptr:
            return stmt

        # LHS is not an array
        if not isinstance(stmt.lhs, sym.Array):
            return stmt

        # RHS contains a literal list (e.g., (/ 1.0, 2.0 /))
        if FindLiteralLists().visit(stmt.rhs):
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

        # --- Step 2: Derive qualified ranges from shapes ---
        if self.derive_qualified_ranges:
            shape_mapper = IterationRangeShapeMapper()
            stmt._update(lhs=shape_mapper(stmt.lhs), rhs=shape_mapper(stmt.rhs))

        # --- Step 3: Identify range-indexed dimensions ---

        # RHS arrays that have at least one RangeIndex dimension
        rhs_vars = FindVariables(unique=False).visit(stmt.rhs)
        rhs_arrays = [
            var for var in rhs_vars
            if isinstance(var, sym.Array)
            and any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions)
        ]
        rhs_dims_per_array = [array.dimensions for array in rhs_arrays]
        rhs_range_positions_per_array = [
            self._find_range_positions(dims) for dims in rhs_dims_per_array
        ]

        # LHS array dimensions
        lhs_array = stmt.lhs
        lhs_dims = lhs_array.dimensions
        lhs_range_positions = self._find_range_positions(lhs_dims)
        lhs_qualified_positions = self._find_qualified_range_positions(
            lhs_dims, lhs_range_positions
        )

        # --- Step 4: Filter to resolvable dimensions ---
        if self.resolve_implicit_rhs_ranges:
            resolvable_dim_indices = lhs_qualified_positions
        else:
            rhs_qualified_positions_per_array = [
                self._find_qualified_range_positions(rhs_dims, rhs_pos)
                for rhs_dims, rhs_pos in zip(rhs_dims_per_array, rhs_range_positions_per_array)
            ]
            resolvable_dim_indices = [
                j for j in lhs_qualified_positions
                if all(
                    j in rhs_qualified
                    for rhs_qualified in rhs_qualified_positions_per_array
                )
            ]

        # Nothing to resolve
        if not resolvable_dim_indices:
            return stmt

        # --- Step 5: Map LHS ranges to loop index variables ---
        resolved_lhs_ranges = [
            lhs_dims[lhs_range_positions[i]] for i in resolvable_dim_indices
        ]
        new_lhs_dims, index_range_map, synthesized_ivars = self._map_ranges_to_indices(
            resolved_lhs_ranges, self.loop_map,
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

        # --- Step 6: Compute RHS index expressions (with offset) ---
        resolved_rhs_ranges_per_array = [
            [array_dims[rhs_pos[i]] for i in resolved_dim_indices]
            for array_dims, rhs_pos in zip(rhs_dims_per_array, rhs_range_positions_per_array)
        ]
        new_rhs_dims_per_array = []
        for array, resolved_rhs_ranges in zip(rhs_arrays, resolved_rhs_ranges_per_array):
            new_rhs_dims = []
            for lhs_range, new_lhs_dim, rhs_range in zip(
                resolved_lhs_ranges, new_lhs_dims, resolved_rhs_ranges
            ):
                is_aligned_dim = (
                    lhs_range == rhs_range or rhs_range == sym.RangeIndex((None, None))
                ) or (
                    isinstance(lhs_range, sym.RangeIndex) and isinstance(rhs_range, sym.RangeIndex) and
                    lhs_range.lower == rhs_range.lower
                )
                if is_aligned_dim:
                    new_rhs_dims.append(new_lhs_dim)
                else:
                    new_rhs_dims.append(
                        self._compute_shifted_index(new_lhs_dim, lhs_range, rhs_range)
                    )
            new_rhs_dims_per_array.append(new_rhs_dims)

        # --- Step 7: Build new array expressions ---

        # New LHS array with loop indices replacing ranges
        new_lhs_arr_dims = list(lhs_dims)
        for i, d in enumerate(new_lhs_dims):
            new_lhs_arr_dims[lhs_range_positions[resolved_dim_indices[i]]] = d
        new_lhs_arr = lhs_array.clone(dimensions=as_tuple(new_lhs_arr_dims))

        # New RHS arrays with loop indices replacing ranges
        new_rhs_array_list = []
        for i_arr, _array in enumerate(rhs_arrays):
            new_arr_dims = list(rhs_dims_per_array[i_arr])
            for i, d in enumerate(new_rhs_dims_per_array[i_arr]):
                new_arr_dims[rhs_range_positions_per_array[i_arr][resolved_dim_indices[i]]] = d
            new_rhs_array_list.append(_array.clone(dimensions=as_tuple(new_arr_dims)))

        # Update the statement in-place
        rhs_substitution = dict(zip(rhs_arrays, new_rhs_array_list))
        stmt._update(
            lhs=new_lhs_arr,
            rhs=SubstituteExpressions(rhs_substitution).visit(stmt.rhs)
        )

        # Record all newly created loop index variables for declaration
        self.index_vars.update(list(index_range_map.keys()))

        # --- Step 8: Substitute derived-type members in synthesized bounds ---
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

        # --- Step 9: Wrap in loop nest ---
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

            return (ir.Comment('! loki resolved vector notation'),) + (loop,)

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
