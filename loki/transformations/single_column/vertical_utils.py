# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

"""
Shared utility functions for vertical loop transformations.

This module provides the building blocks used by
:class:`~loki.transformations.single_column.vertical_kcaching.SCCVerticalKCaching`
to merge vertical loops, convert array dependencies to scalar carry
variables, and demote KLEV-dimensioned temporaries.

The utilities are grouped as follows:

**Loop inspection**
    :func:`_collect_vertical_loops`, :func:`_loop_upper_bound_expr`,
    :func:`_loop_upper_bound_str`, :func:`_loop_is_backward`,
    :func:`_loop_effective_bounds`, :func:`_is_klev_plus_n`,
    :func:`_extract_plus_n`.

**Carry conversion** (Phase 1c)
    :func:`_convert_all_carries`, :func:`_build_carry_expr_entries`,
    :func:`_build_save_assignment`.

**Init-expression substitution** (Phase 1d)
    :func:`_substitute_init_expressions_all_loops`,
    :func:`_build_init_subst_map`, :func:`_apply_subst_outside_loops`.

**Dead-loop / zero-init elimination** (Phases 1b, 1e)
    :func:`_find_dead_loops_all`, :func:`_remove_whole_array_zero_inits`,
    :func:`_find_zero_inits_outside_loops`.

**Loop merging** (Phase 2)
    :func:`_merge_vertical_loops`, :func:`_build_bounds_guard`,
    :func:`_relocate_interloop_code`.

**Post-merge fixup** (Phases 2b, 2c)
    :func:`_cross_loop_carry_substitution`,
    :func:`_insert_writebacks_for_argument_carries`,
    :func:`_hoist_rotates_to_end`.

**Demotion helpers** (Phase 3)
    :func:`_find_demotable_arrays`, :func:`_collect_call_arg_names`,
    :func:`_collect_refs_outside_loops`.

**Cleanup** (Phase 4)
    :func:`_remove_self_assignments`, :func:`_remove_dead_carry_originals`.

**Shared infrastructure**
    :class:`_SkipNodesVisitor`, :class:`_EarlyTermination`,
    :func:`_make_zero_literal`, :func:`_is_zero_literal`,
    :func:`_is_jk_eq_1`, :func:`_collect_loop_node_set`.
"""

from collections import defaultdict, OrderedDict

from loki.expression import symbols as sym
from loki.ir import (
    nodes as ir, FindNodes, Visitor, Transformer, FindVariables,
    SubstituteExpressions, SubstituteExpressionsSkipLHS
)
from loki.logging import info, warning
from loki.tools import as_tuple, CaseInsensitiveDict
from loki.types import BasicType
from loki.expression.symbolic import simplify
from loki.analyse.analyse_dataflow import (
    classify_array_access_offsets, array_loop_carried_dependencies,
    extract_offset,
)


__all__ = []


class _EarlyTermination(Exception):
    """Raised to short-circuit a Visitor walk."""


class _SkipNodesVisitor(Visitor):
    """
    Base :any:`Visitor` that skips specific node instances and recurses
    into transparent container nodes (:any:`Section`, :any:`Associate`,
    :any:`Conditional`, :any:`Loop`).

    Subclasses override ``visit_Node`` (or type-specific ``visit_*``
    methods) to process leaf / non-container nodes.  The default
    ``visit_Node`` does nothing.
    """

    def __init__(self, skip_nodes=None):
        super().__init__()
        self.skip_nodes = skip_nodes or set()

    # -- tuple / list: filter out skipped nodes before dispatching ------

    def visit_tuple(self, o, **kwargs):
        for c in o:
            if c not in self.skip_nodes:
                self.visit(c, **kwargs)

    visit_list = visit_tuple

    # -- expressions / unknown objects: nothing to do -------------------

    def visit_object(self, o, **kwargs):
        pass

    # -- default leaf: subclasses override this -------------------------

    def visit_Node(self, o, **kwargs):
        pass

    # -- transparent containers: recurse --------------------------------

    def visit_Section(self, o, **kwargs):
        self.visit(o.body, **kwargs)

    def visit_Associate(self, o, **kwargs):
        self.visit(o.body, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        self.visit(o.body, **kwargs)
        if o.else_body:
            self.visit(o.else_body, **kwargs)

    def visit_Loop(self, o, **kwargs):
        self.visit(o.body, **kwargs)


def _collect_call_arg_names(routine):
    """
    Return a set of lowercase variable names that appear as arguments
    in any :any:`CallStatement` in *routine*.
    """
    names = set()
    for call in FindNodes(ir.CallStatement).visit(routine.body):
        for arg in call.arguments:
            for v in FindVariables().visit(arg):
                names.add(v.name.lower())
        if call.kwarguments:
            for _kw, arg in call.kwarguments:
                for v in FindVariables().visit(arg):
                    names.add(v.name.lower())
    return names


def _make_zero_literal(orig_type):
    """
    Return a type-appropriate zero literal for carry variable initialisation.

    Returns :any:`IntLiteral` for INTEGER arrays, :any:`LogicLiteral`
    for LOGICAL arrays, and :any:`FloatLiteral` (with the original kind)
    for REAL and all other types.

    When the kind is an :any:`InlineCall` (e.g. ``selected_real_kind(13, 300)``
    as produced by the OMNI frontend), it is omitted from the literal because
    Fortran requires kind parameters in literals to be named constants or
    integer literals.  A plain ``0.0`` is safe here since Fortran performs
    implicit type conversion in assignment context.
    """
    dtype = getattr(orig_type, 'dtype', None)
    if dtype == BasicType.INTEGER:
        return sym.IntLiteral(0)
    if dtype == BasicType.LOGICAL:
        return sym.LogicLiteral('.FALSE.')
    # Default: REAL (or COMPLEX, DEFERRED, etc.)
    kind = orig_type.kind
    if isinstance(kind, sym.InlineCall):
        return sym.FloatLiteral(0.0)
    return sym.FloatLiteral(0.0, kind=kind)


def _loop_is_backward(loop):
    """Return ``True`` if *loop* has a negative step."""
    bounds = loop.bounds
    if bounds is None or bounds.children is None:
        return False
    if len(bounds.children) > 2 and bounds.children[2] is not None:
        # str() is intentional: step may be a symbolic negation (e.g. Product((-1, x)))
        # whose rendered form starts with '-'; no reliable node comparison exists.
        step_str = str(bounds.children[2]).strip()
        return step_str.startswith('-') or step_str.startswith('-(')
    return False


def _loop_effective_bounds(loop):
    """
    Return ``(lower, upper)`` expressions for *loop*, accounting for
    backward (negative-step) loops where children[0] > children[1].
    """
    bounds = loop.bounds
    if bounds is None or bounds.children is None:
        return None, None
    lower, upper = bounds.children[0], bounds.children[1]
    if _loop_is_backward(loop):
        lower, upper = upper, lower
    return lower, upper


def _loop_upper_bound_expr(loop):
    """
    Return the effective upper-bound expression of *loop*'s range, or
    ``None`` if bounds are not available.  For backward loops the start
    expression (children[0]) is returned since it is the larger value.
    """
    _lower, upper = _loop_effective_bounds(loop)
    return upper


def _loop_upper_bound_str(loop):
    """
    Return the upper-bound of *loop*'s range as a normalised
    lowercase string, e.g. ``'klev'``, ``'klev + 1'``.
    """
    upper = _loop_upper_bound_expr(loop)
    if upper is None:
        return ''
    return str(upper).strip().lower()


def _is_klev_plus_n(upper_expr, vertical_size):
    """
    Return ``True`` if *upper_expr* represents the vertical size plus a
    positive integer, e.g. ``KLEV + 1``.

    Parameters
    ----------
    upper_expr : expression or str
        The upper-bound expression (a Loki expression node) or, for
        backward compatibility, a normalised lowercase string.
    vertical_size : str or expression
        The vertical size variable name (e.g. ``'klev'``) or a Loki
        expression node.
    """
    # --- Expression-level path ---
    if not isinstance(upper_expr, str) and upper_expr is not None:
        vs_name = (vertical_size.lower() if isinstance(vertical_size, str)
                   else str(vertical_size).strip().lower())
        # Build a symbolic representation of the vertical size for subtraction
        vs_sym = sym.Variable(name=vs_name)
        try:
            diff = simplify(upper_expr - vs_sym)
        except (TypeError, AttributeError):
            diff = None
        if isinstance(diff, sym.IntLiteral):
            return diff.value >= 1
        if isinstance(diff, int):
            return diff >= 1
        # Expression-level check didn't resolve; fall through to string path

    # --- String fallback ---
    upper_str = (str(upper_expr).strip().lower()
                 if not isinstance(upper_expr, str) else upper_expr)
    vs = (vertical_size.lower() if isinstance(vertical_size, str)
          else str(vertical_size).strip().lower())
    us = upper_str.replace(' ', '')
    if us.startswith(vs + '+'):
        suffix = us[len(vs) + 1:]
        try:
            n = int(suffix)
            return n >= 1
        except ValueError:
            return False
    return False


def _collect_vertical_loops(body, vertical_index):
    """
    Walk *body* (a :any:`Section`, :any:`Associate`, or tuple of nodes)
    and return all :any:`Loop` nodes whose variable is *vertical_index*,
    preserving source order.

    The walk recurses into transparent container nodes (:any:`Section`,
    :any:`Associate`) but NOT into loop or conditional bodies — only
    the outermost JK loop at each position is returned.

    Loops nested inside ``Conditional`` wrappers are returned as
    ``(loop, conditional_wrapper)`` pairs; top-level loops are returned
    as ``(loop, None)`` pairs.

    .. note::

       This function matches the induction variable by name only and
       does not consult ``Dimension.index_aliases``.  If a routine uses
       an alias (e.g. ``NLEV``) instead of the primary index name, those
       loops will not be collected.
    """

    class _VLoopCollector(Visitor):
        def __init__(self):
            super().__init__()
            self.vertical_index_lower = vertical_index.lower()
            self.result = []

        def visit_tuple(self, o, **kwargs):
            for c in o:
                self.visit(c, **kwargs)

        visit_list = visit_tuple

        def visit_object(self, o, **kwargs):
            pass

        def visit_Node(self, o, **kwargs):
            pass  # don't recurse into unrecognised node types

        def visit_Loop(self, o, **kwargs):
            if o.variable.name.lower() == self.vertical_index_lower:
                self.result.append((o, kwargs.get('wrapper')))
            # don't recurse into the loop body

        def visit_Conditional(self, o, **kwargs):  # pylint: disable=unused-argument
            # Only check for JK loops as direct children of conditional
            # branches (matching original non-recursive behavior).
            for branch_nodes in (o.body, o.else_body or ()):
                for child in branch_nodes:
                    if isinstance(child, ir.Loop):
                        if child.variable.name.lower() == self.vertical_index_lower:
                            self.result.append((child, o))

        def visit_Section(self, o, **kwargs):
            self.visit(o.body, **kwargs)

        def visit_Associate(self, o, **kwargs):
            self.visit(o.body, **kwargs)

        def visit_Comment(self, o, **kwargs):
            pass

    collector = _VLoopCollector()
    collector.visit(body)
    return collector.result


def _collect_loop_node_set(all_vloops):
    """
    Build a set of IR nodes (loops and their conditional wrappers) from the
    list returned by :func:`_collect_vertical_loops`.  This set is used by
    several helpers to skip over vertical-loop bodies when walking the IR.
    """
    loop_nodes = set()
    for loop, cond_wrapper in all_vloops:
        loop_nodes.add(loop)
        if cond_wrapper is not None:
            loop_nodes.add(cond_wrapper)
    return loop_nodes


def _is_jk_eq_1(expr, loop_var_name):
    """
    Return True if *expr* represents ``JK == 1`` or ``1 == JK``.

    Uses expression-level comparison via Loki's ``StrCompareMixin``
    (case-insensitive ``==``) and ``IntLiteral`` numeric equality.
    """
    if isinstance(expr, sym.Comparison) and expr.operator == '==':
        left, right = expr.left, expr.right
        # JK == 1
        if (left == loop_var_name and
                isinstance(right, sym.IntLiteral) and right.value == 1):
            return True
        # 1 == JK
        if (right == loop_var_name and
                isinstance(left, sym.IntLiteral) and left.value == 1):
            return True
    return False


def _mark_jk1_conditionals(loop, arr_name, cond_to_remove):
    """
    For Pattern A carries, find ``IF (JK == 1)`` conditionals in the loop
    body that initialise the carry array and mark them for removal.

    The typical pattern is::

        IF (JK==1) THEN
          X(JL, JK) = 0.0
          Y(JL, JK) = 0.0
        ELSE
          X(JL, JK) = X(JL, JK-1)
          Y(JL, JK) = Y(JL, JK-1)
        END IF

    The entire conditional is replaced by load-from-carry assignments::

        X(JL, JK) = z_x_vc(JL)
        Y(JL, JK) = z_y_vc(JL)

    which are already handled by the expression map for the ELSE branch.
    The IF branch (init to 0) is now unnecessary because the carry
    variable is initialised to 0 before the loop.
    """
    loop_var = loop.variable
    loop_var_name = loop_var.name.lower()

    for node in loop.body if isinstance(loop.body, tuple) else (loop.body,):
        # Walk into inner loops (e.g. DO JL) to find conditionals
        for cond in FindNodes(ir.Conditional).visit(node):
            # Check if condition is JK == 1 (or 1 == JK)
            condition = cond.condition
            if not _is_jk_eq_1(condition, loop_var_name):
                continue

            # Check if the ELSE branch contains assignments to our carry array
            # at JK with RHS being ARRAY(JK-1)
            else_body = cond.else_body or ()
            has_array = False
            for stmt in FindNodes(ir.Assignment).visit(else_body):
                if isinstance(stmt.lhs, sym.Array) and stmt.lhs.name.lower() == arr_name.lower():
                    has_array = True
                    break

            if has_array:
                # The ELSE branch assignments like X(JK) = X(JK-1) are already
                # handled by the expr_map (X(JK-1) → carry). But we need to
                # replace the entire IF/ELSE block with just the ELSE branch
                # assignments (after substitution), because the IF branch
                # (X(JK) = 0) is no longer needed (carry is pre-initialised).
                #
                # We'll mark the conditional for replacement with the ELSE
                # branch body (which will be substituted by the expr_map
                # pass that runs first).
                replacement = list(else_body)
                cond_to_remove.append((cond, replacement))
                break  # Only one JK==1 conditional per carry array


def _any_read_outside_node(body, exclude_node, written_names):
    """
    Return ``True`` if any variable in *written_names* is **read**
    (not just written) in any node of *body* other than *exclude_node*.

    Recurses into transparent container nodes (:any:`Section`,
    :any:`Associate`) so that the exclude works correctly when loops
    are nested inside such containers.

    A variable is considered "read" if it appears on the RHS of an
    assignment, in a conditional expression, in a call argument, or
    in any context other than the bare LHS of an assignment.
    """

    class _ReadChecker(_SkipNodesVisitor):
        def __init__(self, exclude_node, written_names):
            skip = {exclude_node} if exclude_node else set()
            super().__init__(skip)
            self.written_names = written_names

        def _check(self, vars_iter):
            for var in vars_iter:
                if var.name.lower() in self.written_names:
                    raise _EarlyTermination()

        def visit_Assignment(self, o, **kwargs):  # pylint: disable=unused-argument
            # RHS reads
            self._check(FindVariables().visit(o.rhs))
            # LHS subscript reads (e.g. A(ZTRPAUS) would be a read)
            lhs = o.lhs
            if hasattr(lhs, 'dimensions') and lhs.dimensions:
                for dim in lhs.dimensions:
                    self._check(FindVariables().visit(dim))

        def visit_Conditional(self, o, **kwargs):
            self._check(FindVariables().visit(o.condition))
            self.visit(o.body, **kwargs)
            if o.else_body:
                self.visit(o.else_body, **kwargs)

        def visit_Loop(self, o, **kwargs):
            self._check(FindVariables().visit(o.bounds))
            self.visit(o.body, **kwargs)

        def visit_Node(self, o, **kwargs):
            self._check(FindVariables().visit(o))

    try:
        _ReadChecker(exclude_node, written_names).visit(body)
        return False
    except _EarlyTermination:
        return True


def _substitute_jk_in_expr(expr, jk_var, replacement):
    """
    Replace all occurrences of *jk_var* in *expr* with *replacement*.

    Handles both scalar references (``JK``) and array subscripts
    containing the loop variable (``A(JL, JK)`` → ``A(JL, NJKT3)``).
    """
    jk_lower = jk_var.name.lower()
    vmap = {}
    all_vars = FindVariables().visit(expr)
    for var in all_vars:
        if isinstance(var, sym.Array) and var.dimensions:
            new_dims = tuple(
                replacement
                if (isinstance(d, (sym.Scalar, sym.DeferredTypeSymbol))
                    and d.name.lower() == jk_lower)
                else d
                for d in var.dimensions
            )
            if new_dims != var.dimensions:
                vmap[var] = var.clone(dimensions=new_dims)
        elif isinstance(var, (sym.Scalar, sym.DeferredTypeSymbol)):
            if var.name.lower() == jk_lower:
                vmap[var] = replacement

    if vmap:
        return SubstituteExpressions(vmap).visit(expr)
    return expr


def _build_init_subst_map(body, loop_nodes, init_map, local_2d,
                           expr_map, substituted, horizontal=None):
    """
    Walk *body* and for any :any:`Array` reference to an init-mapped
    array that appears outside all vertical loops, add a substitution
    entry to *expr_map*.

    The substitution replaces the outside-loop array reference with
    the RHS expression of the first in-loop assignment to that array,
    evaluated at the appropriate level index.

    Non-vertical subscripts (e.g. ``JL``) in the substitution expression
    are replaced with the appropriate range:

    * Horizontal loop variables matching ``horizontal.index`` are
      replaced with ``horizontal.bounds`` (e.g. ``KIDIA:KFDIA``).
    * Other scalar subscripts are replaced with bare ``:``
      (:any:`RangeIndex`).

    Parameters
    ----------
    body : :any:`Section` or tuple
        The routine body to walk.
    loop_nodes : set
        IR nodes to skip (the vertical loops).
    init_map : dict
        ``{array_name_lower: Assignment}`` — the first in-loop write.
    local_2d : dict
        ``{array_name_lower: {'klev_dim_idx': int, ...}}`` — info about
        each local array with a KLEV dimension.
    expr_map : dict
        Accumulator for ``{old_expr: new_expr}`` substitution entries.
    substituted : set
        Accumulator for lowercase array names that were substituted.
    horizontal : :any:`Dimension`, optional
        When provided, horizontal loop variables in substitution
        expressions use bounded ranges from ``horizontal.bounds``.
    """

    class _InitSubstBuilder(_SkipNodesVisitor):
        def __init__(self):
            super().__init__(loop_nodes)

        def visit_Node(self, o, **kwargs):
            all_vars = FindVariables().visit(o)
            for var in all_vars:
                var_lower = var.name.lower()
                if var_lower not in init_map:
                    continue
                if not hasattr(var, 'dimensions') or not var.dimensions:
                    continue

                init_assign = init_map[var_lower]
                info_entry = local_2d[var_lower]
                klev_dim = info_entry['klev_dim_idx']
                if klev_dim >= len(var.dimensions):
                    continue
                level_idx = var.dimensions[klev_dim]

                # Verify non-vertical subscripts match between the
                # reference and the init expression (e.g. reject
                # ZQX(:,:,JM) when init was for ZQX(:,:,NCLDQV)).
                # Range subscripts (`:`) are considered compatible with
                # any scalar subscript (e.g. `:` matches `JL`).
                init_dims = init_assign.lhs.dimensions
                if len(var.dimensions) == len(init_dims):
                    mismatch = False
                    for di, (ref_d, init_d) in enumerate(
                            zip(var.dimensions, init_dims)):
                        if di == klev_dim:
                            continue  # vertical dim — allowed to differ
                        if isinstance(ref_d, sym.RangeIndex):
                            continue
                        if isinstance(init_d, sym.RangeIndex):
                            continue
                        if ref_d != init_d:
                            mismatch = True
                            break
                    if mismatch:
                        continue

                # Build RHS with JK replaced by the level index
                jk_var = init_assign.lhs.dimensions[klev_dim]
                rhs_substituted = _substitute_jk_in_expr(
                    init_assign.rhs, jk_var, level_idx)

                # If the target reference uses ':' (RangeIndex) for a
                # dimension where the init expression used a scalar loop
                # variable (e.g. JL), replace that loop variable with a
                # range in the substituted RHS to avoid out-of-scope refs.
                # When the horizontal Dimension is available we use its
                # bounds (e.g. ``KIDIA:KFDIA``) so that the SCC pipeline's
                # ``resolve_vector_dimension`` can later resolve the range
                # to the scalar index.  Without a horizontal Dimension we
                # fall back to an unbounded ``:`` (sufficient for idem).
                init_dims = init_assign.lhs.dimensions
                if len(var.dimensions) == len(init_dims):
                    if horizontal is not None:
                        _lower = sym.Variable(name=horizontal.bounds[0], scope=None)
                        _upper = sym.Variable(name=horizontal.bounds[1], scope=None)
                        range_idx = sym.RangeIndex((_lower, _upper, None))
                    else:
                        range_idx = sym.RangeIndex((None, None, None))
                    for di, (ref_d, init_d) in enumerate(
                            zip(var.dimensions, init_dims)):
                        if di == klev_dim:
                            continue
                        if (isinstance(ref_d, sym.RangeIndex)
                                and isinstance(init_d, (sym.Scalar, sym.DeferredTypeSymbol))
                                and not isinstance(init_d, sym.IntLiteral)):
                            rhs_substituted = _substitute_jk_in_expr(
                                rhs_substituted, init_d, range_idx)

                expr_map[var] = rhs_substituted
                substituted.add(var_lower)

    _InitSubstBuilder().visit(body)


def _apply_subst_outside_loops(routine, loop_nodes, expr_map):
    """
    Apply *expr_map* substitutions only to top-level nodes outside
    vertical loops.
    """
    nodes = routine.body.body if hasattr(routine.body, 'body') else routine.body
    new_body = []
    changed = False

    for node in nodes:
        if node in loop_nodes:
            new_body.append(node)
            continue
        new_node = SubstituteExpressionsSkipLHS(expr_map).visit(node)
        new_body.append(new_node)
        if new_node is not node:
            changed = True

    if changed:
        routine.body = routine.body.clone(body=tuple(new_body))


def _collect_refs_outside_loops(body, loop_nodes):
    """
    Collect all variable names referenced outside any node in
    *loop_nodes*.  Returns a set of lowercase variable names.
    """

    class _RefCollector(_SkipNodesVisitor):
        def __init__(self):
            super().__init__(loop_nodes)
            self.refs = set()

        def _add_vars(self, expr):
            for var in FindVariables().visit(expr):
                self.refs.add(var.name.lower())

        def visit_Conditional(self, o, **kwargs):
            self._add_vars(o.condition)
            self.visit(o.body, **kwargs)
            if o.else_body:
                self.visit(o.else_body, **kwargs)

        def visit_Loop(self, o, **kwargs):
            self._add_vars(o.variable)
            for bound in (o.bounds.lower, o.bounds.upper, o.bounds.step):
                if bound is not None:
                    self._add_vars(bound)
            self.visit(o.body, **kwargs)

        def visit_Node(self, o, **kwargs):
            self._add_vars(o)

    collector = _RefCollector()
    collector.visit(body)
    return collector.refs


def _find_demotable_arrays(routine, vertical_index, vertical_size):
    """
    Identify local arrays with a vertical dimension that can be safely
    demoted after loop absorption.

    An array is demotable if:

    1. It is a local variable (not an argument, not imported).
    2. It has ``vertical_size`` (or ``vertical_size + 1``) in its shape.
    3. It is NOT accessed outside all vertical loops in the routine.
    4. Within every vertical loop, every access uses offset 0.
    5. It is NOT passed as an argument to any subroutine/function call.
    6. It is NOT referenced in more than one vertical loop (otherwise
       data written in one loop and read in another would be lost after
       demotion).

    Parameters
    ----------
    routine : :any:`Subroutine`
    vertical_index : str
    vertical_size : str

    Returns
    -------
    list of str
        Lowercase names of arrays that are safe to demote.
    """
    vertical_size_lower = vertical_size.lower()

    arg_names = {v.name.lower() for v in routine.arguments}

    # Variables passed to calls (not safe to demote)
    call_arg_names = _collect_call_arg_names(routine)

    # Find all local arrays with KLEV dimension
    candidates = {}
    for var in routine.variables:
        vname = var.name.lower()
        if vname in arg_names:
            continue
        shape = (getattr(var.type, 'shape', None)
                 or getattr(var, 'shape', None))
        if not shape:
            continue
        has_klev = False
        for s in shape:
            # Use expression-level comparison (StrCompareMixin handles
            # case-insensitive equality) and _is_klev_plus_n for KLEV+N.
            if s == vertical_size_lower or _is_klev_plus_n(s, vertical_size_lower):
                has_klev = True
                break
        if has_klev and vname not in call_arg_names:
            candidates[vname] = var

    if not candidates:
        return []

    # Collect all vertical loops
    all_vloops = _collect_vertical_loops(routine.body, vertical_index)
    loop_nodes = _collect_loop_node_set(all_vloops)

    # Check that no candidate is accessed outside vertical loops
    outside_refs = _collect_refs_outside_loops(routine.body, loop_nodes)
    safe = set(candidates.keys()) - outside_refs

    if not safe:
        return []

    # Exclude arrays referenced in more than one vertical loop.
    # After demotion the vertical dimension is removed, so data written
    # in one loop and read in another would be lost.
    if len(all_vloops) > 1:
        arr_loop_count = defaultdict(int)
        for loop, _cond in all_vloops:
            loop_vars = {v.name.lower() for v in FindVariables().visit(loop.body)}
            for cand in safe:
                if cand in loop_vars:
                    arr_loop_count[cand] += 1
        multi_loop = {name for name, count in arr_loop_count.items() if count > 1}
        if multi_loop:
            info('[_find_demotable_arrays] Excluding %s: used in multiple '
                 'vertical loops', ', '.join(sorted(multi_loop)))
            safe -= multi_loop

    if not safe:
        return []

    # Check that within each vertical loop, every access is at offset 0
    for loop, _cond in all_vloops:
        access_map = classify_array_access_offsets(
            loop, loop_var=loop.variable)
        for (arr_name, _dim_idx), offset_map in access_map.items():
            arr_lower = arr_name.lower()
            if arr_lower not in safe:
                continue
            for offset in offset_map:
                if offset != 0:
                    safe.discard(arr_lower)

    return sorted(safe)


def _find_dead_loops_all(routine, vertical_index):
    """
    Identify vertical loops whose written outputs are never read by
    any other code in the routine.

    Unlike the variant in ``vertical_complete`` this version
    does **not** exclude any loop — every vertical loop is a candidate.

    .. note::

       A future improvement could add an optional ``exclude_loop``
       parameter to unify this with the ``_find_dead_loops`` variant
       that excludes a main loop.

    Returns a list of top-level IR nodes (Loop or Conditional wrappers)
    that are dead and can be safely removed.
    """
    all_vloops = _collect_vertical_loops(routine.body, vertical_index)
    dead = []

    arg_names = {v.name.lower() for v in routine.arguments}

    for loop, cond_wrapper in all_vloops:
        written_names = set()
        for assign in FindNodes(ir.Assignment).visit(loop.body):
            written_names.add(assign.lhs.name.lower())

        if not written_names:
            continue

        # Writing to a routine argument is never dead
        if written_names & arg_names:
            continue

        exclude_node = cond_wrapper if cond_wrapper is not None else loop

        if not _any_read_outside_node(routine.body, exclude_node,
                                       written_names):
            dead.append(exclude_node)

    return dead


def _build_carry_expr_entries(all_vars, arr_name, dim_idx, loop_var,
                              target_offsets, carry_decl,
                              alt_carry_decl=None, alt_offsets=None):
    """
    Build expression-substitution map entries that replace array
    accesses at specific loop-variable offsets with a carry variable.

    Parameters
    ----------
    all_vars : iterable of expression
        Result of ``FindVariables(unique=False).visit(loop.body)``.
    arr_name : str
        Name of the original array.
    dim_idx : int
        Index of the vertical dimension in the array's subscript list.
    loop_var : expression
        Loop induction variable (e.g. ``JK``).
    target_offsets : set or callable
        Either a set of integer offsets whose references should be
        rewritten to *carry_decl*, or a callable
        ``(offset) -> bool``.
    carry_decl : expression
        The carry variable to substitute in.
    alt_carry_decl : expression, optional
        A second carry variable (e.g. ``_next``) used for a second
        group of offsets.
    alt_offsets : set or callable, optional
        Offsets whose references should be rewritten to
        *alt_carry_decl* instead of *carry_decl*.

    Returns
    -------
    dict
        Mapping ``{old_expr: new_expr}`` suitable for merging into an
        ``expr_map`` passed to ``SubstituteExpressions``.
    """
    arr_lower = arr_name.lower()
    entries = {}

    def _match(offset, target):
        if callable(target):
            return target(offset)
        return offset in target

    for v in all_vars:
        if not isinstance(v, sym.Array) or not v.dimensions:
            continue
        if v.name.lower() != arr_lower:
            continue
        if dim_idx >= len(v.dimensions):
            continue
        offset = extract_offset(v.dimensions[dim_idx], loop_var)
        if offset is None:
            continue

        new_dims = tuple(
            d for i, d in enumerate(v.dimensions) if i != dim_idx
        )

        if alt_carry_decl is not None and alt_offsets is not None and _match(offset, alt_offsets):
            entries[v] = alt_carry_decl.clone(
                dimensions=new_dims if new_dims else None
            )
        elif _match(offset, target_offsets):
            entries[v] = carry_decl.clone(
                dimensions=new_dims if new_dims else None
            )

    return entries


def _build_save_assignment(orig_decl, carry_decl, orig_shape, dim_idx,
                           save_index_expr, actual_non_vert_dims,
                           out_of_loop_dim_fn):
    """
    Build an assignment ``carry = X(..., <save_index>, ...)`` that
    captures the current iteration's value for use in the next.

    Parameters
    ----------
    orig_decl : expression
        The original array declaration (with full shape).
    carry_decl : expression
        The carry variable declaration.
    orig_shape : tuple
        Shape of the original array.
    dim_idx : int
        Index of the vertical dimension.
    save_index_expr : expression
        The index expression for the vertical dimension in the save
        statement (e.g. ``loop_var`` or ``Sum((loop_var, IntLiteral(1)))``).
    actual_non_vert_dims : tuple or None
        Actual non-vertical dimension expressions from the loop body.
    out_of_loop_dim_fn : callable
        ``(dim_expr) -> dim_expr`` that sanitises dimension
        expressions for out-of-loop context (replaces non-horizontal
        loop variables with ``':'``).

    Returns
    -------
    :any:`Assignment`
    """
    save_orig_dims = []
    save_carry_dims = []
    nv_idx = 0
    for i, _s in enumerate(orig_shape):
        if i == dim_idx:
            save_orig_dims.append(save_index_expr)
        else:
            if (actual_non_vert_dims is not None
                    and nv_idx < len(actual_non_vert_dims)):
                dim_expr = out_of_loop_dim_fn(actual_non_vert_dims[nv_idx])
            else:
                dim_expr = sym.RangeIndex((None, None, None))
            save_orig_dims.append(dim_expr)
            save_carry_dims.append(dim_expr)
            nv_idx += 1
    save_rhs = orig_decl.clone(dimensions=tuple(save_orig_dims))
    save_lhs = carry_decl.clone(
        dimensions=tuple(save_carry_dims) if save_carry_dims else None
    )
    return ir.Assignment(lhs=save_lhs, rhs=save_rhs)


def _convert_all_carries(routine, loop, vertical_size,
                         carry_suffix='_vc', next_suffix='_next',
                         horizontal=None):
    """
    Convert all carry and stencil patterns in *loop* to scalar carry
    variables, making the loop body level-local.

    Shared logic for building expression substitution entries and save
    assignments is delegated to :func:`_build_carry_expr_entries` and
    :func:`_build_save_assignment`.

    Four patterns are recognised:

    **Pattern A** — cumulative sum (read at JK-1, write at JK)::

        IF (JK==1) THEN; X(JL,JK) = 0; ELSE; X(JL,JK) = X(JL,JK-1); END IF
        X(JL,JK) = X(JL,JK) + delta

    Converted with 1 carry: ``x_vc``.

    **Pattern B-simple** — forward propagation, no readback
    (read at JK, write at JK+1, no reads at JK+1)::

        X(JL,JK+1) = f(X(JL,JK))

    Converted with 1 carry: ``x_vc``.

    **Pattern B-readback** — forward propagation with same-iteration
    readback (read at JK, write at JK+1, AND reads at JK+1)::

        X(JL,JK+1) = f(X(JL,JK))
        Y = g(X(JL,JK+1))

    Converted with 2 carries: ``x_vc`` (incoming) + ``x_next``
    (outgoing).  A rotate ``x_vc = x_next`` is appended.

    **Stencil** — read-only backward access (read at JK-1, never
    written in the loop, local array only)::

        Y = f(X(JL,JK-1))

    Converted with 1 carry: ``x_vc``.  Init: ``x_vc = X(:, lower-1)``.
    Save: ``x_vc = X(:, JK)`` at end of loop body.

    Parameters
    ----------
    routine : :any:`Subroutine`
    loop : :any:`Loop`
        The vertical loop to transform.
    vertical_size : str
    carry_suffix : str
    next_suffix : str
    horizontal : :any:`Dimension`, optional
        When provided, horizontal loop variables in carry init, save,
        and rotate statements are replaced with bounded ranges from
        ``horizontal.bounds`` so that the SCC pipeline's
        ``resolve_vector_dimension`` can resolve them.

    Returns
    -------
    tuple
        ``(new_loop, init_stmts, conversions)``
    """
    vertical_size_lower = vertical_size.lower()
    loop_var = loop.variable

    # --- Classify accesses and dependencies ---
    access_map = classify_array_access_offsets(loop, loop_var=loop_var)
    deps = array_loop_carried_dependencies(loop, loop_var=loop_var)

    # Build per-array access summary: {arr_lower: {(arr_name, dim_idx): offset_map}}
    arr_dim_map = defaultdict(lambda: defaultdict(dict))
    for (arr_name, dim_idx), offset_map in access_map.items():
        arr_dim_map[arr_name.lower()][(arr_name, dim_idx)] = offset_map

    # Arguments and call-args (stencil conversion applies only to locals)
    arg_names = {v.name.lower() for v in routine.arguments}
    call_arg_names = _collect_call_arg_names(routine)

    var_map_ci = CaseInsensitiveDict(
        {v.name: v for v in routine.variables}
    )

    # --- Collect carry specs ---
    #
    # Each spec: (arr_name, dim_idx, pattern, carry_names_dict)
    #   pattern: 'A', 'B_simple', 'B_readback', 'stencil'
    #   carry_names_dict: {'vc': name} or {'vc': name, 'next': name}
    carry_specs = []
    seen = set()

    # 1) Detect flow dependencies (Pattern A and B)
    for arr_name, dep_list in deps.items():
        for dep in dep_list:
            if dep['type'] != 'flow':
                continue
            dim_idx = dep['dim_index']
            key = (arr_name.lower(), dim_idx)
            if key in seen:
                continue

            w_off = dep['write_offset']
            r_off = dep['read_offset']

            if w_off == 0 and r_off == -1:
                # Pattern A
                carry_specs.append((arr_name, dim_idx, 'A', {}))
                seen.add(key)
            elif w_off == 1 and r_off == 0:
                # Pattern B — check for readback at +1
                has_readback = False
                for (an, di), omap in access_map.items():
                    if an.lower() == arr_name.lower() and di == dim_idx:
                        if 1 in omap and 'read' in omap[1]:
                            has_readback = True
                            break
                if has_readback:
                    carry_specs.append((arr_name, dim_idx, 'B_readback', {}))
                else:
                    carry_specs.append((arr_name, dim_idx, 'B_simple', {}))
                seen.add(key)

    # 2) Detect stencil patterns (read at non-zero offset, no write at any offset)
    for arr_lower, dim_entries in arr_dim_map.items():
        # Stencils only for locals
        if arr_lower in arg_names or arr_lower in call_arg_names:
            continue
        orig_decl = var_map_ci.get(arr_lower)
        if orig_decl is None:
            continue
        shape = orig_decl.type.shape if orig_decl.type.shape else ()
        if not shape:
            continue

        for (arr_name, dim_idx), offset_map in dim_entries.items():
            key = (arr_lower, dim_idx)
            if key in seen:
                continue
            # Check dimension is vertical
            if dim_idx >= len(shape):
                continue
            s = shape[dim_idx]
            if not (s == vertical_size_lower or
                    _is_klev_plus_n(s, vertical_size_lower)):
                continue

            # Stencil only for *backward* offsets (negative).
            # Forward (+1) reads in read-only arrays cannot be converted
            # to a simple carry (would require look-ahead).
            negative = [off for off in offset_map if off < 0]
            if not negative:
                continue

            # Must be read-only at ALL offsets in this dimension
            has_write = False
            for off, atypes in offset_map.items():
                if 'write' in atypes:
                    has_write = True
                    break
            if has_write:
                continue

            carry_specs.append((arr_name, dim_idx, 'stencil', {}))
            seen.add(key)

    if not carry_specs:
        return loop, [], []

    # --- Build carry variables, expr maps, init/save statements ---

    expr_map = {}
    init_stmts = []
    save_stmts = []
    rotate_stmts = []
    cond_to_remove = []
    conversions = []

    for arr_name, dim_idx, pattern, _ in carry_specs:
        orig_decl = var_map_ci.get(arr_name)
        if orig_decl is None:
            warning('[convert_all_carries] Array %r not found — skipping',
                    arr_name)
            continue

        orig_shape = orig_decl.type.shape if orig_decl.type.shape else ()
        if not orig_shape or dim_idx >= len(orig_shape):
            warning('[convert_all_carries] Cannot determine shape for '
                    '%r dim %d — skipping', arr_name, dim_idx)
            continue

        # --- Create carry variable(s) ---
        carry_shape = tuple(
            s for i, s in enumerate(orig_shape) if i != dim_idx
        )
        carry_type = orig_decl.type.clone(
            shape=carry_shape if carry_shape else None,
            intent=None
        )

        carry_name = arr_name.lower() + carry_suffix
        carry_var = sym.Variable(
            name=carry_name, type=carry_type,
            dimensions=carry_shape if carry_shape else None,
            scope=routine
        )
        if carry_name.lower() not in CaseInsensitiveDict(
            {v.name: v for v in routine.variables}
        ):
            routine.variables += as_tuple(carry_var)
        carry_decl = routine.variable_map.get(carry_name, carry_var)

        # For B_readback, also create the 'next' variable
        next_decl = None
        if pattern == 'B_readback':
            next_name = arr_name.lower() + next_suffix
            next_var = sym.Variable(
                name=next_name, type=carry_type,
                dimensions=carry_shape if carry_shape else None,
                scope=routine
            )
            if next_name.lower() not in CaseInsensitiveDict(
                {v.name: v for v in routine.variables}
            ):
                routine.variables += as_tuple(next_var)
            next_decl = routine.variable_map.get(next_name, next_var)

        # --- Extract actual non-vertical subscripts from loop body ---
        # Scan loop body for the first reference to this array and capture
        # the non-vertical dimension expressions (e.g. JL).  These are used
        # instead of bare ':' (RangeIndex) in init/save/rotate statements
        # so that downstream SCCDevector/SCCDemote produce valid Fortran.
        _body_vars = FindVariables(unique=False).visit(loop.body)
        actual_non_vert_dims = None
        for _bv in _body_vars:
            if not isinstance(_bv, sym.Array) or not _bv.dimensions:
                continue
            if _bv.name.lower() != arr_name.lower():
                continue
            if dim_idx >= len(_bv.dimensions):
                continue
            _candidate = tuple(
                d for i, d in enumerate(_bv.dimensions) if i != dim_idx
            )
            if _candidate:
                actual_non_vert_dims = _candidate
                break

        # --- Helper closures (called within the same loop iteration) ---
        # pylint: disable=cell-var-from-loop

        # Build the range index for out-of-loop (init/save/rotate) context.
        # When horizontal bounds are available, use them (e.g. KIDIA:KFDIA)
        # so that resolve_vector_dimension can later resolve the range to
        # the horizontal index variable.  Fall back to bare ':' otherwise.
        if horizontal is not None:
            _h_lower = sym.Variable(name=horizontal.bounds[0], scope=None)
            _h_upper = sym.Variable(name=horizontal.bounds[1], scope=None)
            _ool_range = sym.RangeIndex((_h_lower, _h_upper, None))
        else:
            _ool_range = sym.RangeIndex((None, None, None))

        # range dims for carry (in-body context)
        def _carry_range_dims():
            if actual_non_vert_dims is not None:
                return actual_non_vert_dims
            if not carry_shape:
                return None
            if len(carry_shape) == 1 and horizontal is not None:
                return (_ool_range,)
            return tuple(
                sym.RangeIndex((None, None, None))
                for _ in carry_shape
            )

        # range dims for init/rotate (out-of-loop context)
        # All non-vertical dimensions (including the horizontal index)
        # are replaced with ':' (RangeIndex) so that init, save, and
        # rotate statements use array-section notation.  This makes the
        # transformation independent of whether SCCDevector has already
        # converted the horizontal loop variable to a scalar parameter.

        def _init_rotate_dims():
            """Dims for init/rotate statements (all non-vertical dims become ranges).

            Uses bounded range (``KIDIA:KFDIA``) for carry variables with
            exactly one non-vertical dimension (assumed horizontal).
            For multi-dimensional carries (e.g. ``(nlon, NCLV)``), bare
            ``:`` is used because we cannot reliably identify which
            dimension is horizontal from shape alone.
            """
            if not carry_shape:
                return None
            if len(carry_shape) == 1 and horizontal is not None:
                return (_ool_range,)
            return tuple(
                sym.RangeIndex((None, None, None))
                for _ in carry_shape
            )

        def _out_of_loop_dim(dim_expr):
            """Sanitize a single dim expr for out-of-loop (init/save/rotate) context.

            Replaces scalar loop variables with a range index.  When the
            variable matches the horizontal index (e.g. ``JL``), a bounded
            range (``KIDIA:KFDIA``) is used; other scalar variables get a
            bare ``:``.  Integer literals and non-variable expressions are
            kept as-is.
            """
            if isinstance(dim_expr, sym.Scalar) and not isinstance(dim_expr, sym.IntLiteral):
                if (horizontal is not None
                        and dim_expr == horizontal.index):
                    return _ool_range
                return sym.RangeIndex((None, None, None))
            return dim_expr

        # pylint: enable=cell-var-from-loop

        # --- Init statement ---
        if pattern == 'stencil':
            # Init: carry = X(:, lower-1, :)  (guarded against OOB)
            lower_expr = loop.bounds.children[0]
            init_idx = sym.Sum((lower_expr, sym.IntLiteral(-1)))
            init_orig_dims = []
            init_carry_dims = []
            nv_idx = 0
            for i, _s in enumerate(orig_shape):
                if i == dim_idx:
                    init_orig_dims.append(init_idx)
                else:
                    if actual_non_vert_dims is not None and nv_idx < len(actual_non_vert_dims):
                        dim_expr = actual_non_vert_dims[nv_idx]
                        # For init (outside loops), replace all loop
                        # variables (including horizontal) with ':'
                        dim_expr = _out_of_loop_dim(dim_expr)
                    else:
                        dim_expr = sym.RangeIndex((None, None, None))
                    init_orig_dims.append(dim_expr)
                    init_carry_dims.append(dim_expr)
                    nv_idx += 1
            stencil_rhs = orig_decl.clone(dimensions=tuple(init_orig_dims))
            stencil_lhs = carry_decl.clone(
                dimensions=tuple(init_carry_dims) if init_carry_dims else None
            )
            stencil_assign = ir.Assignment(lhs=stencil_lhs, rhs=stencil_rhs)

            # Guard against OOB: if lower == 1, X(:, 0) is invalid.
            # Wrap in: IF (lower > 1) THEN carry = X(:,lower-1) ELSE carry = 0 END IF
            zero_lhs = carry_decl.clone(
                dimensions=tuple(init_carry_dims) if init_carry_dims else None
            )
            zero_rhs = _make_zero_literal(orig_decl.type)
            zero_assign = ir.Assignment(lhs=zero_lhs, rhs=zero_rhs)

            guard_cond = sym.Comparison(
                operator='>', left=lower_expr, right=sym.IntLiteral(1)
            )
            init_stmts.append(ir.Conditional(
                condition=guard_cond,
                body=(stencil_assign,),
                else_body=(zero_assign,),
            ))
        else:
            # Init: carry = 0.0
            init_dims = _init_rotate_dims()
            init_lhs = carry_decl.clone(dimensions=init_dims)
            init_rhs = _make_zero_literal(orig_decl.type)
            init_stmts.append(ir.Assignment(lhs=init_lhs, rhs=init_rhs))

        # --- Build expression substitution map ---
        all_vars = FindVariables(unique=False).visit(loop.body)

        if pattern == 'A':
            # Replace reads at JK-1 with carry
            expr_map.update(_build_carry_expr_entries(
                all_vars, arr_name, dim_idx, loop_var,
                target_offsets={-1}, carry_decl=carry_decl
            ))

            # Save: carry = X(JL, JK) at end of body
            save_stmts.append(_build_save_assignment(
                orig_decl, carry_decl, orig_shape, dim_idx,
                save_index_expr=loop_var,
                actual_non_vert_dims=actual_non_vert_dims,
                out_of_loop_dim_fn=_out_of_loop_dim
            ))

            # Remove IF(JK==1) conditionals
            _mark_jk1_conditionals(loop, arr_name, cond_to_remove)

        elif pattern == 'B_simple':
            # Replace reads at JK (offset 0) with carry
            expr_map.update(_build_carry_expr_entries(
                all_vars, arr_name, dim_idx, loop_var,
                target_offsets={0}, carry_decl=carry_decl
            ))

            # Save: carry = X(JL, JK+1) at end of body
            save_stmts.append(_build_save_assignment(
                orig_decl, carry_decl, orig_shape, dim_idx,
                save_index_expr=sym.Sum((loop_var, sym.IntLiteral(1))),
                actual_non_vert_dims=actual_non_vert_dims,
                out_of_loop_dim_fn=_out_of_loop_dim
            ))

        elif pattern == 'B_readback':
            # Two carries: _vc for reads at offset 0, _next for
            # write at +1 and reads at +1
            expr_map.update(_build_carry_expr_entries(
                all_vars, arr_name, dim_idx, loop_var,
                target_offsets={0}, carry_decl=carry_decl,
                alt_carry_decl=next_decl, alt_offsets={1}
            ))

            # Rotate: carry = next at end of body
            rot_lhs = carry_decl.clone(dimensions=_init_rotate_dims())
            rot_rhs = next_decl.clone(dimensions=_init_rotate_dims())
            rotate_stmts.append(ir.Assignment(lhs=rot_lhs, rhs=rot_rhs))

            # NOTE: Write-back for argument arrays (populating the original
            # output array from _next) is handled separately in
            # _insert_writebacks_for_argument_carries(), which runs AFTER
            # Phase 2b cross-loop carry substitution.  Generating write-backs
            # here would cause Phase 2b's SubstituteExpressions to turn them
            # into self-assignments (e.g. ``arr_next = arr_next``) that
            # Phase 4a then removes.

        elif pattern == 'stencil':
            # Replace reads at negative offsets with carry
            expr_map.update(_build_carry_expr_entries(
                all_vars, arr_name, dim_idx, loop_var,
                target_offsets=lambda off: off < 0, carry_decl=carry_decl
            ))

            # Save: carry = X(JL, JK) at end of body
            save_stmts.append(_build_save_assignment(
                orig_decl, carry_decl, orig_shape, dim_idx,
                save_index_expr=loop_var,
                actual_non_vert_dims=actual_non_vert_dims,
                out_of_loop_dim_fn=_out_of_loop_dim
            ))

        # Record conversion
        conv_entry = {
            'array': arr_name,
            'carry': carry_name,
            'pattern': pattern,
            'dim_index': dim_idx,
        }
        if next_decl is not None:
            conv_entry['next'] = next_decl.name
        conversions.append(conv_entry)
        info('[vertical_utils] %s → %s (pattern %s)',
             arr_name, carry_name, pattern)

    if not expr_map and not cond_to_remove:
        return loop, [], []

    # --- Apply transformations to loop body ---

    # 1) Remove IF(JK==1) conditionals (Pattern A)
    new_body = loop.body
    if cond_to_remove:
        cond_map = {}
        for cond, replacement_stmts in cond_to_remove:
            cond_map[cond] = (
                tuple(replacement_stmts) if replacement_stmts else None
            )
        new_body = Transformer(cond_map).visit(new_body)

    # 2) Apply expression substitution
    new_body = SubstituteExpressions(expr_map).visit(new_body)

    # 3) Append save + writeback + rotate statements
    body_tuple = as_tuple(new_body)
    body_tuple = body_tuple + tuple(save_stmts) + tuple(rotate_stmts)

    new_loop = loop.clone(body=body_tuple)
    return new_loop, init_stmts, conversions


def _substitute_init_expressions_all_loops(routine, vertical_index,
                                            vertical_size, horizontal=None):
    """
    For each local 2-D+ array with a KLEV dimension, find the first
    ``X(JL, JK) = f(args)`` assignment (in source order across all
    vertical loops) and substitute outside-loop references to that
    array with the RHS expression evaluated at the appropriate level.

    This is the all-loops variant of ``_substitute_init_expressions``
    from ``vertical_complete``: it searches *all* vertical loops for
    init assignments rather than only a designated main loop.

    Parameters
    ----------
    routine : :any:`Subroutine`
    vertical_index : str
    vertical_size : str
    horizontal : :any:`Dimension`, optional
        Forwarded to :func:`_build_init_subst_map` to replace
        horizontal loop variables with bounded ranges.

    Returns
    -------
    set of str
        Lowercase array names that were substituted.
    """
    vertical_index_lower = vertical_index.lower()
    vertical_size_lower = vertical_size.lower()

    arg_names = {v.name.lower() for v in routine.arguments}

    # Find local arrays with a KLEV dimension
    local_kd = {}
    for var in routine.variables:
        if var.name.lower() in arg_names:
            continue
        shape = getattr(var.type, 'shape', None) or getattr(var, 'shape', None)
        if not shape:
            continue
        for idx, s in enumerate(shape):
            if s == vertical_size_lower or _is_klev_plus_n(s, vertical_size_lower):
                local_kd[var.name.lower()] = {
                    'var': var, 'klev_dim_idx': idx, 'shape': shape}
                break

    if not local_kd:
        return set()

    # Scan ALL vertical loops for init assignments
    all_vloops = _collect_vertical_loops(routine.body, vertical_index)
    init_map = {}  # array_name_lower -> Assignment node

    for loop, _cond in all_vloops:
        for assign in FindNodes(ir.Assignment).visit(loop.body):
            lhs = assign.lhs
            lhs_name = lhs.name.lower()
            if lhs_name not in local_kd:
                continue
            if lhs_name in init_map:
                continue  # Only use the first assignment

            dims = getattr(lhs, 'dimensions', None)
            if not dims:
                continue
            info_entry = local_kd[lhs_name]
            klev_dim = info_entry['klev_dim_idx']
            if klev_dim >= len(dims):
                continue
            jk_sub = dims[klev_dim]
            if not (isinstance(jk_sub, (sym.Scalar, sym.DeferredTypeSymbol))
                    and jk_sub.name.lower() == vertical_index_lower):
                continue

            # RHS must not reference other KLEV locals
            rhs_vars = FindVariables().visit(assign.rhs)
            rhs_uses_local = False
            for rv in rhs_vars:
                rv_lower = rv.name.lower()
                if rv_lower == lhs_name:
                    continue
                if rv_lower in local_kd and rv_lower not in arg_names:
                    rhs_uses_local = True
                    break
            if rhs_uses_local:
                continue

            init_map[lhs_name] = assign

    if not init_map:
        return set()

    # Identify all vertical loop nodes (to exclude from substitution)
    loop_nodes = _collect_loop_node_set(all_vloops)

    # Build expression substitution map for outside-loop reads
    expr_map = {}
    substituted = set()
    _build_init_subst_map(routine.body, loop_nodes, init_map, local_kd,
                          expr_map, substituted, horizontal)

    if expr_map:
        _apply_subst_outside_loops(routine, loop_nodes, expr_map)

    return substituted


def _is_zero_literal(expr):
    """Return True if *expr* is a zero literal (int or float).

    Loki stores FloatLiteral.value as a **string** (e.g. ``'0.0'``)
    to preserve precision, so we must convert before comparing.
    """
    if isinstance(expr, sym.IntLiteral):
        return expr.value == 0
    if isinstance(expr, sym.FloatLiteral):
        try:
            return float(expr.value) == 0.0
        except (ValueError, TypeError):
            return False
    # Handle IntrinsicLiteral (Loki's catch-all for complex/BOZ constants)
    if isinstance(expr, sym.IntrinsicLiteral):
        try:
            return float(expr.value) == 0.0
        except (ValueError, TypeError):
            return False
    # Generic fallback for any node with a value attribute
    if hasattr(expr, 'value'):
        try:
            return float(expr.value) == 0.0
        except (ValueError, TypeError):
            pass
    return False


def _find_zero_inits_outside_loops(body, loop_nodes, klev_locals,
                                    zero_init_map):
    """
    Walk *body* outside loop nodes and find whole-array zero-init
    assignments to vertical locals.
    """

    class _ZeroInitFinder(_SkipNodesVisitor):
        def __init__(self):
            super().__init__(loop_nodes)

        # Don't recurse into Loop bodies (only Section/Associate/Conditional)
        def visit_Loop(self, o, **kwargs):
            pass

        def visit_Assignment(self, o, **kwargs):  # pylint: disable=unused-argument
            lhs = o.lhs
            lhs_name = lhs.name.lower()
            if lhs_name not in klev_locals:
                return
            dims = getattr(lhs, 'dimensions', None)
            if not dims:
                return
            if not all(isinstance(d, sym.RangeIndex) for d in dims):
                return
            if _is_zero_literal(o.rhs):
                zero_init_map[lhs_name] = o

    _ZeroInitFinder().visit(body)


def _collect_outside_refs_for_array(body, loop_nodes, arr_name,
                                     exclude_node=None):
    """
    Collect all IR nodes outside loop_nodes that reference *arr_name*
    (case-insensitive), excluding *exclude_node* if given.

    Returns a list of nodes.
    """

    class _ArrayRefCollector(_SkipNodesVisitor):
        def __init__(self):
            skip = set(loop_nodes)
            if exclude_node is not None:
                skip.add(exclude_node)
            super().__init__(skip)
            self.refs = []

        # Don't recurse into Loop bodies
        def visit_Loop(self, o, **kwargs):
            pass

        def visit_Conditional(self, o, **kwargs):
            # Check condition for references
            for var in FindVariables().visit(o.condition):
                if var.name.lower() == arr_name:
                    self.refs.append(o)
                    break
            self.visit(o.body, **kwargs)
            if o.else_body:
                self.visit(o.else_body, **kwargs)

        def visit_Node(self, o, **kwargs):
            for var in FindVariables().visit(o):
                if var.name.lower() == arr_name:
                    self.refs.append(o)
                    break

    collector = _ArrayRefCollector()
    collector.visit(body)
    return collector.refs


def _remove_whole_array_zero_inits(routine, vertical_index, vertical_size):
    """
    Remove whole-array zero-initialisation statements (e.g.
    ``X(:,:,:) = 0.0``) that sit outside all vertical loops, for local
    arrays that would become demotable if the init were removed.

    An init is removed only if it is the **sole** outside-loop reference
    to that array.

    Parameters
    ----------
    routine : :any:`Subroutine`
    vertical_index : str
    vertical_size : str

    Returns
    -------
    list of str
        Lowercase names of arrays whose inits were removed.
    """
    vertical_size_lower = vertical_size.lower()
    arg_names = {v.name.lower() for v in routine.arguments}

    # Variables passed to calls
    call_arg_names = _collect_call_arg_names(routine)

    # Find local KLEV arrays
    klev_locals = {}
    for var in routine.variables:
        vname = var.name.lower()
        if vname in arg_names or vname in call_arg_names:
            continue
        shape = getattr(var.type, 'shape', None) or getattr(var, 'shape', None)
        if not shape:
            continue
        has_klev = False
        for s in shape:
            if s == vertical_size_lower or _is_klev_plus_n(s, vertical_size_lower):
                has_klev = True
                break
        if has_klev:
            klev_locals[vname] = var

    if not klev_locals:
        return []

    # Collect all vertical loop nodes
    all_vloops = _collect_vertical_loops(routine.body, vertical_index)
    loop_nodes = _collect_loop_node_set(all_vloops)

    # Find whole-array zero-init assignments outside loops
    # Pattern: LHS has all-range-index dimensions, RHS is 0 or 0.0
    zero_init_map = {}  # arr_name_lower -> Assignment node
    _find_zero_inits_outside_loops(
        routine.body, loop_nodes, klev_locals, zero_init_map
    )

    if not zero_init_map:
        return []

    # For each candidate, check if removing the init eliminates ALL
    # outside-loop references
    removable = []
    for arr_name, init_node in zero_init_map.items():
        # Collect all outside-loop refs to this array, excluding the init
        outside_refs = _collect_outside_refs_for_array(
            routine.body, loop_nodes, arr_name, exclude_node=init_node
        )
        if not outside_refs:
            removable.append((arr_name, init_node))

    if not removable:
        return []

    # Remove the init assignments
    node_map = {node: None for _, node in removable}
    routine.body = Transformer(node_map).visit(routine.body)

    removed = [name for name, _ in removable]
    return removed


def _extract_plus_n(upper_expr, vertical_size):
    """
    Extract the integer *N* from an expression of the form ``KLEV + N``.

    Returns ``0`` for a plain ``KLEV`` expression.

    Parameters
    ----------
    upper_expr : expression or str
        The upper-bound expression (a Loki expression node) or, for
        backward compatibility, a normalised lowercase string.
    vertical_size : str or expression
        The vertical size (e.g. ``'klev'``).
    """
    # --- Expression-level path ---
    if not isinstance(upper_expr, str) and upper_expr is not None:
        vs_name = (vertical_size.lower() if isinstance(vertical_size, str)
                   else str(vertical_size).strip().lower())
        vs_sym = sym.Variable(name=vs_name)
        try:
            diff = simplify(upper_expr - vs_sym)
        except (TypeError, AttributeError):
            diff = None
        if isinstance(diff, sym.IntLiteral):
            return diff.value
        if isinstance(diff, int):
            return diff
        # Check if it's exactly KLEV (diff == 0 symbolically)
        if diff is not None and diff == 0:
            return 0
        # Expression-level check didn't resolve; fall through to string

    # --- String fallback ---
    upper_str = (str(upper_expr).strip().lower()
                 if not isinstance(upper_expr, str) else upper_expr)
    vs = (vertical_size.lower() if isinstance(vertical_size, str)
          else str(vertical_size).strip().lower())
    us = upper_str.replace(' ', '')
    if us.startswith(vs + '+'):
        try:
            return int(us[len(vs) + 1:])
        except ValueError:
            return 0
    return 0


def _build_bounds_guard(loop_var, lower_expr, upper_expr):
    """
    Build ``JK >= lower .AND. JK <= upper`` as a Loki expression.
    """
    ge_cond = sym.Comparison(
        operator='>=', left=loop_var, right=lower_expr
    )
    le_cond = sym.Comparison(
        operator='<=', left=loop_var, right=upper_expr
    )
    return sym.LogicalAnd((ge_cond, le_cond))


def _relocate_interloop_code(body, top_nodes, merged_loop):
    """
    Rebuild *body* so that:

    * The first vertical-loop top-node is replaced by the *merged_loop*.
    * All subsequent vertical-loop top-nodes are removed.
    * Any non-loop statements that sit **between** consecutive vertical-
      loop top-nodes are moved to just **before** the merged loop.

    The function uses a :any:`Visitor` to recurse into transparent
    container nodes (:any:`Section`, :any:`Associate`) until it finds
    the level that directly contains the *top_nodes*.

    Parameters
    ----------
    body : :any:`Section`, :any:`Associate`, or tuple of nodes
    top_nodes : set
        Set of IR nodes (Loop or Conditional wrappers) that correspond
        to the original vertical loops.
    merged_loop : :any:`Loop`
        The single merged loop to substitute.

    Returns
    -------
    Rebuilt *body* with the same type as the input.
    """

    class _Relocator(Visitor):
        def __init__(self):
            super().__init__()
            self.top_nodes = top_nodes
            self.merged_loop = merged_loop

        def visit_object(self, o, **kwargs):
            return o

        def visit_Node(self, o, **kwargs):
            return o  # leaf nodes pass through unchanged

        def visit_tuple(self, o, **kwargs):
            has_top = any(n in self.top_nodes for n in o)
            if has_top:
                # Find first/last top-node indices
                first_idx = last_idx = None
                for i, n in enumerate(o):
                    if n in self.top_nodes:
                        if first_idx is None:
                            first_idx = i
                        last_idx = i
                pre = o[:first_idx]
                post = o[last_idx + 1:]
                # Collect inter-loop statements (non-top-nodes in the
                # loop region) and place them before the merged loop
                interloop = tuple(
                    n for i, n in enumerate(o)
                    if first_idx <= i <= last_idx and n not in self.top_nodes
                )
                return pre + interloop + (self.merged_loop,) + post
            # Recurse into children to find the level with top_nodes
            return tuple(self.visit(n, **kwargs) for n in o)

        visit_list = visit_tuple

        def visit_Section(self, o, **kwargs):
            new_body = self.visit(o.body, **kwargs)
            if new_body is o.body:
                return o
            return o.clone(body=new_body)

        def visit_Associate(self, o, **kwargs):
            new_body = self.visit(o.body, **kwargs)
            if new_body is o.body:
                return o
            return o.clone(body=new_body)

    return _Relocator().visit(body)


def _merge_vertical_loops(routine, vertical_index, vertical_size):
    """
    Merge all vertical loops into a single ``DO JK = 1, <max_upper>``
    loop with IF guards for each original loop's bounds.

    Each loop body is wrapped in::

        IF (JK >= lower .AND. JK <= upper) THEN
            body
        END IF

    Conditional-wrapped loops get nested guards::

        IF (condition) THEN
            IF (JK >= lower .AND. JK <= upper) THEN
                body
            END IF
        END IF

    Parameters
    ----------
    routine : :any:`Subroutine`
    vertical_index : str
    vertical_size : str

    Returns
    -------
    :any:`Loop` or None
        The merged loop, or ``None`` if no vertical loops were found.
    """
    all_vloops = _collect_vertical_loops(routine.body, vertical_index)
    if not all_vloops:
        return None

    # Determine the merged upper bound: the maximum across all loops
    vertical_size_lower = vertical_size.lower()
    max_upper = None

    # TODO: this should generate a MAX(upper_bound1, upper_bound2, ...) in case there is
    #  any doubt about which is larger
    for loop, _cond in all_vloops:
        upper_expr = _loop_upper_bound_expr(loop)
        if upper_expr is None:
            continue

        if max_upper is None:
            max_upper = upper_expr
        elif _is_klev_plus_n(upper_expr, vertical_size_lower):
            # This is KLEV+N — use it if larger than current max
            if not _is_klev_plus_n(max_upper, vertical_size_lower):
                max_upper = upper_expr
            else:
                # Both are KLEV+N — compare N values
                n_current = _extract_plus_n(max_upper, vertical_size_lower)
                n_new = _extract_plus_n(upper_expr, vertical_size_lower)
                if n_new > n_current:
                    max_upper = upper_expr
        elif upper_expr == vertical_size_lower:
            if max_upper != vertical_size_lower and \
               not _is_klev_plus_n(max_upper, vertical_size_lower):
                max_upper = upper_expr

    # Use the first loop's variable for the merged loop
    loop_var = all_vloops[0][0].variable

    # Build the merged loop body: guarded bodies in source order
    merged_body = []

    for loop, cond_wrapper in all_vloops:
        # TODO: If the same Conditional wrapper contains vertical loops
        # in both its IF and ELSE branches, only the first-encountered
        # branch's loop is correctly merged.  The second loop (from the
        # other branch) may be lost because _collect_vertical_loops
        # returns the same wrapper reference for both, and the
        # replacement logic only operates on cond_wrapper.body.  This
        # edge case does not occur in the IFS/cloudsc kernels.
        body = loop.body
        lower_expr, upper_expr = _loop_effective_bounds(loop)

        # Build IF guard condition: JK >= lower .AND. JK <= upper
        guard_cond = _build_bounds_guard(loop_var, lower_expr, upper_expr)

        guarded = ir.Conditional(
            condition=guard_cond,
            body=(ir.Section(body=as_tuple(body)),),
            else_body=()
        )

        if cond_wrapper is not None:
            # Nest inside the original conditional, preserving its
            # else_body if present.  If the wrapper has a non-trivial
            # else branch we keep it intact and only replace the loop
            # in the body branch with the guarded version.
            if cond_wrapper.else_body:
                # Preserve the original conditional structure: replace
                # the loop inside the body branch, keep else_body.
                new_body = tuple(
                    guarded if stmt is loop else stmt
                    for stmt in cond_wrapper.body
                )
                nested = cond_wrapper.clone(body=new_body)
            else:
                nested = ir.Conditional(
                    condition=cond_wrapper.condition,
                    body=(guarded,),
                    else_body=()
                )
            merged_body.append(nested)
        else:
            merged_body.append(guarded)

    # Build the merged loop
    one = sym.IntLiteral(1)
    merged_bounds = sym.LoopRange((one, max_upper))
    merged_loop = ir.Loop(
        variable=loop_var.clone(),
        bounds=merged_bounds,
        body=tuple(merged_body)
    )

    # Build the set of top-level IR nodes for all vertical loops
    top_nodes = set()
    for loop, cond_wrapper in all_vloops:
        top_nodes.add(cond_wrapper if cond_wrapper is not None else loop)

    # Relocate inter-loop code: move statements that sit between
    # consecutive vertical loops to just before the merged loop.
    # This prevents scalar resets (e.g. ``result = 0.0``) from ending
    # up after the merged loop when the loops they were sandwiched
    # between are merged together.
    routine.body = _relocate_interloop_code(
        routine.body, top_nodes, merged_loop
    )

    n_loops = len(all_vloops)
    info('[vertical_utils] Merged %d vertical loops into 1 '
         '(DO JK = 1, %s)', n_loops, max_upper)

    return merged_loop


def _collect_rotates_from_node(node, rotate_keys, save_names,  # pylint: disable=unused-argument
                               enclosing_guard_cond, remove_map, hoisted):
    """
    Recursively walk an IR node tree to find carry rotate/save
    assignments and collect them for hoisting.

    Parameters
    ----------
    node : IR node
    rotate_keys : set of (vc_lower, next_lower) tuples
    save_names : set of vc_lower names
    enclosing_guard_cond : expression or None
        The IF condition of the enclosing guard block.
    remove_map : dict
        Updated in-place: ``{assignment: None}`` for nodes to remove.
    hoisted : list
        Updated in-place: ``(guard_condition, assignment)`` pairs.
    """

    class _RotateCollector(Visitor):
        def visit_tuple(self, o, **kwargs):
            for c in o:
                self.visit(c, **kwargs)

        visit_list = visit_tuple

        def visit_object(self, o, **kwargs):
            pass

        def visit_Node(self, o, **kwargs):
            pass

        def visit_Section(self, o, **kwargs):
            self.visit(o.body, **kwargs)

        def visit_Conditional(self, o, **kwargs):
            # Recurse into body with this conditional's guard
            guard_cond = o.condition
            self.visit(o.body, guard_cond=guard_cond)
            # Also check nested conditionals in else_body
            self.visit(o.else_body, **kwargs)

        def visit_Assignment(self, o, guard_cond=None, **kwargs):  # pylint: disable=unused-argument
            lhs_name = o.lhs.name.lower() if hasattr(o.lhs, 'name') else ''
            rhs_name = o.rhs.name.lower() if hasattr(o.rhs, 'name') else ''

            # Check for B_readback rotate: _vc = _next
            if (lhs_name, rhs_name) in rotate_keys:
                remove_map[o] = None
                hoisted.append((guard_cond, o))

            # Check for Pattern A / stencil save: the save is typically the
            # LAST assignment to _vc in the loop body.  We identify it by
            # LHS being a carry variable AND the RHS NOT being a _next
            # variable and NOT being part of the computation (i.e., the
            # RHS is a simple variable, not a complex expression with _vc).
            # Actually, we should NOT hoist Pattern A saves because they
            # set _vc to the CURRENT level value, which is needed by
            # cross-loop reads at offset 0 within the same iteration.
            # Only hoist B_readback rotates.

    _RotateCollector().visit(node, guard_cond=enclosing_guard_cond)


def _hoist_rotates_to_end(routine, merged_loop, carry_registry):
    """
    Move carry rotate and save statements to the very end of the
    merged loop body, so that all guarded blocks within an iteration
    see the pre-rotate carry values.

    After merging, guarded blocks from different original loops appear
    in source order.  A carry rotate like ``arr_vc = arr_next``
    that was at the end of one original loop's body now sits inside a
    guarded block that may precede another guarded block reading
    ``arr_vc``.  If the reader block expects the value at offset 0
    (current JK), but the rotate has already advanced it, the reader
    gets the wrong value.

    This function:

    1. Identifies rotate/save assignments by matching carry_registry
       entries (``_vc = _next`` for B_readback, ``_vc = <expr>`` for
       Pattern A and stencil).
    2. Removes them from their current position inside guarded blocks.
    3. Re-appends them at the end of the merged loop body, inside the
       same IF guard that surrounded them.

    Only B_readback rotates (``_vc = _next``) and stencil/Pattern-A
    saves are moved.

    Parameters
    ----------
    routine : :any:`Subroutine`
    merged_loop : :any:`Loop`
        The single merged vertical loop (modified in-place via
        Transformer).
    carry_registry : dict
        ``{array_name_lower: {'carry': str, 'pattern': str,
        'next': str or None, 'dim_index': int}}``

    Returns
    -------
    int
        Number of statements hoisted.
    """
    if not carry_registry:
        return 0

    # Build a set of (carry_name_lower, next_name_lower) pairs for
    # B_readback rotates, and a set of carry_name_lower for all saves.
    rotate_keys = set()   # (vc_name_lower, next_name_lower) for B_readback
    save_names = set()    # vc_name_lower for Pattern A and stencil saves
    for _arr, entry in carry_registry.items():
        vc_lower = entry['carry'].lower()
        if entry['pattern'] == 'B_readback' and entry.get('next'):
            rotate_keys.add((vc_lower, entry['next'].lower()))
        # For Pattern A and stencil, the save is _vc = <some_expression>
        # where the expression is NOT a _next variable.
        # We identify saves by the LHS being a carry variable.
        save_names.add(vc_lower)

    # Walk the merged loop body to find rotate/save assignments and
    # their enclosing IF guards.  We only look at the top-level
    # children of the merged loop body (which are Conditional guards).
    remove_map = {}     # assignment node → None (remove)
    hoisted = []        # (guard_condition, assignment) pairs

    for top_node in merged_loop.body:
        _collect_rotates_from_node(
            top_node, rotate_keys, save_names,
            None, remove_map, hoisted
        )

    if not remove_map:
        return 0

    # Remove the rotate/save statements from their original positions
    # and build the new loop with them appended at the end.
    new_loop = Transformer(remove_map).visit(merged_loop)

    # Build new guarded blocks at the end of the loop body for the
    # hoisted statements.  Group by guard condition string to combine
    # rotates under the same guard.
    guard_groups = OrderedDict()
    for guard_cond, assign in hoisted:
        # str() is intentional: used as a hashable grouping key for guard conditions
        key = str(guard_cond) if guard_cond is not None else '__no_guard__'
        if key not in guard_groups:
            guard_groups[key] = (guard_cond, [])
        guard_groups[key][1].append(assign)

    appended = []
    for key, (guard_cond, assigns) in guard_groups.items():
        if guard_cond is not None:
            block = ir.Conditional(
                condition=guard_cond,
                body=tuple(assigns),
                else_body=()
            )
            appended.append(block)
        else:
            appended.extend(assigns)

    # Append the hoisted blocks at the end of the merged loop body
    new_body = tuple(list(new_loop.body) + appended)
    final_loop = new_loop.clone(body=new_body)

    # Replace the ORIGINAL merged_loop in the routine body with the
    # final loop that has rotates removed from guarded blocks and
    # appended at the end.
    routine.body = Transformer({merged_loop: final_loop}).visit(routine.body)

    return len(remove_map)


def _cross_loop_carry_substitution(routine, merged_loop, carry_registry):
    """
    Replace remaining raw array references in the merged loop body with
    the carry variables created in Phase 1c.

    After per-loop carry conversion (Phase 1c) and merge (Phase 2),
    some loops that were *not* the source of a carry conversion may
    still reference the original array at offsets that are served by
    a carry variable from another loop.  This function resolves those
    cross-loop references.

    Substitution rules by offset:

    * **offset 0** → ``<array>_vc``  (all patterns except ``A`` and ``stencil``)
    * **offset +1** → ``<array>_next``  (B_readback only)
    * **offset -1** → ``<array>_vc``  (Pattern A / stencil)

    Parameters
    ----------
    routine : :any:`Subroutine`
    merged_loop : :any:`Loop`
        The single merged vertical loop.
    carry_registry : dict
        ``{array_name_lower: {'carry': str, 'pattern': str,
        'next': str or None, 'dim_index': int}}``

    Returns
    -------
    int
        Number of individual expression substitutions performed.
    """
    if not carry_registry:
        return 0

    loop_var = merged_loop.variable
    var_map = routine.variable_map

    expr_map = {}
    all_vars = FindVariables(unique=False).visit(merged_loop.body)

    for v in all_vars:
        if not isinstance(v, sym.Array) or not v.dimensions:
            continue

        arr_lower = v.name.lower()
        if arr_lower not in carry_registry:
            continue

        entry = carry_registry[arr_lower]
        dim_idx = entry['dim_index']
        pattern = entry['pattern']
        carry_name = entry['carry']
        next_name = entry.get('next')

        if dim_idx >= len(v.dimensions):
            continue

        offset = extract_offset(v.dimensions[dim_idx], loop_var)
        if offset is None:
            continue

        # Determine which carry variable to use for this offset
        target_name = None
        if offset == 0:
            # For stencil patterns, _vc holds JK-1 value, NOT JK.
            # Offset 0 must not be substituted for stencil arrays.
            #
            # For Pattern A, _vc also holds JK-1 value (before the
            # save statement at end of body).  After merging, a
            # *different* loop's init write (e.g. X(JK) = expr)
            # runs BEFORE the carrier loop's guarded block.  If
            # Phase 2b turned that init write into ``x_vc = expr``,
            # it would overwrite the JK-1 carry value.  Leaving
            # offset-0 as plain array refs lets Phase 3 demote
            # them to scalars independently of the carry.
            if pattern not in ('stencil', 'A'):
                target_name = carry_name
        elif offset == 1 and pattern == 'B_readback' and next_name:
            target_name = next_name
        elif offset == -1 and pattern in ('A', 'stencil'):
            target_name = carry_name
        else:
            # Offset not served by this carry pattern — leave as-is
            continue

        target_decl = var_map.get(target_name)
        if target_decl is None:
            continue

        # Build replacement: drop the vertical dimension
        new_dims = tuple(
            d for i, d in enumerate(v.dimensions) if i != dim_idx
        )
        expr_map[v] = target_decl.clone(
            dimensions=new_dims if new_dims else None
        )

    if not expr_map:
        return 0

    # Apply substitutions to the merged loop body
    new_body = SubstituteExpressions(expr_map).visit(merged_loop.body)
    new_loop = merged_loop.clone(body=new_body)
    routine.body = Transformer({merged_loop: new_loop}).visit(routine.body)

    n_subs = len(expr_map)
    return n_subs


def _insert_writebacks_for_argument_carries(routine, merged_loop,
                                             carry_registry,
                                             horizontal=None):
    """
    Insert write-back statements for INTENT(OUT) argument arrays that
    were converted to B_readback carry patterns.

    After carry conversion (Phase 1c) and cross-loop carry substitution
    (Phase 2b), the original output arrays (e.g. ``PFSQLF``) are never
    written to — all writes go to the ``_next`` carry variable.  This
    function inserts::

        ARRAY(JL, JK + 1) = array_next

    just before the rotate statement ``array_vc = array_next``, so that
    the output array is populated with the correct values.

    **Why this is done after Phase 2b, not in Phase 1c:**

    Phase 2b uses :func:`SubstituteExpressions` which replaces both LHS
    and RHS.  If write-backs were generated in Phase 1c, Phase 2b would
    replace ``PFSQLF(JL, JK+1)`` on the LHS with ``pfsqlf_next``,
    creating a self-assignment that Phase 4a removes.

    Parameters
    ----------
    routine : :any:`Subroutine`
    merged_loop : :any:`Loop`
        The single merged vertical loop.
    carry_registry : dict
        ``{array_name_lower: {'carry': str, 'pattern': str,
        'next': str or None, 'dim_index': int}}``
    horizontal : :any:`Dimension`, optional
        When provided, the horizontal dimension in write-back array
        subscripts uses bounded ranges from ``horizontal.bounds``
        instead of bare ``:``.

    Returns
    -------
    int
        Number of write-back statements inserted.
    """
    loop_var = merged_loop.variable
    var_map = CaseInsensitiveDict(
        {v.name: v for v in routine.variables}
    )
    arg_names = {v.name.lower() for v in routine.arguments}

    # Collect B_readback entries for argument arrays
    writebacks = []
    for arr_lower, entry in carry_registry.items():
        if entry['pattern'] != 'B_readback':
            continue
        if arr_lower not in arg_names:
            continue
        next_name = entry.get('next')
        if not next_name:
            continue

        orig_decl = var_map.get(arr_lower)
        next_decl = var_map.get(next_name)
        carry_decl = var_map.get(entry['carry'])
        if orig_decl is None or next_decl is None or carry_decl is None:
            continue

        dim_idx = entry['dim_index']
        orig_shape = orig_decl.type.shape if orig_decl.type.shape else ()

        # Build write-back: ARRAY(JL, JK+1) = array_next
        # For the vertical dimension, use JK+1.
        # For the horizontal dimension, use the horizontal index variable
        # (e.g. JL) on BOTH LHS and RHS to keep ranks consistent.
        # Using ':' (RangeIndex) on one side and a scalar on the other
        # would produce a rank mismatch in the generated Fortran.
        wb_orig_dims = []
        wb_next_dims = []
        nv_idx = 0
        for i, _s in enumerate(orig_shape):
            if i == dim_idx:
                wb_orig_dims.append(
                    sym.Sum((loop_var, sym.IntLiteral(1)))
                )
            else:
                next_shape = next_decl.type.shape if next_decl.type.shape else ()
                # Use a range for non-vertical dimensions in write-backs.
                # The write-back sits outside any horizontal (DO JL) loop,
                # so scalar JL would be out-of-scope.  For dimensions that
                # match the horizontal size, use bounded range (KIDIA:KFDIA)
                # so that resolve_vector_dimension can resolve it.
                # Other non-vertical dims use bare ':'.
                _is_horiz = (horizontal is not None
                             and _s == horizontal.size)
                if _is_horiz:
                    _lower = sym.Variable(name=horizontal.bounds[0], scope=None)
                    _upper = sym.Variable(name=horizontal.bounds[1], scope=None)
                    range_dim = sym.RangeIndex((_lower, _upper, None))
                else:
                    range_dim = sym.RangeIndex((None, None, None))
                wb_orig_dims.append(range_dim)
                if nv_idx < len(next_shape):
                    wb_next_dims.append(range_dim)
                nv_idx += 1

        wb_lhs = orig_decl.clone(dimensions=tuple(wb_orig_dims))
        wb_rhs = next_decl.clone(
            dimensions=tuple(wb_next_dims) if wb_next_dims else None
        )
        wb_stmt = ir.Assignment(lhs=wb_lhs, rhs=wb_rhs)

        # Find the rotate statement: carry_vc = carry_next
        # The write-back goes just before it
        writebacks.append((entry['carry'], next_name, wb_stmt))

    if not writebacks:
        return 0

    # Find rotate statements in the merged loop and insert write-backs
    # before them.  We walk the loop body to find assignments where
    # LHS name matches carry_vc and RHS name matches carry_next.
    carry_to_wb = {}
    for carry_name, next_name, wb_stmt in writebacks:
        carry_to_wb[(carry_name.lower(), next_name.lower())] = wb_stmt

    node_map = {}
    for assign in FindNodes(ir.Assignment).visit(merged_loop.body):
        lhs_name = assign.lhs.name.lower() if hasattr(assign.lhs, 'name') else ''
        rhs_name = assign.rhs.name.lower() if hasattr(assign.rhs, 'name') else ''
        key = (lhs_name, rhs_name)
        if key in carry_to_wb:
            wb = carry_to_wb.pop(key)
            # Insert write-back before rotate
            node_map[assign] = (wb, assign)

    if not node_map:
        return 0

    new_loop = Transformer(node_map).visit(merged_loop)
    routine.body = Transformer({merged_loop: new_loop}).visit(routine.body)

    return len(node_map)

def _remove_self_assignments(routine, merged_loop):
    """
    Remove self-assignment no-ops (``x = x``) from *merged_loop*.

    These arise when carry conversion (Phase 1c) generates a save
    statement ``x_vc(:) = X(:, JK)`` and a later phase (1d or 2b)
    substitutes the original array ``X`` with ``x_vc``, turning the
    save into ``x_vc(:) = x_vc(:)``.

    Parameters
    ----------
    routine : :any:`Subroutine`
    merged_loop : :any:`Loop` or list of :any:`Loop`
        The vertical loop(s) to clean up.  When a list is given every
        loop is processed in a single tree-rebuild pass.

    Returns
    -------
    int
        Number of self-assignments removed.
    """
    from loki.backend import fgen  # pylint: disable=import-outside-toplevel

    loops = merged_loop if isinstance(merged_loop, (list, tuple)) else [merged_loop]

    # Collect self-assignments across all target loops
    loop_map = {}   # old_loop -> new_loop (only for loops with removals)
    n_total = 0
    for loop in loops:
        to_remove = {}
        for assign in FindNodes(ir.Assignment).visit(loop.body):
            lhs_str = fgen(assign.lhs).strip().lower()
            rhs_str = fgen(assign.rhs).strip().lower()
            if lhs_str == rhs_str:
                to_remove[assign] = None
        if to_remove:
            new_body = Transformer(to_remove).visit(loop.body)
            loop_map[loop] = loop.clone(body=new_body)
            n_total += len(to_remove)

    if not loop_map:
        return 0

    routine.body = Transformer(loop_map).visit(routine.body)
    return n_total


def _remove_dead_carry_originals(routine, carry_registry, demoted_names):
    """
    Remove declarations of demoted local arrays that have zero
    remaining executable references after carry conversion.

    A variable is removed only if ALL conditions are met:

    1. It is in *carry_registry* (i.e. a carry was created for it).
    2. Its name (lowercase) is in *demoted_names*.
    3. It is **not** a subroutine argument.
    4. It has **zero** references in the routine body (executable code).

    Parameters
    ----------
    routine : :any:`Subroutine`
    carry_registry : dict
        ``{array_name_lower: {'carry': str, ...}}``
    demoted_names : list of str
        Lowercase names of arrays that were demoted in Phase 3.

    Returns
    -------
    list of str
        Lowercase names of variables removed.
    """
    arg_names = {v.name.lower() for v in routine.arguments}
    demoted_set = set(n.lower() for n in demoted_names)

    # Collect all variable references in the routine body
    all_refs = FindVariables(unique=False).visit(routine.body)
    ref_names = set(v.name.lower() for v in all_refs)

    removed = []
    for arr_lower in sorted(carry_registry):
        if arr_lower not in demoted_set:
            continue
        if arr_lower in arg_names:
            continue
        if arr_lower in ref_names:
            continue
        # No references — safe to remove the declaration
        removed.append(arr_lower)

    if removed:
        removed_set = set(removed)
        routine.variables = tuple(
            v for v in routine.variables
            if v.name.lower() not in removed_set
        )

    return removed
