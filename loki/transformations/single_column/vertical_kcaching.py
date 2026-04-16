# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
K-caching vertical-loop merge transformation for SCC pipelines.

:class:`SCCVerticalKCaching` fuses **all** vertical loops in a routine
into a single ``DO JK = 1, KLEV+1`` loop with IF guards, converts
multi-level array dependencies to scalar carry variables (two-scalar
``_vc``/``_next`` approach with explicit rotate), demotes KLEV-
dimensioned temporaries to scalars, and performs cross-loop carry
substitution — dramatically reducing GPU memory pressure.

The transformation operates in the following phases:

**Phase 1 — Preparation**
    1a. Loop interchange to expose vertical loops as outermost.
    1b. Dead-loop elimination.
    1c. Comprehensive carry conversion in *all* vertical loops.
    1d. Init-expression substitution for outside-loop reads.
    1e. Whole-array zero-init removal.

**Phase 2 — Merge**
    Merge all remaining vertical loops into a single
    ``DO JK = 1, <max_upper>`` loop.  Each original loop body is
    wrapped in ``IF (JK >= lower .AND. JK <= upper)`` guards.

**Phase 2b — Cross-loop carry substitution**
    Replace remaining raw array references in the merged loop body
    with carry variables created in Phase 1c.  This resolves forward
    read-only accesses that span multiple original loops and enables
    further demotion.

**Phase 3 — Demotion**
    Demote all local arrays whose vertical dimension became
    level-local after merging.

**Phase 4 — Cleanup**
    4a. Remove self-assignment no-ops.
    4b. Remove declarations of demoted local arrays that have zero
        remaining executable references.
"""

from loki.batch import Transformation
from loki.ir import nodes as ir, FindNodes, Transformer
from loki.logging import info, warning
from loki.transformations.transform_loop import do_loop_interchange
from loki.transformations.array_indexing import demote_variables

# Shared utilities for vertical loop transformations
from loki.transformations.single_column.vertical_utils import (
    _collect_vertical_loops,
    _find_demotable_arrays,
    _any_read_outside_node,
    _find_dead_loops_all,
    _convert_all_carries,
    _substitute_init_expressions_all_loops,
    _remove_whole_array_zero_inits,
    _merge_vertical_loops,
    _hoist_rotates_to_end,
    _cross_loop_carry_substitution,
    _insert_writebacks_for_argument_carries,
    _remove_self_assignments,
    _remove_dead_carry_originals,
    _is_zero_literal,
)

__all__ = ['SCCVerticalKCaching']


# ---------------------------------------------------------------------------
# Carry-aware dead-loop elimination
# ---------------------------------------------------------------------------

def _find_dead_loops_carry_aware(routine, vertical_index, carry_registry):
    """
    Identify vertical loops whose written outputs will be dead after
    cross-loop carry substitution in Phase 2b.

    This is a carry-aware variant of :func:`_find_dead_loops_all`.
    A loop is considered dead if **every** variable it writes satisfies
    at least one of:

    1. It is never read anywhere else in the routine (truly dead).
    2. It is in the *carry_registry*, meaning all remaining reads will
       be replaced by carry variables during Phase 2b cross-loop carry
       substitution.

    Without this, zeroing loops like ``X(JK,JM) = 0.0`` survive
    the merge and zero the carry variable on every iteration, destroying
    inter-level accumulation (e.g. precipitation sedimentation).

    Parameters
    ----------
    routine : :any:`Subroutine`
    vertical_index : str
    carry_registry : dict
        Mapping of lowercase array names to carry info dicts.

    Returns
    -------
    list
        Top-level IR nodes (Loop or Conditional wrappers) that are dead.
    """
    all_vloops = _collect_vertical_loops(routine.body, vertical_index)
    dead = []

    arg_names = {v.name.lower() for v in routine.arguments}
    carry_names = set(carry_registry.keys())  # already lowercase

    for loop, cond_wrapper in all_vloops:
        written_names = set()
        for assign in FindNodes(ir.Assignment).visit(loop.body):
            written_names.add(assign.lhs.name.lower())

        if not written_names:
            continue

        # Writing to a routine argument is never dead
        if written_names & arg_names:
            continue

        # Partition written names: carry-registered vs non-carry
        non_carry_written = written_names - carry_names

        exclude_node = cond_wrapper if cond_wrapper is not None else loop

        # For non-carry names, check if they are read outside
        if non_carry_written:
            if _any_read_outside_node(routine.body, exclude_node,
                                       non_carry_written):
                continue

        # For carry-registered names: they will be handled by Phase 2b
        # cross-loop carry substitution, so we treat them as "will be
        # dead" — but ONLY if the loop body is a pure zeroing/init loop.
        # We verify this by checking that every assignment in the loop
        # to a carry-registered array has a literal zero RHS.
        carry_written = written_names & carry_names
        if carry_written:
            is_pure_init = True
            for assign in FindNodes(ir.Assignment).visit(loop.body):
                arr_lower = assign.lhs.name.lower()
                if arr_lower in carry_names:
                    # Check if RHS is a literal zero (0, 0.0, 0.0_JPRB, etc.)
                    rhs = assign.rhs
                    if _is_zero_literal(rhs):
                        continue
                    # Not a pure zero init — keep the loop
                    is_pure_init = False
                    break
            if not is_pure_init:
                continue

        dead.append(exclude_node)

    return dead


def _auto_interchange_vertical_loops(routine, vertical_index):
    """
    Automatically interchange loop nests where a non-vertical loop
    (e.g. ``DO JM = 1, NCLV-1``) is the outermost loop and the
    vertical loop (``DO JK = 1, KLEV``) is immediately nested inside.

    After interchange the vertical loop becomes outermost, making it
    visible to ``_collect_vertical_loops`` and eligible for merging.

    The interchange is safe when the outer loop's body consists solely
    of the inner vertical loop (possibly preceded/followed only by
    pragmas or comments), because that means there is no code between
    the outer loop header and the inner loop that depends on the outer
    loop variable in a way that would be violated by reordering.

    Returns the number of interchanges performed.
    """
    vi_lower = vertical_index.lower()
    loop_map = {}

    # NOTE: FindNodes(ir.Loop) visits all loops (not only outermost), but
    # the filter below (exactly one inner vertical loop, no other executable
    # code) ensures only genuine two-level nests are interchanged.
    for outer_loop in FindNodes(ir.Loop).visit(routine.body):
        # Skip if this is already a vertical loop
        if outer_loop.variable.name.lower() == vi_lower:
            continue

        # Find the immediate inner loop(s) — skip pragmas/comments
        inner_loops = [n for n in outer_loop.body
                       if isinstance(n, ir.Loop)]
        non_loop_exec = [n for n in outer_loop.body
                         if not isinstance(n, (ir.Loop, ir.Comment,
                                               ir.Pragma))]

        # Only interchange if:
        # 1. There is exactly one inner loop
        # 2. That inner loop IS a vertical loop
        # 3. There is no executable code between outer and inner
        #    (only pragmas/comments are allowed)
        if (len(inner_loops) == 1 and
                inner_loops[0].variable.name.lower() == vi_lower and
                len(non_loop_exec) == 0):
            inner_loop = inner_loops[0]

            # Build interchanged nest: JK outer, JM inner
            # The inner loop body stays the same, just swap variables
            # and bounds between the two loop levels.
            # Collect any pragmas/comments from the outer loop body
            # (e.g., !DIR$ IVDEP) and attach them to the new inner loop.
            # TODO: Pragmas/comments between the outer loop header and
            # the inner loop are currently placed inside the new inner
            # loop body.  For directive-style pragmas that annotate the
            # *following* loop, this may change which loop they apply to
            # after interchange.  Consider keeping them as siblings
            # before the new inner loop in the outer loop body instead.
            non_loop_items = tuple(
                n for n in outer_loop.body
                if isinstance(n, (ir.Comment, ir.Pragma))
            )

            new_inner = inner_loop.clone(
                variable=outer_loop.variable,
                bounds=outer_loop.bounds,
                pragma=inner_loop.pragma,
                pragma_post=inner_loop.pragma_post,
                body=non_loop_items + inner_loop.body,
            )
            new_outer = outer_loop.clone(
                variable=inner_loop.variable,
                bounds=inner_loop.bounds,
                body=(new_inner,),
                pragma=outer_loop.pragma,
                pragma_post=outer_loop.pragma_post,
            )
            loop_map[outer_loop] = new_outer

    # TODO: Consider extracting loop interchange into a reusable utility
    # (e.g. ``do_loop_interchange``) if other transformations need it.
    if loop_map:
        routine.body = Transformer(loop_map).visit(routine.body)
    return len(loop_map)


class SCCVerticalKCaching(Transformation):
    """
    Full k-caching vertical-loop merge transformation.

    Merges **all** vertical loops in a routine into a single
    vertical loop with IF guards, converts array
    dependencies to scalar carry variables (``_vc``/``_next``),
    performs cross-loop carry substitution, and demotes all eligible
    local arrays.

    This is designed as a general-purpose transformation that works
    on any kernel with proper-dimensioned vertical loops.

    Parameters
    ----------
    horizontal : :any:`Dimension`, optional
        Dimension object describing the horizontal (index, size, bounds).
    vertical : :any:`Dimension`
        Dimension object describing the vertical (index, size, bounds).
    apply_to : list of str, optional
        Routine names to apply to (lowercase).  If not provided,
        apply to all kernel routines.
    """

    def __init__(self, horizontal=None, vertical=None, apply_to=None):
        self.horizontal = horizontal
        self.vertical = vertical
        self.apply_to = apply_to or ()

        if self.vertical is None:
            info('[SCCVerticalKCaching] is not applied as the '
                 'vertical dimension is not defined!')

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def transform_subroutine(self, routine, **kwargs):
        """Dispatch to :meth:`process_kernel` for kernel-role routines."""
        if self.vertical is None:
            return
        role = kwargs.get('role', 'kernel')
        if role == 'kernel':
            if self.apply_to and routine.name.lower() not in \
                    [n.lower() for n in self.apply_to]:
                return
            self.process_kernel(routine)

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def process_kernel(self, routine):
        """
        Orchestrate the k-caching transformation.

        Phases 1-4 as described in the module docstring.
        """
        # NOTE: Only the primary index/size names are used here.
        # Dimension.index_aliases and Dimension.size_aliases are not
        # consulted, so routines using alias names will not be matched.
        vertical_index = self.vertical.index
        vertical_size = str(self.vertical.size)
        horizontal_index = (
            self.horizontal.index if self.horizontal is not None else None
        )

        # Phase 1a: Loop interchange (pragma-based + automatic)
        do_loop_interchange(routine)
        n_auto = _auto_interchange_vertical_loops(routine, vertical_index)
        if n_auto:
            info('[SCCVerticalKCaching] %s: Phase 1a - auto-interchanged '
                 '%d loop nest(s) to expose vertical loops', routine.name,
                 n_auto)

        # Phase 1b: Dead-loop elimination
        dead_loops = _find_dead_loops_all(routine, vertical_index)
        if dead_loops:
            node_map = {node: None for node in dead_loops}
            routine.body = Transformer(node_map).visit(routine.body)
            info('[SCCVerticalKCaching] %s: Phase 1b - removed '
                 '%d dead loop(s)', routine.name, len(dead_loops))

        # Phase 1c: Carry conversion in ALL vertical loops
        all_vloops = _collect_vertical_loops(routine.body, vertical_index)
        all_carry_init_stmts = []
        total_conversions = 0
        carry_registry = {}

        for loop, _cond in all_vloops:
            new_loop, init_stmts, conversions = _convert_all_carries(
                routine, loop, vertical_size
            )
            if conversions:
                routine.body = Transformer(
                    {loop: new_loop}).visit(routine.body)
                all_carry_init_stmts.extend(init_stmts)
                total_conversions += len(conversions)
                for conv in conversions:
                    info('[SCCVerticalKCaching] %s:   %s -> %s '
                         '(pattern %s)', routine.name, conv['array'],
                         conv['carry'], conv['pattern'])
                    arr_lower = conv['array'].lower()
                    if arr_lower not in carry_registry:
                        carry_registry[arr_lower] = {
                            'carry': conv['carry'],
                            'pattern': conv['pattern'],
                            'next': conv.get('next'),
                            'dim_index': conv['dim_index'],
                        }

        if total_conversions:
            info('[SCCVerticalKCaching] %s: Phase 1c - converted '
                 '%d carries', routine.name, total_conversions)

        # Place carry init statements before the first vertical loop
        if all_carry_init_stmts:
            all_vloops = _collect_vertical_loops(
                routine.body, vertical_index)
            if all_vloops:
                first_loop = all_vloops[0][0]
                first_node = (all_vloops[0][1]
                              if all_vloops[0][1] is not None
                              else first_loop)
                init_section = ir.Section(
                    body=tuple(all_carry_init_stmts))
                routine.body = Transformer(
                    {first_node: (init_section, first_node)}
                ).visit(routine.body)

        # Phase 1c-post: Carry-aware dead-loop elimination.
        # After carry conversion, some zeroing loops (e.g. ``arr = 0``)
        # that were NOT dead before (because other loops still read the
        # original array) become dead because Phase 2b cross-loop carry
        # substitution will replace all remaining reads with carry
        # variables.  We detect these proactively using the carry
        # registry so they don't get merged and zero the carry variable
        # on every JK iteration.
        if carry_registry:
            dead_loops_post = _find_dead_loops_carry_aware(
                routine, vertical_index, carry_registry
            )
            if dead_loops_post:
                node_map = {node: None for node in dead_loops_post}
                routine.body = Transformer(node_map).visit(routine.body)
                info('[SCCVerticalKCaching] %s: Phase 1c-post - removed '
                     '%d dead loop(s) after carry conversion (carry-aware)',
                     routine.name, len(dead_loops_post))

        # Phase 1d: Init-expression substitution
        substituted = _substitute_init_expressions_all_loops(
            routine, vertical_index, vertical_size
        )
        if substituted:
            info('[SCCVerticalKCaching] %s: Phase 1d - substituted '
                 'init expressions for %s', routine.name,
                 ', '.join(sorted(substituted)))

        # Phase 1e: Remove whole-array zero-inits
        removed = _remove_whole_array_zero_inits(
            routine, vertical_index, vertical_size
        )
        if removed:
            info('[SCCVerticalKCaching] %s: Phase 1e - removed '
                 'zero-inits for %s', routine.name,
                 ', '.join(sorted(removed)))

        # Phase 2: Merge all vertical loops
        merged = _merge_vertical_loops(
            routine, vertical_index, vertical_size
        )
        if merged is None:
            warning('[SCCVerticalKCaching] %s: no vertical loops to '
                    'merge', routine.name)
            return

        # Phase 2 post-merge: Hoist carry rotates to end of merged loop
        if carry_registry:
            all_vloops = _collect_vertical_loops(
                routine.body, vertical_index)
            if all_vloops:
                merged_loop = all_vloops[0][0]
                n_hoisted = _hoist_rotates_to_end(
                    routine, merged_loop, carry_registry
                )
                if n_hoisted:
                    info('[SCCVerticalKCaching] %s: Phase 2 post-merge '
                         '- hoisted %d rotate/save(s)', routine.name,
                         n_hoisted)

        # Phase 2b: Cross-loop carry substitution (always enabled)
        if carry_registry:
            all_vloops = _collect_vertical_loops(
                routine.body, vertical_index)
            if all_vloops:
                merged_loop = all_vloops[0][0]
                n_subs = _cross_loop_carry_substitution(
                    routine, merged_loop, carry_registry
                )
                if n_subs:
                    info('[SCCVerticalKCaching] %s: Phase 2b - '
                         'cross-loop carry substitution: %d replacement(s)',
                         routine.name, n_subs)

        # Phase 2c: Insert write-back statements for argument arrays
        if carry_registry:
            all_vloops = _collect_vertical_loops(
                routine.body, vertical_index)
            if all_vloops:
                merged_loop = all_vloops[0][0]
                n_wb = _insert_writebacks_for_argument_carries(
                    routine, merged_loop, carry_registry,
                    horizontal_index=horizontal_index
                )
                if n_wb:
                    info('[SCCVerticalKCaching] %s: Phase 2c - '
                         'inserted %d write-back(s) for argument arrays',
                         routine.name, n_wb)

        # Phase 3: Demotion
        demotable = []
        all_vloops = _collect_vertical_loops(routine.body, vertical_index)
        if all_vloops:
            merged_loop = all_vloops[0][0]
            demotable = _find_demotable_arrays(
                routine, vertical_index, vertical_size
            )
            if demotable:
                dims_to_demote = (
                    self.vertical.size_expressions +
                    (f"{self.vertical.size}+1",))
                demote_variables(routine, demotable, dims_to_demote)
                info('[SCCVerticalKCaching] %s: Phase 3 - demoted '
                     '%d array(s): %s', routine.name, len(demotable),
                     ', '.join(demotable))

        # Phase 4: Cleanup
        all_vloops = _collect_vertical_loops(routine.body, vertical_index)
        if all_vloops:
            merged_loop = all_vloops[0][0]

            # 4a. Remove self-assignment no-ops
            n_noops = _remove_self_assignments(routine, merged_loop)
            if n_noops:
                info('[SCCVerticalKCaching] %s: Phase 4a - removed '
                     '%d self-assignment no-op(s)', routine.name, n_noops)

        # 4b. Remove dead carry-original declarations
        if carry_registry and demotable:
            dead = _remove_dead_carry_originals(
                routine, carry_registry, demotable
            )
            if dead:
                info('[SCCVerticalKCaching] %s: Phase 4b - removed '
                     '%d dead variable(s): %s', routine.name,
                     len(dead), ', '.join(dead))
