# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to split horizontal loops that contain a mix of
vectorisable and non-vectorisable statements into multiple loops so
that the vectorisable portions can receive SIMD pragma annotations.

After the upstream transformations in the CPU vectorisation pipeline
(inlining, I/O hoisting, safe-denominator guards, MERGE conversion),
some horizontal loops still contain non-vectorisable nodes such as
subroutine calls, intrinsic I/O statements, or nested loops.  These
prevent ``InsertSIMDPragmaDirectives`` from annotating the loop with
``!$OMP SIMD``.

This transformation analyses each horizontal loop body and partitions
its top-level nodes into contiguous *vectorisable* and
*non-vectorisable* sections.  Each section becomes its own horizontal
loop with the same iteration bounds.  Scalar variables that are
written in one resulting loop and read in a subsequent one are
automatically promoted to arrays with the horizontal dimension so
that per-iteration values are preserved.

Assignments and conditionals that access arrays via *indirect
indexing* on the horizontal dimension (e.g. ``A(IDX(JL))`` where
``IDX`` is itself an array) are classified as non-vectorisable
because the resulting gather/scatter memory-access pattern prevents
effective SIMD execution.  The horizontal subscript position is
determined from the array's declared shape; if the shape is not
available, position 0 is assumed (consistent with the IFS convention
that the horizontal dimension is the leading dimension).

When *fp_strict* mode is enabled, assignments and conditionals that
contain calls to certain math intrinsics (``EXP``, ``LOG``,
``SQRT``, etc.) are additionally classified as non-vectorisable.
Under conservative floating-point compiler flags these intrinsics
may prevent the compiler from auto-vectorising the loop, so
splitting them out allows the remaining pure-arithmetic statements
to still benefit from SIMD annotations.

By convention there are no loop-carried dependencies for the
horizontal dimension, so the fission is always semantically safe.

**Numerical impact**: None -- the split is semantically equivalent.
"""

from loki.batch import Transformation
from loki.expression import symbols as sym, Array
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, FindInlineCalls, Transformer,
)
from loki.logging import info

from loki.transformations.array_indexing import promote_variables
from loki.transformations.utilities import (
    check_routine_sequential, get_local_variables, get_integer_variable,
)


__all__ = ['SplitLoopForVectorisation']


# Node types that are safe for SIMD execution (vectorisable).
_VECTORISABLE_TYPES = (ir.Assignment, ir.Conditional)

# Node types that carry no semantic weight and can be attached to any
# section without affecting vectorisability.
_NEUTRAL_TYPES = (ir.Comment, ir.Pragma, ir.CommentBlock)

# Default set of math intrinsics that are considered non-vectorisable
# under conservative (strict) floating-point compiler flags.  Matches
# the ``DANGEROUS_INTRINSICS`` set in :any:`SafeDenominatorGuard`.
_DEFAULT_FP_STRICT_INTRINSICS = frozenset({'EXP', 'LOG', 'SQRT'})


class SplitLoopForVectorisation(Transformation):
    """
    Split horizontal loops that contain a mix of vectorisable and
    non-vectorisable statements into separate loops.

    After splitting, the vectorisable loops contain only assignments
    and conditionals (the same criterion used by
    :any:`InsertSIMDPragmaDirectives`) and can therefore receive SIMD
    pragma annotations.

    Statements that use indirect (gather/scatter) array indexing on
    the horizontal dimension (e.g. ``A(IDX(JL))``) are always
    classified as non-vectorisable because they prevent effective SIMD
    execution.

    Local scalar variables whose values cross a split boundary are
    promoted to arrays with the horizontal dimension so that
    per-iteration values are preserved.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the horizontal data
        dimension and iteration space.
    min_num_assignments : int, optional
        Minimum number of :any:`Assignment` nodes required in a
        vectorisable section for it to justify its own loop.  Sections
        below this threshold are merged with an adjacent
        non-vectorisable section.  Default ``1``.
    fp_strict : bool, optional
        When ``True``, assignments and conditionals whose expression
        trees contain calls to intrinsics listed in
        *fp_strict_intrinsics* are classified as non-vectorisable.
        This is useful when compiling with conservative floating-point
        flags (e.g. ``-fp-model strict``) where intrinsics such as
        ``EXP``, ``LOG``, and ``SQRT`` prevent the compiler from
        auto-vectorising a loop.  Default ``False``.
    fp_strict_intrinsics : set of str or None, optional
        Upper-cased names of intrinsic functions to treat as
        non-vectorisable when *fp_strict* is enabled.  If ``None``
        (the default), the built-in set ``{'EXP', 'LOG', 'SQRT'}``
        is used.  A custom set can be passed to extend or narrow
        the list (e.g. ``{'EXP', 'LOG', 'SQRT', 'SIN', 'COS'}``).
    """

    def __init__(self, horizontal, min_num_assignments=1,
                 fp_strict=False, fp_strict_intrinsics=None):
        self.horizontal = horizontal
        self.min_num_assignments = min_num_assignments
        self.fp_strict = fp_strict
        self.fp_strict_intrinsics = (
            frozenset(n.upper() for n in fp_strict_intrinsics)
            if fp_strict_intrinsics is not None
            else _DEFAULT_FP_STRICT_INTRINSICS
        )

    # -----------------------------------------------------------------
    # Transformation entry point
    # -----------------------------------------------------------------

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply loop splitting to a single :any:`Subroutine`.

        Skips driver-role routines and routines marked sequential.
        """
        role = kwargs.get('role', 'kernel')
        if role == 'driver':
            return
        if check_routine_sequential(routine):
            return

        self._split_horizontal_loops(routine)

    # -----------------------------------------------------------------
    # Core algorithm
    # -----------------------------------------------------------------

    def _split_horizontal_loops(self, routine):
        """
        Walk *routine*, find horizontal loops whose body mixes
        vectorisable and non-vectorisable statements, and split them.
        """
        # Collect all variables to promote across all split loops so
        # that a single call to promote_variables covers everything.
        vars_to_promote = set()
        loop_map = {}

        # Collect ALL loop indices from ALL horizontal loops (outer +
        # nested) in the routine.  A variable that is used as a loop
        # index in ANY horizontal loop must never be promoted, even if
        # it is used as a plain scalar in another horizontal loop.
        # Promoting it would turn ``DO JO=1,N`` into ``DO JO(JL)=1,N``
        # which is nonsensical Fortran.
        all_loop_indices = set()
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if not self._is_horizontal_loop(loop):
                continue
            all_loop_indices.add(loop.variable.name.lower())
            for nested_loop in FindNodes(ir.Loop).visit(loop.body):
                all_loop_indices.add(nested_loop.variable.name.lower())

        for loop in FindNodes(ir.Loop).visit(routine.body):
            if not self._is_horizontal_loop(loop):
                continue

            sections = self._partition_loop_body(loop)

            # Nothing to do if the body is homogeneous (single section
            # or all sections of the same kind).
            has_vec = any(kind == 'vec' for kind, _ in sections)
            has_nonvec = any(kind == 'non' for kind, _ in sections)
            if not (has_vec and has_nonvec):
                continue

            # Identify scalars that need promotion across splits.
            promote_names = self._find_scalars_to_promote(
                sections, loop, routine
            )
            vars_to_promote |= promote_names

            # Build replacement: a sequence of new loops, one per
            # section.
            new_nodes = self._build_split_loops(loop, sections)
            loop_map[loop] = tuple(new_nodes)

        if not loop_map:
            return

        # Remove any loop indices from the promotion set — a variable
        # used as a loop index anywhere in the routine must not be
        # promoted to an array.
        vars_to_promote -= all_loop_indices

        # Apply the loop replacement in the IR.
        routine.body = Transformer(loop_map).visit(routine.body)

        # Promote scalars that cross split boundaries.
        if vars_to_promote:
            hor_index, hor_size = self._resolve_horizontal(routine)
            if hor_index is not None and hor_size is not None:
                promote_variables(
                    routine,
                    variable_names=sorted(vars_to_promote),
                    pos=0,
                    index=hor_index,
                    size=hor_size,
                )

        n_splits = sum(len(v) - 1 for v in loop_map.values())
        info(
            '%s: split %d horizontal loop(s) into %d additional loop(s).',
            routine.name, len(loop_map), n_splits,
        )

    # -----------------------------------------------------------------
    # Section partitioning
    # -----------------------------------------------------------------

    def _partition_loop_body(self, loop):
        """
        Partition the top-level nodes of *loop*.body into contiguous
        sections of vectorisable (``'vec'``) and non-vectorisable
        (``'non'``) nodes.

        Neutral nodes (comments, pragmas) are attached to the *next*
        non-neutral section.  Trailing neutral nodes are attached to
        the *last* section.

        When :attr:`fp_strict` is enabled, assignments and
        conditionals containing calls to intrinsics listed in
        :attr:`fp_strict_intrinsics` are downgraded to
        non-vectorisable.

        Assignments and conditionals that use indirect array indexing
        on the horizontal dimension (gather/scatter patterns) are
        always downgraded to non-vectorisable regardless of
        :attr:`fp_strict`.

        Vectorisable sections that contain fewer than
        :attr:`min_num_assignments` assignments are downgraded to
        non-vectorisable.

        Adjacent sections of the same kind are merged.

        Returns
        -------
        list of (str, list of :any:`Node`)
            Each element is ``('vec', [nodes])`` or ``('non', [nodes])``.
        """
        # First pass: classify each node.
        classified = []
        for node in loop.body:
            if isinstance(node, _NEUTRAL_TYPES):
                classified.append(('neutral', node))
            elif isinstance(node, _VECTORISABLE_TYPES):
                # In fp_strict mode, demote assignments/conditionals
                # that contain calls to unsafe math intrinsics.
                if self.fp_strict and self._node_has_unsafe_intrinsics(node):
                    classified.append(('non', node))
                # Demote assignments/conditionals that use indirect
                # (gather/scatter) array indexing on the horizontal
                # dimension — these prevent effective SIMD execution.
                elif self._node_has_indirect_indexing(node):
                    classified.append(('non', node))
                else:
                    classified.append(('vec', node))
            else:
                classified.append(('non', node))

        # Second pass: resolve neutrals by attaching them to the next
        # non-neutral section.  Trailing neutrals go to the previous.
        resolved = []
        pending_neutrals = []
        for kind, node in classified:
            if kind == 'neutral':
                pending_neutrals.append(node)
            else:
                if pending_neutrals:
                    for n in pending_neutrals:
                        resolved.append((kind, n))
                    pending_neutrals = []
                resolved.append((kind, node))

        # Attach any trailing neutrals to the last section kind.
        if pending_neutrals and resolved:
            last_kind = resolved[-1][0]
            for n in pending_neutrals:
                resolved.append((last_kind, n))
        elif pending_neutrals:
            # Entire body is neutral — treat as vectorisable.
            for n in pending_neutrals:
                resolved.append(('vec', n))

        # Third pass: group into contiguous sections.
        if not resolved:
            return []

        sections = []
        cur_kind, cur_nodes = resolved[0][0], [resolved[0][1]]
        for kind, node in resolved[1:]:
            if kind == cur_kind:
                cur_nodes.append(node)
            else:
                sections.append((cur_kind, cur_nodes))
                cur_kind, cur_nodes = kind, [node]
        sections.append((cur_kind, cur_nodes))

        # Fourth pass: downgrade small vectorisable sections.
        if self.min_num_assignments > 1:
            sections = [
                (kind if kind != 'vec' or
                 self._count_assignments(nodes) >= self.min_num_assignments
                 else 'non', nodes)
                for kind, nodes in sections
            ]

        # Fifth pass: merge adjacent sections of the same kind.
        merged = [sections[0]]
        for kind, nodes in sections[1:]:
            if kind == merged[-1][0]:
                merged[-1] = (kind, merged[-1][1] + nodes)
            else:
                merged.append((kind, nodes))

        return merged

    # -----------------------------------------------------------------
    # Scalar promotion analysis
    # -----------------------------------------------------------------

    def _find_scalars_to_promote(self, sections, loop, routine):
        """
        Identify local scalar variables that are written in one section
        and read in a later section, requiring promotion.

        Returns a set of lower-cased variable names.
        """
        local_names = {
            v.name.lower() for v in get_local_variables(routine, routine.spec)
        }

        # For each section, collect defined and used variable names.
        sec_defs = []
        sec_uses = []
        for _, nodes in sections:
            defined = set()
            used = set()
            for node in nodes:
                # Collect LHS names of assignments (definitions).
                for assign in FindNodes(ir.Assignment).visit(node):
                    name = assign.lhs.name.lower()
                    if not isinstance(assign.lhs, Array):
                        defined.add(name)
                    # Also collect any variables used on the RHS.
                    for v in FindVariables().visit(assign.rhs):
                        used.add(v.name.lower())
                    # Variables used in subscripts on LHS also count as used.
                    if isinstance(assign.lhs, Array):
                        for v in FindVariables().visit(assign.lhs):
                            if v.name.lower() != assign.lhs.name.lower():
                                used.add(v.name.lower())

                # Collect variables used in conditionals and other nodes.
                for cond in FindNodes(ir.Conditional).visit(node):
                    for v in FindVariables().visit(cond.condition):
                        used.add(v.name.lower())

                # Collect variables used in call arguments.
                for call in FindNodes(ir.CallStatement).visit(node):
                    for arg in call.arguments:
                        for v in FindVariables().visit(arg):
                            used.add(v.name.lower())
                    if call.kwarguments:
                        for _, arg in call.kwarguments:
                            for v in FindVariables().visit(arg):
                                used.add(v.name.lower())

                # Collect variables from intrinsics etc. via a catch-all.
                all_vars = FindVariables().visit(node)
                for v in all_vars:
                    used.add(v.name.lower())

            sec_defs.append(defined)
            sec_uses.append(used)

        # A scalar needs promotion if it is:
        # 1. Local (not an argument or import)
        # 2. Defined (assigned as a scalar) in section i
        # 3. Used in section j where j > i
        # 4. Not already an array with the horizontal dimension
        promote = set()
        cumulative_defs = set()
        for i, (kind, nodes) in enumerate(sections):
            if i > 0:
                # Check if any scalar defined in a previous section is
                # used in this section.
                cross = cumulative_defs & sec_uses[i]
                promote |= cross
            cumulative_defs |= sec_defs[i]

        # Filter to local-only and exclude loop indices (outer + nested)
        # and arrays already dimensioned by horizontal.
        loop_indices = {loop.variable.name.lower()}
        for nested_loop in FindNodes(ir.Loop).visit(loop.body):
            loop_indices.add(nested_loop.variable.name.lower())
        promote -= loop_indices
        promote &= local_names

        # Exclude variables that are already arrays with horizontal
        # as their fast dimension.
        var_map = routine.variable_map
        to_remove = set()
        for name in promote:
            var = var_map.get(name)
            if var is not None and isinstance(var, Array):
                if var.shape and str(var.shape[0]).lower() in [
                    s.lower() for s in self.horizontal.sizes
                ]:
                    to_remove.add(name)
        promote -= to_remove

        return promote

    # -----------------------------------------------------------------
    # Loop construction
    # -----------------------------------------------------------------

    def _build_split_loops(self, loop, sections):
        """
        Build a sequence of new :any:`Loop` nodes, one per section,
        each with the same bounds and variable as the original *loop*.
        """
        new_nodes = []
        for i, (kind, nodes) in enumerate(sections):
            # Build a comment to mark split boundaries (except for the
            # first section which inherits the original position).
            preamble = []
            if i > 0:
                preamble.append(ir.Comment(text='! Loki loop-split'))

            new_loop = ir.Loop(
                variable=loop.variable.clone(),
                bounds=sym.LoopRange(loop.bounds.children),
                body=tuple(nodes),
            )
            new_nodes.extend(preamble)
            new_nodes.append(new_loop)

        return new_nodes

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------

    def _is_horizontal_loop(self, loop):
        """Return ``True`` if *loop* iterates over the horizontal dimension."""
        return loop.variable == self.horizontal.index

    def _node_has_unsafe_intrinsics(self, node):
        """
        Return ``True`` if *node* contains any :any:`InlineCall` whose
        name (upper-cased) is in :attr:`fp_strict_intrinsics`.

        For an :any:`Assignment`, the RHS expression tree is scanned.
        For a :any:`Conditional`, both the condition and all nested
        assignment RHS expressions are scanned.
        """
        if isinstance(node, ir.Assignment):
            for call in FindInlineCalls().visit(node.rhs):
                if call.name.upper() in self.fp_strict_intrinsics:
                    return True
        elif isinstance(node, ir.Conditional):
            # Check the condition expression itself.
            for call in FindInlineCalls().visit(node.condition):
                if call.name.upper() in self.fp_strict_intrinsics:
                    return True
            # Check assignments in the conditional body and else-body.
            for assign in FindNodes(ir.Assignment).visit(node):
                for call in FindInlineCalls().visit(assign.rhs):
                    if call.name.upper() in self.fp_strict_intrinsics:
                        return True
        return False

    def _node_has_indirect_indexing(self, node):
        """
        Return ``True`` if *node* contains any array access whose
        horizontal-dimension subscript uses indirect (gather/scatter)
        indexing -- i.e. the subscript contains another :any:`Array`.

        Both LHS and RHS of :any:`Assignment` nodes are checked, as
        well as conditions and nested assignments inside
        :any:`Conditional` nodes.

        The horizontal subscript position is determined per-array from
        the declared shape via :meth:`_find_horizontal_dim_index`.
        """
        if isinstance(node, ir.Assignment):
            return self._expr_has_indirect_indexing(node.lhs, node.rhs)
        if isinstance(node, ir.Conditional):
            if self._expr_has_indirect_indexing(node.condition):
                return True
            for assign in FindNodes(ir.Assignment).visit(node):
                if self._expr_has_indirect_indexing(assign.lhs, assign.rhs):
                    return True
        return False

    def _expr_has_indirect_indexing(self, *exprs):
        """
        Return ``True`` if any :any:`Array` variable found in *exprs*
        uses indirect indexing on its horizontal dimension.

        Indirect indexing means the subscript at the horizontal
        dimension position itself contains an :any:`Array`
        sub-expression (e.g. ``A(IDX(jl))`` where ``IDX`` is an
        array).
        """
        for expr in exprs:
            for var in FindVariables().visit(expr):
                if not isinstance(var, Array):
                    continue
                if not var.dimensions:
                    continue
                hdim = self._find_horizontal_dim_index(var)
                if hdim >= len(var.dimensions):
                    continue
                subscript = var.dimensions[hdim]
                # Check if the subscript expression contains an Array
                # (indicating indirect / gather-scatter access).
                for sub_var in FindVariables().visit(subscript):
                    if isinstance(sub_var, Array):
                        return True
        return False

    def _find_horizontal_dim_index(self, var):
        """
        Return the subscript position that corresponds to the horizontal
        dimension for the given :any:`Array` variable.

        The position is determined by matching elements of *var.shape*
        against :attr:`horizontal.sizes`.  If no match is found (e.g.
        because shape information is unavailable), position ``0`` is
        returned as a fallback consistent with the IFS convention that
        the horizontal dimension is always the leading (fastest-varying)
        dimension.

        Parameters
        ----------
        var : :any:`Array`
            An array variable whose shape is inspected.

        Returns
        -------
        int
            Zero-based index into *var.dimensions* for the horizontal
            subscript.
        """
        hor_sizes_lower = {s.lower() for s in self.horizontal.sizes}
        if var.shape:
            for i, s in enumerate(var.shape):
                if str(s).lower() in hor_sizes_lower:
                    return i
        return 0

    @staticmethod
    def _count_assignments(nodes):
        """Count :any:`Assignment` nodes in a list of IR nodes."""
        count = 0
        for node in nodes:
            count += len(FindNodes(ir.Assignment).visit(node))
        return count

    def _resolve_horizontal(self, routine):
        """
        Resolve the horizontal index and size variables from the
        routine's variable map.

        Returns ``(index_var, size_var)`` or ``(None, None)`` if they
        cannot be resolved.
        """
        variable_map = routine.variable_map
        hor_index = None
        hor_size = None

        for index in self.horizontal.indices:
            if index.split('%', maxsplit=1)[0] in variable_map:
                hor_index = get_integer_variable(routine, index)
                break

        for size in self.horizontal.sizes:
            if size.split('%', maxsplit=1)[0] in variable_map:
                hor_size = get_integer_variable(routine, size)
                break

        return hor_index, hor_size
