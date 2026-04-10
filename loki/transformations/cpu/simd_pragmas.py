# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformation to insert ``!$OMP SIMD`` or ``!DIR$ SIMD`` pragmas on
horizontal vector loops that would benefit from explicit SIMD hints.

This addresses Intel optrpt remark ``#15541`` (outer loop was not
auto-vectorised) and ``#15335`` (vectorisation deemed inefficient),
affecting ~12 loops across CLOUDSC and CUBASEN.

The transformation also detects scalar variables that are assigned
inside horizontal loops and adds ``PRIVATE(...)`` clauses to
``!$OMP SIMD`` directives, which addresses Intel optrpt remark
``#15316`` (loop was not vectorized: existence of vector dependence)
when the dependence is a false-positive on a loop-private scalar.

The transformation only *annotates* loops with compiler directives and
does **not** alter computational semantics.  It should run late in the
CPU vectorisation pipeline (after T1-T3, T5) so that loop bodies have
already been simplified by the preceding transformations.
"""

from loki.batch import Transformation
from loki.expression import Array
from loki.ir import nodes as ir, FindNodes, Transformer

from loki.transformations.utilities import check_routine_sequential


__all__ = ['InsertSIMDPragmaDirectives']


class InsertSIMDPragmaDirectives(Transformation):
    """
    Insert SIMD pragma directives on horizontal vector loops.

    By default, inserts ``!$OMP SIMD`` before every horizontal loop
    whose body contains only simple statements (assignments, comments,
    existing pragmas).  An Intel-specific ``!DIR$ SIMD`` variant is
    also supported via the *directive* parameter.

    When using the ``'OMP SIMD'`` directive (default), the
    transformation also detects scalar variables that are assigned
    inside the horizontal loop body and adds ``PRIVATE(...)`` clauses
    to eliminate false vector-dependence reports from the Intel
    compiler (optrpt remark ``#15316``).

    Parameters
    ----------
    horizontal : :any:`Dimension`
        The dimension object describing the horizontal iteration space.
    directive : str, optional
        The pragma directive string to insert.  Use ``'OMP SIMD'``
        (default) for portable OpenMP 4.0+, or ``'DIR$ SIMD'`` for
        Intel-specific, or ``'DIR$ IVDEP'`` for a no-dependence
        assertion.
    collapse_outer : bool, optional
        When ``True``, also targets outer (non-horizontal) loops that
        contain exactly one inner horizontal loop, inserting
        ``!$OMP SIMD COLLAPSE(2)`` to collapse both loops.
        Default is ``False``.
    insert_ivdep : bool, optional
        When ``True``, additionally inserts a ``!DIR$ IVDEP`` comment
        before horizontal loops (complementing the primary directive).
        Default is ``False``.
    insert_vector_always : bool, optional
        When ``True``, additionally inserts a ``!DIR$ VECTOR ALWAYS``
        comment before horizontal loops.  This tells the Intel compiler
        to override its cost model and always vectorise the loop.
        Default is ``False``.
    """

    def __init__(self, horizontal, directive='OMP SIMD',
                 collapse_outer=False, insert_ivdep=False,
                 insert_vector_always=False):
        self.horizontal = horizontal
        self.directive = directive
        self.collapse_outer = collapse_outer
        self.insert_ivdep = insert_ivdep
        self.insert_vector_always = insert_vector_always

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SIMD pragma insertion to a single subroutine.

        Skips driver-role routines and sequential routines.
        """
        role = kwargs.get('role', 'kernel')
        if role == 'driver':
            return
        if check_routine_sequential(routine):
            return

        loop_map = {}

        for loop in FindNodes(ir.Loop).visit(routine.body):
            # Check for collapsible outer loop nests first
            if self.collapse_outer and self._is_collapsible_nest(loop):
                if not self._already_has_simd_pragma(loop, routine):
                    pragmas = self._build_collapse_pragmas()
                    loop_map[loop] = (*pragmas, loop)
                continue

            if self._is_horizontal_loop(loop):
                if self._already_has_simd_pragma(loop, routine):
                    continue
                if self._is_simple_loop_body(loop):
                    private_vars = self._get_private_scalars(loop)
                    pragmas = self._build_pragmas(private_vars=private_vars)
                    loop_map[loop] = (*pragmas, loop)

        if loop_map:
            routine.body = Transformer(loop_map).visit(routine.body)

    # -----------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------

    def _get_private_scalars(self, loop):
        """
        Identify scalar variables assigned inside *loop* that should be
        declared ``PRIVATE`` in an ``!$OMP SIMD`` directive.

        A variable is considered a private scalar when:

        1. It appears on the LHS of an assignment inside the loop body
           (including nested conditionals).
        2. It is **not** an :any:`Array` reference (i.e. it has no
           subscripts / dimensions).
        3. It is **not** the loop induction variable itself.

        Returns
        -------
        tuple of str
            Sorted tuple of unique uppercase variable names.
        """
        assigns = FindNodes(ir.Assignment).visit(loop.body)
        loop_var = loop.variable.name.upper()

        scalars = set()
        for assign in assigns:
            lhs = assign.lhs
            if not isinstance(lhs, Array) and lhs.name.upper() != loop_var:
                scalars.add(lhs.name.upper())

        return tuple(sorted(scalars))

    def _is_horizontal_loop(self, loop):
        """Return ``True`` if *loop* iterates over the horizontal dimension."""
        return loop.variable == self.horizontal.index

    def _is_simple_loop_body(self, loop):
        """
        Return ``True`` if the loop body contains only "simple" nodes
        that are safe targets for SIMD annotation.

        Simple nodes are: :any:`Assignment`, :any:`Comment`,
        :any:`Pragma`, :any:`CommentBlock`, and :any:`Conditional`
        (masked branches are fine in SIMD execution).
        """
        for node in loop.body:
            if not isinstance(node, (ir.Assignment, ir.Comment,
                                     ir.Pragma, ir.CommentBlock,
                                     ir.Conditional)):
                return False
        return True

    def _is_collapsible_nest(self, loop):
        """
        Return ``True`` if *loop* is a non-horizontal outer loop with
        exactly one child that is a horizontal loop.
        """
        if self._is_horizontal_loop(loop):
            return False
        inner_loops = [n for n in loop.body if isinstance(n, ir.Loop)]
        if len(inner_loops) != 1:
            return False
        # The single inner loop must be horizontal and the outer body
        # must not contain anything other than the inner loop,
        # comments, and pragmas (perfectly nested).
        inner = inner_loops[0]
        if not self._is_horizontal_loop(inner):
            return False
        for node in loop.body:
            if node is inner:
                continue
            if not isinstance(node, (ir.Comment, ir.Pragma, ir.CommentBlock)):
                return False
        return True

    def _already_has_simd_pragma(self, loop, routine):
        """
        Return ``True`` if a SIMD directive already appears immediately
        before *loop* in the routine body (either as a :any:`Pragma` or
        :any:`Comment`).
        """
        # Check the loop's own pragma attribute (set when pragmas_attached
        # context was used, or when constructed with pragma=)
        if loop.pragma:
            for p in loop.pragma:
                if p.content and 'SIMD' in p.content.upper():
                    return True

        # Walk the routine body and check the node immediately before
        # the loop for a SIMD pragma/comment.
        body = _flatten_body(routine.body)
        for i, node in enumerate(body):
            if node is loop and i > 0:
                prev = body[i - 1]
                if isinstance(prev, ir.Pragma) and prev.content and \
                        'SIMD' in prev.content.upper():
                    return True
                if isinstance(prev, ir.Comment) and prev.text and \
                        'SIMD' in prev.text.upper():
                    return True
        return False

    # -----------------------------------------------------------------
    # Pragma construction helpers
    # -----------------------------------------------------------------

    def _build_pragmas(self, private_vars=None):
        """
        Build the sequence of pragma/comment nodes to insert before a
        horizontal loop.

        Parameters
        ----------
        private_vars : tuple of str, optional
            Variable names to include in a ``PRIVATE(...)`` clause
            on the ``!$OMP SIMD`` directive.  Only applies when the
            primary directive is ``'OMP SIMD'`` (ignored for ``DIR$``
            directives, which have no ``PRIVATE`` clause syntax).

        Returns a tuple of IR nodes.
        """
        nodes = []

        if self.insert_ivdep:
            nodes.append(ir.Comment(text='!DIR$ IVDEP'))

        if self.insert_vector_always:
            nodes.append(ir.Comment(text='!DIR$ VECTOR ALWAYS'))

        nodes.append(self._make_directive_node(self.directive,
                                               private_vars=private_vars))
        return tuple(nodes)

    def _build_collapse_pragmas(self):
        """
        Build the pragma nodes for a collapsible outer-loop nest.

        Always uses ``!$OMP SIMD COLLAPSE(2)`` regardless of the
        primary *directive* setting, because ``COLLAPSE`` is an
        OpenMP-only clause.
        """
        nodes = []

        if self.insert_ivdep:
            nodes.append(ir.Comment(text='!DIR$ IVDEP'))

        if self.insert_vector_always:
            nodes.append(ir.Comment(text='!DIR$ VECTOR ALWAYS'))

        nodes.append(ir.Pragma(keyword='OMP', content='SIMD COLLAPSE(2)'))
        return tuple(nodes)

    @staticmethod
    def _make_directive_node(directive, private_vars=None):
        """
        Create the appropriate IR node for *directive*.

        Intel ``!DIR$`` directives are rendered as :any:`Comment` nodes
        (since fgen renders ``Pragma`` as ``!$keyword content`` which
        would produce ``!$DIR$ SIMD`` instead of ``!DIR$ SIMD``).

        OpenMP directives use :any:`Pragma` nodes.

        Parameters
        ----------
        directive : str
            The directive string, e.g. ``'OMP SIMD'`` or ``'DIR$ SIMD'``.
        private_vars : tuple of str, optional
            Variable names to include in a ``PRIVATE(...)`` clause.
            Only applied for OpenMP-style directives (not ``DIR$``).
        """
        parts = directive.split(None, 1)
        keyword = parts[0]
        content = parts[1] if len(parts) > 1 else ''

        if keyword.startswith('DIR'):
            # Intel compiler directives: use Comment for correct rendering
            return ir.Comment(text=f'!{keyword} {content}'.rstrip())
        else:
            # OpenMP / other: use Pragma node
            # Append PRIVATE clause if we have private scalars
            if private_vars:
                private_clause = f' PRIVATE({", ".join(private_vars)})'
                content = (content or '') + private_clause
            return ir.Pragma(keyword=keyword, content=content or None)


def _flatten_body(body):
    """
    Recursively flatten a routine body into a flat list of IR nodes.

    This is used by :meth:`_already_has_simd_pragma` to locate nodes
    immediately preceding a loop, even when they are nested inside
    ``Section`` or similar container nodes.

    *body* may be a :any:`Section` node, a tuple of nodes, or a list.
    """
    # Handle Section nodes (e.g. routine.body is a Section)
    if isinstance(body, ir.Section):
        body = body.body

    result = []
    for node in body:
        if isinstance(node, ir.Section) and hasattr(node, 'body'):
            result.extend(_flatten_body(node.body))
        else:
            result.append(node)
    return result
