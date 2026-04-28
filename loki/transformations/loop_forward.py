# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Transformation to make all backward DO loops iterate in the forward
(ascending-index) direction.

Three categories of backward loop are handled:

**Category A — no loop-carried dependency**
    The iteration order does not affect the result.  The loop bounds are
    simply flipped (start ↔ stop) and the step is dropped (defaulting to
    +1).  The loop body is left unchanged.

    Example::

        ! before
        DO JK = KLEV, 1, -1
          A(JK) = 0.0
        END DO

        ! after
        DO JK = 1, KLEV
          A(JK) = 0.0
        END DO

**Category B — backward recurrence**
    Each iteration reads a value written by the previous iteration (in the
    backward direction), so the computation order matters.  An index
    substitution ``var → start + stop - var`` is applied to the loop body
    so that the forward-going loop counter visits the same data in the
    same order as the original backward loop.  The loop bounds are then
    flipped.

    Example::

        ! before  (start=KLEV-1, stop=3)
        DO JK = KLEV-1, 3, -1
          A(JK) = A(JK+1) + 1.0
        END DO

        ! after  (substitution JK → KLEV+2-JK)
        DO JK = 3, KLEV-1
          A(KLEV+2-JK) = A(KLEV+2-JK+1) + 1.0
        END DO

**Category C — outer backward loop containing inner backward loops**
    Both the outer and nested inner loops are handled iteratively.
    Inner loops are processed first (they appear later in a DFS walk),
    which means the outer loop's substitution subsumes any variable-
    expression bounds (e.g. ``JKK-1``) that were produced by the inner
    transformation.

    After processing all inner loops, the outer loop itself falls into
    Category A or B and is handled accordingly.

The public entry-point is :func:`do_loop_forward`, which drives the
iterative algorithm until no backward loops remain.  A
:class:`~loki.batch.Transformation` wrapper is provided as
:class:`LoopForwardTransformation`.
"""

from loki.analyse import loop_carried_dependencies
from loki.analyse.analyse_dataflow import dataflow_analysis_attached
from loki.batch import Transformation
from loki.expression import symbols as sym, simplify, LoopRange
from loki.ir import FindNodes, Loop, SubstituteExpressions, Transformer

__all__ = ['do_loop_forward', 'LoopForwardTransformation']

_MAX_PASSES = 200  # safety cap for the while-loop


def _is_backward(loop):
    """Return ``True`` when *loop* has a negative step."""
    bounds = loop.bounds
    if bounds is None or bounds.children is None:
        return False
    if len(bounds.children) > 2 and bounds.children[2] is not None:
        step_str = str(bounds.children[2]).strip()
        return step_str.startswith('-') or step_str.startswith('-(')
    return False


def _forward_bounds(loop):
    """
    Return a :any:`LoopRange` with the bounds of *loop* flipped to
    ascending order (step dropped, i.e. defaults to +1).
    """
    return LoopRange((loop.bounds.stop, loop.bounds.start))


def _make_forward_loop(loop, deps):
    """
    Given a single backward *loop* and its set of loop-carried *deps*,
    return an equivalent forward :any:`Loop`.

    Parameters
    ----------
    loop : :any:`Loop`
        A backward loop (``_is_backward(loop)`` must be True).
    deps : set
        The loop-carried dependencies from ``loop_carried_dependencies``.

    Returns
    -------
    :any:`Loop`
        An equivalent loop iterating in the forward direction.
    """
    start = loop.bounds.start
    stop = loop.bounds.stop
    var = loop.variable
    new_bounds = _forward_bounds(loop)

    if not deps:
        # Category A: no carried dependency — just flip the bounds.
        return Loop(variable=var, bounds=new_bounds, body=loop.body,
                    pragma=loop.pragma, pragma_post=loop.pragma_post,
                    loop_label=loop.loop_label, name=loop.name,
                    has_end_do=loop.has_end_do)

    # Category B / C: backward recurrence — apply index substitution
    # var → start + stop - var, then flip bounds.
    subst_expr = simplify(
        sym.Sum((start, sym.Sum((stop, sym.Product((-1, var))))))
    )
    new_body = SubstituteExpressions({var: subst_expr}).visit(loop.body)
    return Loop(variable=var, bounds=new_bounds, body=new_body,
                pragma=loop.pragma, pragma_post=loop.pragma_post,
                loop_label=loop.loop_label, name=loop.name,
                has_end_do=loop.has_end_do)


def do_loop_forward(routine):
    """
    Make all backward DO loops in *routine* iterate in the forward
    (ascending-index) direction.

    The transformation is applied iteratively.  On each pass the
    *innermost* backward loop (last in a depth-first walk) is processed
    first so that outer loops see the already-transformed inner bounds
    when their own substitution is applied.  The loop is replaced in the
    routine body and the process repeats until no backward loops remain.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine whose loop bodies are to be transformed.

    Notes
    -----
    Loops that already iterate in the forward direction are not touched.
    The routine's variable declarations are not modified; the loop
    induction variable retains its original name and type.
    """
    for _ in range(_MAX_PASSES):
        # Re-analyse every pass because the tree is rebuilt each time.
        with dataflow_analysis_attached(routine):
            backward = [
                l for l in FindNodes(Loop).visit(routine.body)
                if _is_backward(l)
            ]
            if not backward:
                break
            # Process the last loop in DFS order = innermost occurrence.
            loop = backward[-1]
            deps = loop_carried_dependencies(loop)

        new_loop = _make_forward_loop(loop, deps)
        routine.body = Transformer({loop: new_loop}).visit(routine.body)


class LoopForwardTransformation(Transformation):
    """
    :any:`Transformation` wrapper around :func:`do_loop_forward`.

    Can be used inside a :any:`Pipeline` or applied directly via
    ``transformation.apply(routine)``.
    """

    def transform_subroutine(self, routine, **kwargs):
        do_loop_forward(routine)
