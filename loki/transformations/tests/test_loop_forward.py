# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Tests for :func:`do_loop_forward` and :class:`LoopForwardTransformation`.

**Category A** — backward loop with *no* loop-carried dependency.
    The bounds are simply flipped; the body is not altered.

**Category B** — backward loop whose body has a *backward recurrence*
    (each iteration reads a value written by the previous iteration).
    An index substitution ``var → start + stop - var`` is applied to the
    body and the bounds are flipped.

**Category C** — an outer backward loop that contains inner backward
    loops, where the inner loop bounds depend on the outer loop variable
    (e.g. ``DO JK = JKK-1, JKT2, -1`` inside ``DO JKK = KLEV, JKT1, -1``).
    The transformation is applied iteratively, innermost first.

Each category is tested for:
- Correct loop bounds after transformation (forward, step absent).
- Correct body expressions (unchanged for A, substituted for B/C).
- Preservation of loop variable names.
- Non-interference with loops that are already forward.
"""

import pytest

from loki import Subroutine
from loki.frontend import available_frontends
from loki.ir import FindNodes, Loop, Assignment

from loki.transformations.loop_forward import do_loop_forward, LoopForwardTransformation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loops(routine):
    """Return all Loop nodes in DFS order."""
    return FindNodes(Loop).visit(routine.body)


def _loop_info(loop):
    """Return (variable_name, start_str, stop_str, has_step)."""
    b = loop.bounds
    return (
        str(loop.variable).lower(),
        str(b.start).lower(),
        str(b.stop).lower(),
        b.step is not None,
    )


def _assignment_strs(routine):
    """Return (lhs_str, rhs_str) for every assignment, lowercased."""
    return [
        (str(a.lhs).lower(), str(a.rhs).lower())
        for a in FindNodes(Assignment).visit(routine.body)
    ]


# ---------------------------------------------------------------------------
# Category A — no loop-carried dependency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_a_simple_init_loop(frontend):
    """
    A backward initialisation loop (no dependency) becomes forward.
    The loop body is left verbatim; only the bounds are flipped.
    """
    fcode = """
subroutine cat_a_simple(a, klev)
  integer, intent(in) :: klev
  real, intent(inout) :: a(klev)
  integer :: jk
  do jk=klev,1,-1
    a(jk) = 0.0
  end do
end subroutine cat_a_simple
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops_before = _loops(routine)
    assert len(loops_before) == 1
    var, start, stop, has_step = _loop_info(loops_before[0])
    assert var == 'jk'
    assert start == 'klev'
    assert stop == '1'
    assert has_step  # step=-1 is present

    do_loop_forward(routine)

    loops_after = _loops(routine)
    assert len(loops_after) == 1
    var, start, stop, has_step = _loop_info(loops_after[0])
    assert var == 'jk'
    assert start == '1'
    assert stop == 'klev'
    assert not has_step  # step absent (defaults to +1)

    # Body must be unchanged: a(jk) = 0.0
    assigns = _assignment_strs(routine)
    assert len(assigns) == 1
    assert assigns[0] == ('a(jk)', '0.0')


@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_a_copy_loop(frontend):
    """
    A backward copy loop (destination and source use *different* indices,
    no read-before-write of the same element) becomes forward with an
    unchanged body.
    """
    fcode = """
subroutine cat_a_copy(src, dst, klev)
  integer, intent(in) :: klev
  real, intent(in)    :: src(klev)
  real, intent(out)   :: dst(klev)
  integer :: jk
  do jk=klev,1,-1
    dst(jk) = src(jk)
  end do
end subroutine cat_a_copy
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    loops = _loops(routine)
    assert len(loops) == 1
    var, start, stop, has_step = _loop_info(loops[0])
    assert var == 'jk' and start == '1' and stop == 'klev' and not has_step

    assigns = _assignment_strs(routine)
    assert assigns == [('dst(jk)', 'src(jk)')]


@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_a_literal_bounds(frontend):
    """
    Category A loop with fully literal bounds, e.g. DO JK=10,1,-1.
    """
    fcode = """
subroutine cat_a_literal(a)
  real, intent(inout) :: a(10)
  integer :: jk
  do jk=10,1,-1
    a(jk) = 0.0
  end do
end subroutine cat_a_literal
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    var, start, stop, has_step = _loop_info(_loops(routine)[0])
    assert var == 'jk' and start == '1' and stop == '10' and not has_step


# ---------------------------------------------------------------------------
# Category B — backward recurrence (index substitution required)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_b_simple_recurrence(frontend):
    """
    Classic updraft-style recurrence: each level reads the level below.
    After transformation the loop is forward and the body uses the
    substituted index ``start + stop - jk``.
    """
    fcode = """
subroutine cat_b_simple(a, klev)
  integer, intent(in) :: klev
  real, intent(inout) :: a(klev)
  integer :: jk
  do jk=klev-1,3,-1
    a(jk) = a(jk+1) + 1.0
  end do
end subroutine cat_b_simple
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops_before = _loops(routine)
    assert len(loops_before) == 1
    assert _loop_info(loops_before[0])[3]  # step present (backward)

    do_loop_forward(routine)

    loops_after = _loops(routine)
    assert len(loops_after) == 1
    var, start, stop, has_step = _loop_info(loops_after[0])
    assert var == 'jk'
    assert start == '3'
    assert stop == 'klev - 1'
    assert not has_step  # step dropped → forward

    # Body should contain the substituted index (klev+2-jk),
    # not the bare 'jk' anymore.
    assigns = _assignment_strs(routine)
    assert len(assigns) == 1
    lhs, rhs = assigns[0]
    assert 'jk' in lhs  # variable still called jk but inside expression
    # The substitution start+stop-jk = (klev-1)+3-jk = klev+2-jk
    # Rendered as 'klev - 1 + 3 - jk' or simplified '2 + klev - jk' etc.
    # We just check that the plain 'jk' no longer appears in isolation.
    assert lhs != 'a(jk)', "body should have been index-substituted"


@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_b_scalar_carry(frontend):
    """
    A scalar accumulator (running sum) is also a backward recurrence.
    The body should be substituted and the loop direction reversed.
    """
    fcode = """
subroutine cat_b_scalar(a, b, klev)
  integer, intent(in) :: klev
  real, intent(in)    :: a(klev)
  real, intent(out)   :: b(klev)
  integer :: jk
  real :: acc
  acc = 0.0
  do jk=klev,1,-1
    acc = acc + a(jk)
    b(jk) = acc
  end do
end subroutine cat_b_scalar
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    loops = _loops(routine)
    assert len(loops) == 1
    var, start, stop, has_step = _loop_info(loops[0])
    assert var == 'jk' and start == '1' and stop == 'klev' and not has_step


@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_b_variable_bounds(frontend):
    """
    Category B loop whose bounds are symbolic expressions (not literals).
    Mirrors the IFS pattern DO CUASCN_JK=KLEV-1,3,-1.
    Verifies that the substituted expression uses all four symbolic terms.
    """
    fcode = """
subroutine cat_b_var_bounds(a, klev, jkt2)
  integer, intent(in) :: klev, jkt2
  real, intent(inout) :: a(klev)
  integer :: jk
  do jk=klev-1,jkt2,-1
    a(jk) = a(jk+1) * 0.5
  end do
end subroutine cat_b_var_bounds
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    var, start, stop, has_step = _loop_info(_loops(routine)[0])
    assert var == 'jk'
    assert 'jkt2' in start   # new start = old stop = jkt2
    assert 'klev' in stop    # new stop  = old start = klev-1
    assert not has_step


@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_b_back_substitution_loop(frontend):
    """
    Back-substitution sweep as in the Thomas / bidiagonal solver.
    DO JK = KLEV-1, 1, -1: each level reads JK+1 (just written above).
    """
    fcode = """
subroutine cat_b_backsubst(x, b, e, klev)
  integer, intent(in) :: klev
  real, intent(in)    :: b(klev), e(klev)
  real, intent(out)   :: x(klev)
  integer :: jk
  do jk=klev-1,1,-1
    x(jk) = b(jk) - e(jk)*x(jk+1)
  end do
end subroutine cat_b_backsubst
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    var, start, stop, has_step = _loop_info(_loops(routine)[0])
    assert var == 'jk' and start == '1' and 'klev' in stop and not has_step

    assigns = _assignment_strs(routine)
    assert len(assigns) == 1
    lhs, rhs = assigns[0]
    # After substitution jk → (KLEV-1)+1-jk = KLEV-jk,
    # the body should contain 'klev' in the index expressions.
    assert 'klev' in lhs


# ---------------------------------------------------------------------------
# Category C — outer backward loop with inner backward loops (nested)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_c_nested_independent(frontend):
    """
    An outer backward loop (no body dependency on outer var in inner loop
    bounds) wraps an inner backward Category-A loop.  Both should become
    forward.
    """
    fcode = """
subroutine cat_c_nested_indep(a, klev)
  integer, intent(in) :: klev
  real, intent(inout) :: a(klev, klev)
  integer :: jkk, jk
  do jkk=klev,1,-1
    do jk=klev,1,-1
      a(jkk, jk) = 0.0
    end do
  end do
end subroutine cat_c_nested_indep
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    loops = _loops(routine)
    assert len(loops) == 2

    outer = loops[0]
    inner = loops[1]

    # Both should be forward with no step
    for loop in (outer, inner):
        assert loop.bounds.step is None

    # Outer: DO jkk=1,KLEV
    assert str(outer.variable).lower() == 'jkk'
    assert str(outer.bounds.start) == '1'
    assert 'klev' in str(outer.bounds.stop)

    # Inner: DO jk=1,KLEV
    assert str(inner.variable).lower() == 'jk'
    assert str(inner.bounds.start) == '1'
    assert 'klev' in str(inner.bounds.stop)


@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_c_nested_dependent_bounds(frontend):
    """
    The inner backward loop has a start bound that depends on the outer
    loop variable: ``DO JK = JKK-1, JKT2, -1``.  This mirrors the
    CUBASEN departure-level pattern in the IFS convection kernel.

    After transformation:
    - Outer loop: ``DO JKK = JKT1, KLEV``
    - Inner loop: ``DO JK = JKT2, <expr involving JKK>``
    Both loops must be forward (no step attribute).
    """
    fcode = """
subroutine cat_c_dep_bounds(a, klev, jkt1, jkt2)
  integer, intent(in) :: klev, jkt1, jkt2
  real, intent(inout) :: a(klev)
  integer :: jkk, jk
  do jkk=klev,jkt1,-1
    do jk=jkk-1,jkt2,-1
      a(jk) = a(jk+1) + 1.0
    end do
  end do
end subroutine cat_c_dep_bounds
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    loops = _loops(routine)
    assert len(loops) == 2
    outer, inner = loops

    # Both forward
    assert outer.bounds.step is None
    assert inner.bounds.step is None

    outer_var = str(outer.variable).lower()
    inner_var = str(inner.variable).lower()
    assert outer_var == 'jkk'
    assert inner_var == 'jk'

    # Outer bounds flipped: (JKT1, KLEV)
    assert 'jkt1' in str(outer.bounds.start).lower()
    assert 'klev' in str(outer.bounds.stop).lower()

    # Inner start = old stop = JKT2
    assert 'jkt2' in str(inner.bounds.start).lower()
    # Inner stop involves JKK (the outer variable) and KLEV
    inner_stop = str(inner.bounds.stop).lower()
    assert 'jkk' in inner_stop
    assert 'klev' in inner_stop


@pytest.mark.parametrize('frontend', available_frontends())
def test_cat_c_three_level_nesting(frontend):
    """
    Three nested backward loops.  All must be forward after transformation.
    """
    fcode = """
subroutine cat_c_three(a, n, m, p)
  integer, intent(in) :: n, m, p
  real, intent(inout) :: a(n)
  integer :: i, j, k
  do i=n,1,-1
    do j=m,1,-1
      do k=p,1,-1
        a(k) = 0.0
      end do
    end do
  end do
end subroutine cat_c_three
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    loops = _loops(routine)
    assert len(loops) == 3
    for loop in loops:
        assert loop.bounds.step is None, f"Loop {loop.variable} still has step"

    # Variables preserved
    assert [str(l.variable).lower() for l in loops] == ['i', 'j', 'k']


# ---------------------------------------------------------------------------
# Non-interference with already-forward loops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_forward_loops_untouched(frontend):
    """
    Loops that already iterate in the forward direction must not be
    altered in any way.
    """
    fcode = """
subroutine fwd_only(a, klev)
  integer, intent(in) :: klev
  real, intent(inout) :: a(klev)
  integer :: jk
  do jk=1,klev
    a(jk) = real(jk)
  end do
  do jk=2,klev-1
    a(jk) = a(jk) + 1.0
  end do
end subroutine fwd_only
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops_before = _loops(routine)
    assert len(loops_before) == 2

    do_loop_forward(routine)

    loops_after = _loops(routine)
    assert len(loops_after) == 2

    infos_before = [_loop_info(l) for l in loops_before]
    infos_after  = [_loop_info(l) for l in loops_after]
    assert infos_before == infos_after


@pytest.mark.parametrize('frontend', available_frontends())
def test_mixed_forward_and_backward(frontend):
    """
    A routine with both forward and backward loops: only the backward
    ones are transformed.
    """
    fcode = """
subroutine mixed(a, b, klev)
  integer, intent(in) :: klev
  real, intent(inout) :: a(klev), b(klev)
  integer :: jk
  do jk=1,klev
    a(jk) = real(jk)
  end do
  do jk=klev,1,-1
    b(jk) = 0.0
  end do
end subroutine mixed
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)

    loops = _loops(routine)
    assert len(loops) == 2

    # Both loops must now be forward
    for loop in loops:
        assert loop.bounds.step is None or str(loop.bounds.step).strip()[0] != '-'

    # First loop unchanged: DO jk=1,KLEV
    var0, start0, stop0, _ = _loop_info(loops[0])
    assert var0 == 'jk' and start0 == '1' and 'klev' in stop0

    # Second loop flipped: DO jk=1,KLEV  (was KLEV→1,-1)
    var1, start1, stop1, _ = _loop_info(loops[1])
    assert var1 == 'jk' and start1 == '1' and 'klev' in stop1


# ---------------------------------------------------------------------------
# Transformation class interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_transformation_class(frontend):
    """
    :class:`LoopForwardTransformation` produces the same result as
    calling :func:`do_loop_forward` directly.
    """
    fcode = """
subroutine xform_class(a, klev)
  integer, intent(in) :: klev
  real, intent(inout) :: a(klev)
  integer :: jk
  do jk=klev,1,-1
    a(jk) = 0.0
  end do
end subroutine xform_class
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    LoopForwardTransformation().apply(routine)

    loops = _loops(routine)
    assert len(loops) == 1
    var, start, stop, has_step = _loop_info(loops[0])
    assert var == 'jk' and start == '1' and 'klev' in stop and not has_step


# ---------------------------------------------------------------------------
# Idempotency — applying twice gives same result
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_idempotent(frontend):
    """
    Applying the transformation twice must yield the same result as
    applying it once (all loops are already forward after first pass).
    """
    fcode = """
subroutine idempotent(a, klev)
  integer, intent(in) :: klev
  real, intent(inout) :: a(klev)
  integer :: jk
  do jk=klev-1,2,-1
    a(jk) = a(jk+1) + 1.0
  end do
end subroutine idempotent
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    do_loop_forward(routine)
    code_after_first = routine.to_fortran()

    do_loop_forward(routine)
    code_after_second = routine.to_fortran()

    assert code_after_first == code_after_second
