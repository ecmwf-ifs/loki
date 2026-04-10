# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`ConditionalFPGuardToMerge` (T1).

Covers:

* Simple IF / ELSE to MERGE conversion
* Multi-assignment blocks
* Nested conditionals (outer IF stays, inner IF may convert)
* Unconvertible bodies (CALL, loops, etc.)
* Accumulation patterns (``allow_accumulations`` flag)
* ELSEIF chains -> nested MERGE
* Edge cases (non-horizontal loop, driver role, asymmetric branches)
"""

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, Assignment, Conditional, Loop, FindInlineCalls
from loki.backend import fgen

from loki.transformations.cpu.merge_conditionals import ConditionalFPGuardToMerge


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _count_merge_calls(node):
    """Return the total number of MERGE InlineCalls in *node*."""
    calls = FindInlineCalls().visit(node)
    return sum(1 for c in calls if c.name.upper() == 'MERGE')


def _merge_assigns(routine):
    """Return assignments whose RHS contains at least one MERGE call."""
    return [
        a for a in FindNodes(Assignment).visit(routine.body)
        if any(c.name.upper() == 'MERGE'
               for c in FindInlineCalls().visit(a.rhs))
    ]


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# =================================================================
# Original tests (simple IF/ELSE patterns)
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_simple_if_to_merge(frontend, horizontal):
    """
    Single IF with one assignment, no ELSE branch.
    The false-branch value should be the LHS itself (retain previous value).
    """
    fcode = """
subroutine test_simple_if(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      zresult(jl) = zarg(jl) * 2.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # The conditional should be gone
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    # There should be one loop with one assignment using MERGE
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1

    assigns = FindNodes(Assignment).visit(loops[0].body)
    assert len(assigns) == 1

    # Check the MERGE call
    inline_calls = FindInlineCalls().visit(assigns[0].rhs)
    merge_calls = [c for c in inline_calls if c.name.upper() == 'MERGE']
    assert len(merge_calls) == 1
    assert len(merge_calls[0].parameters) == 3

    # LHS should be zresult(jl)
    assert assigns[0].lhs.name.lower() == 'zresult'


@pytest.mark.parametrize('frontend', available_frontends())
def test_if_else_to_merge(frontend, horizontal):
    """
    IF/ELSE with explicit false-branch value.
    """
    fcode = """
subroutine test_if_else(kidia, kfdia, klon, llgo, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llgo(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llgo(jl)) then
      za(jl) = zb(jl) + 1.0
    else
      za(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Conditional should be replaced
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_multiple_assignments_in_if(frontend, horizontal):
    """
    Multiple assignments in the IF body should all become MERGE calls.
    """
    fcode = """
subroutine test_multi_assign(kidia, kfdia, klon, llmask, za, zb, zc, zd)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon), zd(klon)
  real, intent(inout) :: za(klon), zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za(jl) = zb(jl)
      zc(jl) = zd(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_skip_nested_conditional(frontend, horizontal):
    """
    Nested IF inside IF should NOT transform the outer one (body has
    a Conditional, not an Assignment).  The inner IF *may* be
    converted if it is itself a pure-assignment block.
    """
    fcode = """
subroutine test_nested_if(kidia, kfdia, klon, llmask, llflag, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon), llflag(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      if (llflag(jl)) then
        za(jl) = zb(jl)
      end if
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # The outer conditional should still be present
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) >= 1

    outer_conds = [c for c in conditionals if not c.has_elseif]
    assert len(outer_conds) >= 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_skip_call_in_body(frontend, horizontal):
    """
    Conditional with a CALL in the body should NOT be transformed.
    """
    fcode = """
subroutine test_call_in_body(kidia, kfdia, klon, llmask, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      call some_sub(za(jl), zb(jl))
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Conditional should remain
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_non_horizontal_loop_ignored(frontend, horizontal):
    """
    Conditionals inside non-horizontal loops should not be transformed.
    """
    fcode = """
subroutine test_nonhor_loop(kidia, kfdia, klon, klev, llmask, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jk

  do jk = 1, klev
    if (llmask(jk)) then
      za(jk) = zb(jk)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Conditional should remain (not a horizontal loop)
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_driver_role_skipped(frontend, horizontal):
    """
    Transformation should not apply when role is 'driver'.
    """
    fcode = """
subroutine test_driver_skip(kidia, kfdia, klon, llmask, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za(jl) = zb(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='driver')

    # Should be untouched
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_division_in_conditional_to_merge(frontend, horizontal):
    """
    Conditional with division (FP guard pattern) converts to MERGE.
    This tests the core use case from optrpt #15326.
    """
    fcode = """
subroutine test_div_merge(kidia, kfdia, klon, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zdenom(jl) > 0.0) then
      zresult(jl) = znum(jl) / zdenom(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Conditional should be replaced with MERGE
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1


# =================================================================
# Extension 1: Accumulation patterns
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_accumulation_converted_by_default(frontend, horizontal):
    """
    Accumulation pattern (LHS in RHS) should be converted when
    allow_accumulations=True (the default).

    IF (cond) X(JL) = X(JL) + expr
    -->  X(JL) = MERGE(X(JL) + expr, X(JL), cond)
    """
    fcode = """
subroutine test_accumulation(kidia, kfdia, klon, llmask, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za(jl) = za(jl) + zb(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Conditional should be gone (accumulations allowed by default)
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1
    assert assigns[0].lhs.name.lower() == 'za'

    # The MERGE should have 3 parameters: tsource, fsource, mask
    merge_calls = [c for c in FindInlineCalls().visit(assigns[0].rhs)
                   if c.name.upper() == 'MERGE']
    assert len(merge_calls) == 1
    assert len(merge_calls[0].parameters) == 3

    # Verify generated code contains the accumulation in the tsource
    code = fgen(routine)
    assert 'MERGE' in code.upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_accumulation_skipped_when_disabled(frontend, horizontal):
    """
    With allow_accumulations=False, accumulations should NOT be converted.
    """
    fcode = """
subroutine test_accum_skip(kidia, kfdia, klon, llmask, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za(jl) = za(jl) + zb(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(
        horizontal=horizontal, allow_accumulations=False
    )
    trafo.apply(routine, role='kernel')

    # Conditional should remain
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_accumulation_if_else(frontend, horizontal):
    """
    Accumulation in both IF and ELSE branches.

    IF (cond) X = X + a  ELSE  X = X - b
    -->  X = MERGE(X + a, X - b, cond)
    """
    fcode = """
subroutine test_accum_ifelse(kidia, kfdia, klon, llmask, za, zb, zc)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon), zc(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za(jl) = za(jl) + zb(jl)
    else
      za(jl) = za(jl) - zc(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_accumulation_multiple_vars(frontend, horizontal):
    """
    Multiple accumulation variables in the same IF block.
    """
    fcode = """
subroutine test_multi_accum(kidia, kfdia, klon, llmask, za, zb, zc, zd)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon), zd(klon)
  real, intent(inout) :: za(klon), zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za(jl) = za(jl) + zb(jl)
      zc(jl) = zc(jl) + zd(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_accumulation_else_identity(frontend, horizontal):
    """
    Accumulation in ELSE branch (the "true" branch is a reset).

    IF (cond)  X = 0.0
    ELSE       X = X + delta
    --> X = MERGE(0.0, X + delta, cond)
    """
    fcode = """
subroutine test_accum_else_id(kidia, kfdia, klon, za, zb, zdp)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zb(klon), zdp(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zb(jl) > 0.5) then
      za(jl) = 0.0
    else
      za(jl) = za(jl) + zdp(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1

    # Verify the structure: MERGE(0.0, za(jl) + zdp(jl), ...)
    code = fgen(routine).upper()
    assert 'MERGE' in code


# =================================================================
# Extension 2: ELSEIF chain conversion
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_chain_to_nested_merge(frontend, horizontal):
    """
    IF / ELSEIF / ELSE with a single LHS variable should become
    a nested MERGE.

    IF (c1) za = r1  ELSEIF (c2) za = r2  ELSE za = r3
    --> za = MERGE(r1, MERGE(r2, r3, c2), c1)
    """
    fcode = """
subroutine test_elseif(kidia, kfdia, klon, zval, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      za(jl) = zb(jl)
    else if (zval(jl) > 0.5) then
      za(jl) = zb(jl) * 0.5
    else
      za(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # All conditionals should be gone
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1

    # Should have nested MERGE: outer MERGE + inner MERGE = 2 calls
    total_merges = _count_merge_calls(routine.body)
    assert total_merges == 2

    # Verify output
    code = fgen(routine).upper()
    assert 'MERGE' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_no_final_else(frontend, horizontal):
    """
    IF / ELSEIF with no final ELSE — the implicit false value is
    the LHS identity.

    IF (c1) za = r1  ELSEIF (c2) za = r2
    --> za = MERGE(r1, MERGE(r2, za, c2), c1)
    """
    fcode = """
subroutine test_elseif_noe(kidia, kfdia, klon, zval, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      za(jl) = zb(jl)
    else if (zval(jl) > 0.5) then
      za(jl) = zb(jl) * 0.5
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1

    # 2 nested MERGE calls
    total_merges = _count_merge_calls(routine.body)
    assert total_merges == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_three_branches(frontend, horizontal):
    """
    IF / ELSEIF / ELSEIF / ELSE  (4-way) should produce 3 nested MERGEs.
    """
    fcode = """
subroutine test_elseif3(kidia, kfdia, klon, zval, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 2.0) then
      za(jl) = zb(jl)
    else if (zval(jl) > 1.0) then
      za(jl) = zb(jl) * 0.75
    else if (zval(jl) > 0.5) then
      za(jl) = zb(jl) * 0.5
    else
      za(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1

    # 3 nested MERGE calls (one per IF/ELSEIF branch)
    total_merges = _count_merge_calls(routine.body)
    assert total_merges == 3


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_multi_variable(frontend, horizontal):
    """
    ELSEIF chain that assigns two variables in every branch.
    Each variable gets its own nested MERGE chain.
    """
    fcode = """
subroutine test_elseif_mv(kidia, kfdia, klon, zval, za, zb, zc)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon), zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      za(jl) = zb(jl)
      zc(jl) = zb(jl) * 2.0
    else if (zval(jl) > 0.5) then
      za(jl) = zb(jl) * 0.5
      zc(jl) = zb(jl)
    else
      za(jl) = 0.0
      zc(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    # Two variables: za, zc — each gets one assignment with nested MERGE
    assert len(assigns) == 2

    # Each variable has 2 MERGE calls (IF + ELSEIF)
    total_merges = _count_merge_calls(routine.body)
    assert total_merges == 4  # 2 per variable * 2 variables


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_with_accumulations(frontend, horizontal):
    """
    ELSEIF chain with accumulation in each branch.

    IF (c1) X = X * 20  ELSEIF (c2) X = X * 2  [implicit ELSE: X = X]
    --> X = MERGE(X*20, MERGE(X*2, X, c2), c1)
    """
    fcode = """
subroutine test_elseif_accum(kidia, kfdia, klon, ktype, plude, za)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  integer, intent(in) :: ktype(klon)
  real, intent(in) :: plude(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (ktype(jl) >= 2 .and. plude(jl) > 0.001) then
      za(jl) = 20.0 * za(jl)
    else if (ktype(jl) == 1 .and. plude(jl) > 0.001) then
      za(jl) = 2.0 * za(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 1

    total_merges = _count_merge_calls(routine.body)
    assert total_merges == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_skip_call_in_branch(frontend, horizontal):
    """
    ELSEIF chain with a CALL in one branch should NOT be converted.
    """
    fcode = """
subroutine test_elseif_call(kidia, kfdia, klon, zval, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      za(jl) = zb(jl)
    else if (zval(jl) > 0.5) then
      call some_sub(za(jl), zb(jl))
    else
      za(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Should remain unconverted
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) >= 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_skip_no_horizontal_ref(frontend, horizontal):
    """
    ELSEIF chain where one branch condition does not reference the
    horizontal loop variable should NOT be converted.
    """
    fcode = """
subroutine test_elseif_nohor(kidia, kfdia, klon, zval, za, zb, zflag)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon)
  logical, intent(in) :: zflag
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      za(jl) = zb(jl)
    else if (zflag) then
      za(jl) = zb(jl) * 0.5
    else
      za(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Should remain unconverted because second condition is scalar
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) >= 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_partial_variable_coverage(frontend, horizontal):
    """
    ELSEIF chain where different branches assign to different
    variable sets.  Variables missing in a branch should get their
    identity (LHS) value.

    IF (c1)       za = r1; zc = s1
    ELSEIF (c2)   za = r2
    ELSE          za = r3; zc = s3

    --> za = MERGE(r1, MERGE(r2, r3, c2), c1)
        zc = MERGE(s1, MERGE(zc, s3, c2), c1)
                              ^^ identity for missing branch
    """
    fcode = """
subroutine test_elseif_part(kidia, kfdia, klon, zval, za, zb, zc)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon), zc(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      za(jl) = zb(jl)
      zc(jl) = zb(jl) * 3.0
    else if (zval(jl) > 0.5) then
      za(jl) = zb(jl) * 0.5
    else
      za(jl) = 0.0
      zc(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    assigns = _merge_assigns(routine)
    assert len(assigns) == 2

    # za: 2 MERGE (outer + inner), zc: 2 MERGE (outer + inner)
    total_merges = _count_merge_calls(routine.body)
    assert total_merges == 4


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_empty_body_rejected(frontend, horizontal):
    """
    ELSEIF chain where the IF body is empty should NOT be converted.
    (Empty body = no assignments to produce MERGE for.)
    """
    fcode = """
subroutine test_elseif_empty(kidia, kfdia, klon, zval, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      ! empty body
    else if (zval(jl) > 0.5) then
      za(jl) = zb(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # Should remain — empty body in first branch is not convertible
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) >= 1


# =================================================================
# Edge cases / regression tests
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_non_array_lhs_skipped(frontend, horizontal):
    """
    Scalar LHS (no array subscript) inside horizontal loop should be
    skipped.
    """
    fcode = """
subroutine test_scalar_lhs(kidia, kfdia, klon, llmask, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon)
  real :: za
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za = zb(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_condition_no_loop_var_skipped(frontend, horizontal):
    """
    Condition that does not reference the horizontal loop variable
    (loop-invariant) should be skipped.
    """
    fcode = """
subroutine test_invariant_cond(kidia, kfdia, klon, llflag, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llflag
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llflag) then
      za(jl) = zb(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_fgen_output_valid_fortran(frontend, horizontal):
    """
    Verify that the generated Fortran code from both simple and ELSEIF
    conversions is syntactically reasonable (contains MERGE, no IF).
    """
    fcode = """
subroutine test_fgen_valid(kidia, kfdia, klon, zval, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zval(klon), zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zval(jl) > 1.0) then
      za(jl) = zb(jl)
    else if (zval(jl) > 0.5) then
      za(jl) = zb(jl) * 0.5
    else
      za(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = ConditionalFPGuardToMerge(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    code = fgen(routine)
    code_upper = code.upper()

    # Should contain MERGE, not IF/ELSEIF
    assert 'MERGE' in code_upper
    # The IF/ELSEIF/ELSE structure should be gone from the loop
    # (but note fgen might still emit IF for other reasons)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    loop_code = fgen(loops[0]).upper()
    assert 'IF' not in loop_code or 'MERGE' in loop_code
