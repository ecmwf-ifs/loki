# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`SafeDenominatorGuard` (T2).
"""

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends
from loki.ir import FindNodes, Assignment, Conditional, FindInlineCalls
from loki.backend import fgen

from loki.transformations.cpu.safe_denominator import SafeDenominatorGuard


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


@pytest.mark.parametrize('frontend', available_frontends())
def test_division_guard(frontend, horizontal):
    """
    Division guarded by IF (ZDENOM > 0) should have denominator
    wrapped with MAX(ZDENOM, TINY(1.0)).
    """
    fcode = """
subroutine test_div_guard(kidia, kfdia, klon, zdenom, znum, zresult)
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
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # The conditional should still be present (T2 only guards, T1 removes)
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1

    # The assignment RHS should now contain MAX(..., TINY(...))
    assigns = FindNodes(Assignment).visit(conditionals[0].body)
    assert len(assigns) == 1

    code = fgen(assigns[0]).lower()
    assert 'max' in code
    assert 'tiny' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_sqrt_guard(frontend, horizontal):
    """
    SQRT guarded by IF (ZARG >= 0) should have argument wrapped
    with MAX(ZARG, 0.0).
    """
    fcode = """
subroutine test_sqrt_guard(kidia, kfdia, klon, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zarg(jl) >= 0.0) then
      zresult(jl) = sqrt(zarg(jl))
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1

    assigns = FindNodes(Assignment).visit(conditionals[0].body)
    code = fgen(assigns[0]).lower()
    assert 'max' in code
    assert 'sqrt' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_log_guard(frontend, horizontal):
    """
    LOG guarded by IF (ZARG > 0) should have argument wrapped
    with MAX(ZARG, TINY(1.0)).
    """
    fcode = """
subroutine test_log_guard(kidia, kfdia, klon, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zarg(jl) > 0.0) then
      zresult(jl) = log(zarg(jl))
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1

    assigns = FindNodes(Assignment).visit(conditionals[0].body)
    code = fgen(assigns[0]).lower()
    assert 'max' in code
    assert 'tiny' in code
    assert 'log' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_exp_guard(frontend, horizontal):
    """
    EXP guarded by IF (ZARG < 500) should have argument wrapped
    with MIN(ZARG, 500.0).
    """
    fcode = """
subroutine test_exp_guard(kidia, kfdia, klon, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zarg(jl) < 500.0) then
      zresult(jl) = exp(zarg(jl))
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1

    assigns = FindNodes(Assignment).visit(conditionals[0].body)
    code = fgen(assigns[0]).lower()
    assert 'min' in code
    assert 'exp' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_no_dangerous_ops_no_change(frontend, horizontal):
    """
    Conditional without dangerous ops should not be modified.
    """
    fcode = """
subroutine test_no_danger(kidia, kfdia, klon, llmask, za, zb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zb(klon)
  real, intent(inout) :: za(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      za(jl) = zb(jl) * 2.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    original_code = fgen(routine)

    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # No MAX/MIN should be introduced
    assigns = FindNodes(Assignment).visit(routine.body)
    for a in assigns:
        code = fgen(a).lower()
        assert 'max' not in code
        assert 'min' not in code
        assert 'tiny' not in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_unrelated_condition_not_transformed(frontend, horizontal):
    """
    Division where the condition guards a different variable than the
    denominator should NOT be transformed.
    """
    fcode = """
subroutine test_unrelated(kidia, kfdia, klon, lflag, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: lflag(klon)
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (lflag(jl)) then
      zresult(jl) = znum(jl) / zdenom(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # No MAX/MIN should be introduced (condition is a logical flag,
    # not a comparison, so no guarded variables are extracted)
    assigns = FindNodes(Assignment).visit(routine.body)
    for a in assigns:
        code = fgen(a).lower()
        assert 'max' not in code
        assert 'tiny' not in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_t2_then_t1_combined(frontend, horizontal):
    """
    T2 followed by T1 should produce a MERGE with safe denominator.
    This tests the intended pipeline ordering.
    """
    fcode = """
subroutine test_combined(kidia, kfdia, klon, zdenom, znum, zresult)
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

    from loki.transformations.cpu.merge_conditionals import ConditionalFPGuardToMerge

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Apply T2 first
    t2 = SafeDenominatorGuard(horizontal=horizontal)
    t2.apply(routine, role='kernel')

    # Then apply T1
    t1 = ConditionalFPGuardToMerge(horizontal=horizontal)
    t1.apply(routine, role='kernel')

    # Conditional should now be gone
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    # Assignment should use MERGE with safe denominator
    assigns = FindNodes(Assignment).visit(routine.body)
    merge_assigns = [a for a in assigns
                     if FindInlineCalls().visit(a.rhs)
                     and any(c.name.upper() == 'MERGE'
                             for c in FindInlineCalls().visit(a.rhs))]
    assert len(merge_assigns) == 1

    code = fgen(merge_assigns[0]).lower()
    assert 'merge' in code
    assert 'max' in code
    assert 'tiny' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_lhs_not_substituted(frontend, horizontal):
    """
    When a variable appears both as denominator on the RHS and as the
    LHS of another assignment in the same conditional, the LHS must
    NOT be wrapped with MAX(..., TINY(...)).

    Regression test for the LHS substitution bug found in CLOUDSC
    (ZCOVPTOT(JL) on LHS turned into MAX(ZCOVPTOT(JL), TINY(1.0))).
    """
    fcode = """
subroutine test_lhs_safe(kidia, kfdia, klon, zcov, za, zrain)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(inout) :: zcov(klon), za(klon), zrain(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zcov(jl) > 0.001) then
      zcov(jl) = 1.0 - (1.0 - zcov(jl)) * (1.0 - za(jl))
      zrain(jl) = 0.5 / zcov(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    # The division RHS should be guarded
    assigns = FindNodes(Assignment).visit(routine.body)
    for a in assigns:
        code = fgen(a)
        # Check that no LHS starts with MAX(
        lhs_str = fgen(a.lhs).strip().upper()
        assert not lhs_str.startswith('MAX('), \
            f"LHS should not be wrapped: {code}"

    # But the division denominator should be guarded on the RHS
    rain_assigns = [a for a in assigns if 'zrain' in fgen(a.lhs).lower()]
    assert len(rain_assigns) == 1
    rhs_code = fgen(rain_assigns[0].rhs).lower()
    assert 'max' in rhs_code
    assert 'tiny' in rhs_code


# ---------------------------------------------------------------
# Else-branch guarding tests
# ---------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_else_branch_division_guard(frontend, horizontal):
    """
    Division in the else-branch should have its denominator clamped
    unconditionally (no causal link required).
    """
    fcode = """
subroutine test_else_div(kidia, kfdia, klon, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zdenom(jl) > 0.0) then
      zresult(jl) = znum(jl) * zdenom(jl)
    else
      zresult(jl) = znum(jl) / zdenom(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    cond = conditionals[0]

    # True branch should NOT have MAX/TINY (no dangerous op there)
    body_assigns = FindNodes(Assignment).visit(cond.body)
    assert len(body_assigns) == 1
    body_code = fgen(body_assigns[0]).lower()
    assert 'max' not in body_code
    assert 'tiny' not in body_code

    # Else branch should have MAX/TINY on the denominator
    else_assigns = FindNodes(Assignment).visit(cond.else_body)
    assert len(else_assigns) == 1
    else_code = fgen(else_assigns[0]).lower()
    assert 'max' in else_code
    assert 'tiny' in else_code


@pytest.mark.parametrize('frontend', available_frontends())
def test_else_branch_sqrt_guard(frontend, horizontal):
    """
    SQRT in the else-branch should have its argument clamped with
    MAX(arg, 0.0).
    """
    fcode = """
subroutine test_else_sqrt(kidia, kfdia, klon, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zarg(jl) >= 0.0) then
      zresult(jl) = zarg(jl) * 2.0
    else
      zresult(jl) = sqrt(zarg(jl) + 10.0)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    cond = conditionals[0]

    # Else branch should have MAX wrapping SQRT argument
    else_assigns = FindNodes(Assignment).visit(cond.else_body)
    assert len(else_assigns) == 1
    else_code = fgen(else_assigns[0]).lower()
    assert 'max' in else_code
    assert 'sqrt' in else_code


@pytest.mark.parametrize('frontend', available_frontends())
def test_else_branch_no_causal_link_needed(frontend, horizontal):
    """
    Dangerous ops in the else-branch are clamped even when the
    condition references a completely different variable (no causal
    link).  The true-branch division should NOT be clamped because
    the condition guards zflag, not zdenom.
    """
    fcode = """
subroutine test_else_no_link(kidia, kfdia, klon, zflag, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zflag(klon), zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zflag(jl) > 0.0) then
      zresult(jl) = znum(jl) * 2.0
    else
      zresult(jl) = znum(jl) / zdenom(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    cond = conditionals[0]

    # True branch: no dangerous ops → no clamping
    body_assigns = FindNodes(Assignment).visit(cond.body)
    body_code = fgen(body_assigns[0]).lower()
    assert 'max' not in body_code
    assert 'tiny' not in body_code

    # Else branch: division clamped despite no causal link
    else_assigns = FindNodes(Assignment).visit(cond.else_body)
    else_code = fgen(else_assigns[0]).lower()
    assert 'max' in else_code
    assert 'tiny' in else_code


@pytest.mark.parametrize('frontend', available_frontends())
def test_both_branches_guarded(frontend, horizontal):
    """
    Dangerous operations in both branches should both be guarded.
    """
    fcode = """
subroutine test_both_guard(kidia, kfdia, klon, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zdenom(jl) > 0.0) then
      zresult(jl) = znum(jl) / zdenom(jl)
    else
      zresult(jl) = log(zdenom(jl) + 1.0)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    cond = conditionals[0]

    # True branch: division denominator clamped (causal link exists)
    body_assigns = FindNodes(Assignment).visit(cond.body)
    body_code = fgen(body_assigns[0]).lower()
    assert 'max' in body_code
    assert 'tiny' in body_code

    # Else branch: LOG argument clamped
    else_assigns = FindNodes(Assignment).visit(cond.else_body)
    else_code = fgen(else_assigns[0]).lower()
    assert 'max' in else_code
    assert 'log' in else_code


@pytest.mark.parametrize('frontend', available_frontends())
def test_else_branch_safe_no_change(frontend, horizontal):
    """
    Else-branch with no dangerous ops should not be modified.
    """
    fcode = """
subroutine test_else_safe(kidia, kfdia, klon, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zdenom(jl) > 0.0) then
      zresult(jl) = znum(jl) / zdenom(jl)
    else
      zresult(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    cond = conditionals[0]

    # True branch: division denominator clamped
    body_assigns = FindNodes(Assignment).visit(cond.body)
    body_code = fgen(body_assigns[0]).lower()
    assert 'max' in body_code
    assert 'tiny' in body_code

    # Else branch: just "zresult = 0.0", no MAX/MIN
    else_assigns = FindNodes(Assignment).visit(cond.else_body)
    else_code = fgen(else_assigns[0]).lower()
    assert 'max' not in else_code
    assert 'min' not in else_code
    assert 'tiny' not in else_code


@pytest.mark.parametrize('frontend', available_frontends())
def test_t2_then_t1_else_branch(frontend, horizontal):
    """
    T2 followed by T1 with dangerous ops in both branches should
    produce a MERGE where both tsource and fsource are safe.
    """
    fcode = """
subroutine test_combined_else(kidia, kfdia, klon, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zdenom(jl) > 0.0) then
      zresult(jl) = znum(jl) / zdenom(jl)
    else
      zresult(jl) = sqrt(znum(jl))
    end if
  end do
end subroutine
""".strip()

    from loki.transformations.cpu.merge_conditionals import ConditionalFPGuardToMerge

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Apply T2 first
    t2 = SafeDenominatorGuard(horizontal=horizontal)
    t2.apply(routine, role='kernel')

    # Verify both branches are guarded before T1
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    cond = conditionals[0]

    body_code = fgen(cond.body).lower()
    assert 'max' in body_code and 'tiny' in body_code  # division guard

    else_code = fgen(cond.else_body).lower()
    assert 'max' in else_code and 'sqrt' in else_code  # sqrt guard

    # Then apply T1
    t1 = ConditionalFPGuardToMerge(horizontal=horizontal)
    t1.apply(routine, role='kernel')

    # Conditional should now be gone
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 0

    # Assignment should use MERGE
    assigns = FindNodes(Assignment).visit(routine.body)
    merge_assigns = [a for a in assigns
                     if FindInlineCalls().visit(a.rhs)
                     and any(c.name.upper() == 'MERGE'
                             for c in FindInlineCalls().visit(a.rhs))]
    assert len(merge_assigns) == 1

    code = fgen(merge_assigns[0]).lower()
    assert 'merge' in code
    # Both branches should be safe
    assert 'max' in code
    assert 'tiny' in code  # from division guard
    assert 'sqrt' in code  # from sqrt guard


@pytest.mark.parametrize('frontend', available_frontends())
def test_elseif_dangerous_ops(frontend, horizontal):
    """
    ELSEIF chain where multiple branches contain dangerous operations.
    T2 should guard all of them.
    """
    fcode = """
subroutine test_elseif_guard(kidia, kfdia, klon, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zarg(jl) > 10.0) then
      zresult(jl) = log(zarg(jl))
    else if (zarg(jl) > 0.0) then
      zresult(jl) = sqrt(zarg(jl))
    else
      zresult(jl) = 1.0 / zarg(jl)
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    code = fgen(routine).lower()

    # The LOG in the first branch should be guarded (causal link: zarg > 10)
    # The SQRT in the second branch should be guarded (causal link: zarg > 0)
    # The division in the else should be guarded (no causal link needed)
    # All three should have MAX or MIN wrappers
    assert code.count('max') >= 3  # LOG -> MAX(arg, TINY), SQRT -> MAX(arg, 0), DIV -> MAX(denom, TINY)
    assert 'tiny' in code
    assert 'log' in code
    assert 'sqrt' in code


# ---------------------------------------------------------------
# Unconditional guarding tests (guard_unconditional=True)
# ---------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_unconditional_division_guard(frontend, horizontal):
    """
    Division in an unconditional assignment directly inside a
    horizontal loop should have its denominator clamped with
    MAX(denom, TINY(1.0)) when guard_unconditional=True.
    """
    fcode = """
subroutine test_uncond_div(kidia, kfdia, klon, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = znum(jl) / zdenom(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal, guard_unconditional=True)
    trafo.apply(routine, role='kernel')

    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 1

    code = fgen(assigns[0]).lower()
    assert 'max' in code
    assert 'tiny' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_unconditional_exp_guard(frontend, horizontal):
    """
    EXP call in an unconditional assignment directly inside a
    horizontal loop should have its argument clamped with
    MIN(arg, 500.0) when guard_unconditional=True.
    """
    fcode = """
subroutine test_uncond_exp(kidia, kfdia, klon, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = exp(zarg(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal, guard_unconditional=True)
    trafo.apply(routine, role='kernel')

    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 1

    code = fgen(assigns[0]).lower()
    assert 'min' in code
    assert 'exp' in code
    assert '500' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_unconditional_guard_disabled_by_default(frontend, horizontal):
    """
    With the default guard_unconditional=False, dangerous operations
    in unconditional assignments should NOT be modified.
    """
    fcode = """
subroutine test_uncond_default(kidia, kfdia, klon, zdenom, znum, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = znum(jl) / zdenom(jl)
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal)
    trafo.apply(routine, role='kernel')

    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 1

    code = fgen(assigns[0]).lower()
    assert 'max' not in code
    assert 'tiny' not in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_unconditional_mixed_conditional_and_bare(frontend, horizontal):
    """
    A horizontal loop containing both a conditional (with guarded
    division) and a bare unconditional assignment (with EXP) should
    have both guarded when guard_unconditional=True.  The conditional
    is handled by the first pass; the bare assignment by the second.
    """
    fcode = """
subroutine test_uncond_mixed(kidia, kfdia, klon, zdenom, znum, zarg, zr1, zr2)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: zdenom(klon), znum(klon), zarg(klon)
  real, intent(inout) :: zr1(klon), zr2(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zdenom(jl) > 0.0) then
      zr1(jl) = znum(jl) / zdenom(jl)
    end if
    zr2(jl) = exp(zarg(jl))
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = SafeDenominatorGuard(horizontal=horizontal, guard_unconditional=True)
    trafo.apply(routine, role='kernel')

    # The conditional should still be present
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1

    # Division inside the conditional should be guarded (first pass)
    cond_assigns = FindNodes(Assignment).visit(conditionals[0].body)
    assert len(cond_assigns) == 1
    cond_code = fgen(cond_assigns[0]).lower()
    assert 'max' in cond_code
    assert 'tiny' in cond_code

    # The bare EXP assignment should be guarded (second pass)
    # Find all assignments, then pick the one with EXP
    all_assigns = FindNodes(Assignment).visit(routine.body)
    exp_assigns = [a for a in all_assigns if 'zr2' in fgen(a.lhs).lower()]
    assert len(exp_assigns) == 1
    exp_code = fgen(exp_assigns[0]).lower()
    assert 'min' in exp_code
    assert 'exp' in exp_code
    assert '500' in exp_code
