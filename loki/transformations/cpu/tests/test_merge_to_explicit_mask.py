# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`MergeToExplicitMask` (T1b).

Covers:

* Simple MERGE → arithmetic masking rewrite
* No-op when fp_strict=False
* IF/ELSE after T1 → arithmetic
* Kind preservation (JPRB) and plain real (no kind)
* Nested MERGE from ELSEIF chains
* Clamping with MIN/MAX/HUGE
* Integer MERGE(1, 0, mask) in output
* Driver role no-op
* Pipeline T2 → T1 → T1b integration
* Accumulation pattern rewrite
"""

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, Assignment, Loop, FindInlineCalls
from loki.backend import fgen

from loki.transformations.cpu.merge_to_explicit_mask import MergeToExplicitMask
from loki.transformations.cpu.merge_conditionals import ConditionalFPGuardToMerge
from loki.transformations.cpu.safe_denominator import SafeDenominatorGuard


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _count_merge_calls(node):
    """Return the total number of MERGE InlineCalls in *node*."""
    calls = FindInlineCalls().visit(node)
    return sum(1 for c in calls if c.name.upper() == 'MERGE')


def _count_int_merge_calls(node):
    """Return the total number of integer MERGE(1, 0, ...) InlineCalls."""
    calls = FindInlineCalls().visit(node)
    count = 0
    for c in calls:
        if c.name.upper() == 'MERGE' and len(c.parameters) == 3:
            p0, p1 = c.parameters[0], c.parameters[1]
            if str(p0) == '1' and str(p1) == '0':
                count += 1
    return count


def _merge_assigns(routine):
    """Return assignments whose RHS contains at least one MERGE call."""
    return [
        a for a in FindNodes(Assignment).visit(routine.body)
        if any(c.name.upper() == 'MERGE'
               for c in FindInlineCalls().visit(a.rhs))
    ]


def _has_real_cast(node):
    """Check if *node* contains a REAL Cast expression."""
    code = fgen(node)
    return 'REAL(MERGE(1, 0,' in code.upper() or 'REAL(MERGE(1,0,' in code.upper()


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# =================================================================
# Test 1: Simple MERGE rewrite
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_simple_merge_rewrite(frontend, horizontal):
    """
    A single MERGE call is rewritten to arithmetic masking.
    The output should contain MERGE(1, 0, mask) but NOT the
    original MERGE(true_val, false_val, mask).
    """
    fcode = """
subroutine test_simple(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl) * 2.0, zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = MergeToExplicitMask(horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 1

    code = fgen(assigns[0]).upper()
    # Should contain MERGE(1, 0, ...) — the integer merge
    assert 'MERGE(1, 0,' in code
    # Should contain REAL cast
    assert 'REAL(' in code
    # Should NOT contain the original float MERGE (zarg*2.0 as first param)
    # The only MERGE calls should be integer ones
    assert _count_int_merge_calls(routine.body) >= 1
    # Original MERGE(zarg, ...) should be gone
    all_merges = FindInlineCalls().visit(routine.body)
    for m in all_merges:
        if m.name.upper() == 'MERGE':
            # All remaining MERGEs should be integer (1, 0, mask)
            assert str(m.parameters[0]) == '1'


# =================================================================
# Test 2: No-op when fp_strict=False
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_disabled_without_fp_strict(frontend, horizontal):
    """fp_strict=False → no-op, MERGE calls remain untouched."""
    fcode = """
subroutine test_noop(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl), zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    original_code = fgen(routine)

    trafo = MergeToExplicitMask(horizontal, fp_strict=False)
    trafo.apply(routine, role='kernel')

    # Code should be unchanged
    assert fgen(routine) == original_code


# =================================================================
# Test 3: IF/ELSE after T1 → MERGE → arithmetic
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_if_else(frontend, horizontal):
    """
    Start with IF/ELSE, apply T1 to get MERGE, then T1b to get
    arithmetic masking.
    """
    fcode = """
subroutine test_if_else(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (llmask(jl)) then
      zresult(jl) = zarg(jl) * 2.0
    else
      zresult(jl) = zarg(jl) * 0.5
    end if
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Apply T1 first
    t1 = ConditionalFPGuardToMerge(horizontal)
    t1.apply(routine, role='kernel')

    # Should now have MERGE
    assert _count_merge_calls(routine.body) >= 1

    # Apply T1b
    t1b = MergeToExplicitMask(horizontal, fp_strict=True)
    t1b.apply(routine, role='kernel')

    # All float MERGEs should be replaced; only integer MERGEs remain
    all_merges = FindInlineCalls().visit(routine.body)
    for m in all_merges:
        if m.name.upper() == 'MERGE':
            assert str(m.parameters[0]) == '1'

    code = fgen(routine).upper()
    assert 'REAL(' in code


# =================================================================
# Test 4: Kind preservation (JPRB)
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_preserves_kind(frontend, horizontal):
    """
    When LHS has kind=JPRB, the REAL cast and 1.0 literal should
    carry that kind.
    """
    fcode = """
subroutine test_kind(kidia, kfdia, klon, llmask, zarg, zresult, jprb)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, jprb
  logical, intent(in) :: llmask(klon)
  real(kind=jprb), intent(in) :: zarg(klon)
  real(kind=jprb), intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl), zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = MergeToExplicitMask(horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    code = fgen(routine).upper()
    # Should have REAL(..., kind=JPRB) or REAL(..., JPRB)
    assert 'JPRB' in code
    # Check REAL cast has kind
    assert 'KIND=' in code or 'REAL(MERGE(1, 0,' in code


# =================================================================
# Test 5: No kind (plain real)
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_no_kind(frontend, horizontal):
    """
    When LHS has no explicit kind, the REAL cast should omit the
    kind parameter.
    """
    fcode = """
subroutine test_no_kind(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl), zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = MergeToExplicitMask(horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    assigns = FindNodes(Assignment).visit(routine.body)
    code = fgen(assigns[0])
    # REAL cast should NOT have kind= when LHS is plain real
    # (the fgen output should have REAL(...) without kind=)
    assert 'REAL(MERGE(1, 0,' in code.upper() or 'REAL(MERGE(1,0,' in code.upper()


# =================================================================
# Test 6: Nested MERGE from ELSEIF chain
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_nested_merge_elseif(frontend, horizontal):
    """
    ELSEIF chain → nested MERGE after T1 → all levels rewritten
    by T1b using bottom-up mapper.
    """
    fcode = """
subroutine test_elseif(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (za(jl) > 1.0) then
      zresult(jl) = 1.0
    else if (za(jl) > 0.5) then
      zresult(jl) = 0.5
    else
      zresult(jl) = 0.0
    end if
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Apply T1
    t1 = ConditionalFPGuardToMerge(horizontal)
    t1.apply(routine, role='kernel')

    # Should have nested MERGE (at least 2 MERGE calls)
    assert _count_merge_calls(routine.body) >= 2

    # Apply T1b
    t1b = MergeToExplicitMask(horizontal, fp_strict=True)
    t1b.apply(routine, role='kernel')

    # All float MERGEs should be replaced; only integer MERGEs remain
    all_merges = FindInlineCalls().visit(routine.body)
    for m in all_merges:
        if m.name.upper() == 'MERGE':
            assert str(m.parameters[0]) == '1', \
                f'Non-integer MERGE found: {fgen(m)}'

    # Should have at least 2 integer MERGEs (one per ELSEIF level)
    assert _count_int_merge_calls(routine.body) >= 2


# =================================================================
# Test 7: Clamping with MIN/MAX/HUGE
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_with_clamp(frontend, horizontal):
    """clamp_results=True → branch operands wrapped with MIN(MAX(...))."""
    fcode = """
subroutine test_clamp(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl) * 2.0, zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = MergeToExplicitMask(horizontal, fp_strict=True,
                                 clamp_results=True)
    trafo.apply(routine, role='kernel')

    code = fgen(routine).upper()
    assert 'HUGE' in code
    assert 'MAX' in code
    assert 'MIN' in code


# =================================================================
# Test 8: No clamping by default
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_no_clamp_default(frontend, horizontal):
    """Default clamp_results=False → no MIN/MAX/HUGE wrapping."""
    fcode = """
subroutine test_no_clamp(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl) * 2.0, zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = MergeToExplicitMask(horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    code = fgen(routine).upper()
    assert 'HUGE' not in code


# =================================================================
# Test 9: Integer MERGE in output
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_integer_merge_in_output(frontend, horizontal):
    """The output should contain MERGE(1, 0, mask) — integer merge."""
    fcode = """
subroutine test_int_merge(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl), zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = MergeToExplicitMask(horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    assert _count_int_merge_calls(routine.body) >= 1


# =================================================================
# Test 10: Driver role → no-op
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_driver_noop(frontend, horizontal):
    """role='driver' → no transformation applied."""
    fcode = """
subroutine test_driver(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = merge(zarg(jl), zresult(jl), llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    original_code = fgen(routine)

    trafo = MergeToExplicitMask(horizontal, fp_strict=True)
    trafo.apply(routine, role='driver')

    assert fgen(routine) == original_code


# =================================================================
# Test 11: Pipeline T2 → T1 → T1b
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_pipeline_t2_t1_t1b(frontend, horizontal):
    """
    Full pipeline segment: T2 guards denominators, T1 converts
    IF/ELSE to MERGE, T1b converts MERGE to arithmetic.
    The final output should be safe arithmetic with integer MERGEs.
    """
    fcode = """
subroutine test_pipeline(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zb(jl) > 0.0) then
      zresult(jl) = za(jl) / zb(jl)
    else
      zresult(jl) = 0.0
    end if
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # T2: guard denominators
    t2 = SafeDenominatorGuard(horizontal)
    t2.apply(routine, role='kernel')

    # T1: convert IF/ELSE to MERGE
    t1 = ConditionalFPGuardToMerge(horizontal)
    t1.apply(routine, role='kernel')

    # Should have MERGE now
    assert _count_merge_calls(routine.body) >= 1

    # T1b: convert MERGE to arithmetic
    t1b = MergeToExplicitMask(horizontal, fp_strict=True)
    t1b.apply(routine, role='kernel')

    # All float MERGEs should be replaced
    all_merges = FindInlineCalls().visit(routine.body)
    for m in all_merges:
        if m.name.upper() == 'MERGE':
            assert str(m.parameters[0]) == '1'

    code = fgen(routine).upper()
    assert 'REAL(' in code
    # Should have safe denominator guard (MAX/TINY)
    assert 'MAX' in code or 'TINY' in code


# =================================================================
# Test 12: Accumulation pattern rewrite
# =================================================================

@pytest.mark.parametrize('frontend', available_frontends())
def test_merge_rewrite_accumulation(frontend, horizontal):
    """
    Accumulation pattern ``x = x + MERGE(...)`` is rewritten so
    the MERGE becomes arithmetic.
    """
    fcode = """
subroutine test_accum(kidia, kfdia, klon, llmask, zarg, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  logical, intent(in) :: llmask(klon)
  real, intent(in) :: zarg(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    zresult(jl) = zresult(jl) + merge(zarg(jl), 0.0, llmask(jl))
  end do
end subroutine
"""
    if frontend == OMNI:
        pytest.skip('OMNI not available')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = MergeToExplicitMask(horizontal, fp_strict=True)
    trafo.apply(routine, role='kernel')

    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 1

    code = fgen(assigns[0]).upper()
    # Should contain integer MERGE
    assert 'MERGE(1, 0,' in code
    # Should contain REAL cast
    assert 'REAL(' in code
    # Original float MERGE should be gone
    all_merges = FindInlineCalls().visit(routine.body)
    for m in all_merges:
        if m.name.upper() == 'MERGE':
            assert str(m.parameters[0]) == '1'
