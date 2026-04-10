# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Integration tests for the CPU vectorisation pipeline (T1-T6).

Pipeline ordering: T5 (Inline) -> T3 (HoistIO) -> T2 (SafeDenom) ->
T1 (MERGE) -> T6 (SIMD Pragmas) -> T4 (Outline)
"""

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    FindNodes, Loop, Assignment, CallStatement,
    Conditional, Comment, Pragma, Intrinsic,
)
from loki.backend import fgen

from loki.transformations.cpu.inline_calls import InlineCallSiteForVectorisation
from loki.transformations.cpu.hoist_io import HoistWriteFromLoop
from loki.transformations.cpu.safe_denominator import SafeDenominatorGuard
from loki.transformations.cpu.merge_conditionals import ConditionalFPGuardToMerge
from loki.transformations.cpu.simd_pragmas import InsertSIMDPragmaDirectives


@pytest.fixture(scope='module')
def horizontal():
    return Dimension(
        name='horizontal', index='jl', size='klon',
        lower='kidia', upper='kfdia'
    )


# -----------------------------------------------------------------
# Full pipeline integration: T5 -> T3 -> T2 -> T1 -> T6
# (T4 omitted because it requires pragma annotations and is tested
#  separately in test_outline_sections.py)
# -----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_full_pipeline_combined(frontend, horizontal):
    """
    A routine that exercises multiple transformations in sequence:
    - An IF/ELSE guard around a division (T2 + T1)
    - A WRITE inside the horizontal loop (T3)
    - A horizontal loop that should get SIMD pragmas (T6)

    After applying T3 -> T2 -> T1 -> T6 in order, we verify that:
    - The WRITE is hoisted out of the loop
    - The IF/ELSE with safe division becomes a MERGE
    - The loop gets SIMD pragmas
    """
    fcode = """
subroutine test_pipeline(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (za(jl) > 0.0) then
      write(*, *) 'Processing JL=', jl
    end if

    if (zb(jl) /= 0.0) then
      zresult(jl) = za(jl) / zb(jl)
    else
      zresult(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Verify initial state
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert len(FindNodes(Conditional).visit(loops[0].body)) >= 1

    # --- T3: Hoist I/O ---
    t3 = HoistWriteFromLoop(horizontal=horizontal)
    t3.apply(routine, role='kernel')

    # WRITE should be gone from loop
    loops = FindNodes(Loop).visit(routine.body)
    io_in_loop = [n for n in FindNodes(Intrinsic).visit(loops[0].body)
                  if 'WRITE' in n.text.upper()]
    assert len(io_in_loop) == 0

    # Flag variable should exist
    code = fgen(routine).upper()
    assert 'LLHOIST' in code

    # --- T2: Safe denominator ---
    t2 = SafeDenominatorGuard(horizontal=horizontal)
    t2.apply(routine, role='kernel')

    # The denominator should now be wrapped with MAX(..., TINY)
    code = fgen(routine).upper()
    # After T2, the division's denominator should be guarded
    # (either SIGN*MAX or MAX pattern)
    assert 'MAX' in code or 'TINY' in code or 'SIGN' in code

    # --- T1: Conditional to MERGE ---
    t1 = ConditionalFPGuardToMerge(horizontal=horizontal)
    t1.apply(routine, role='kernel')

    # The IF/ELSE for the division guard should now be a MERGE
    code = fgen(routine).upper()
    assert 'MERGE' in code

    # --- T6: SIMD pragmas ---
    t6 = InsertSIMDPragmaDirectives(horizontal=horizontal)
    t6.apply(routine, role='kernel')

    # The horizontal loop should have OMP SIMD pragma
    loops = FindNodes(Loop).visit(routine.body)
    code = fgen(routine).upper()
    assert 'OMP' in code and 'SIMD' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_pipeline_preserves_computation(frontend, horizontal):
    """
    Verify that transformations preserve the core computation structure.
    After all transformations, the routine should still contain the
    assignment to zresult and the loop over the horizontal dimension.
    """
    fcode = """
subroutine test_preserve(kidia, kfdia, klon, za, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (za(jl) > 0.0) then
      zresult(jl) = sqrt(za(jl))
    else
      zresult(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Apply T2 then T1 then T6
    t2 = SafeDenominatorGuard(horizontal=horizontal)
    t2.apply(routine, role='kernel')

    t1 = ConditionalFPGuardToMerge(horizontal=horizontal)
    t1.apply(routine, role='kernel')

    t6 = InsertSIMDPragmaDirectives(horizontal=horizontal)
    t6.apply(routine, role='kernel')

    code = fgen(routine).upper()

    # Loop structure preserved
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) >= 1

    # Result variable still referenced
    assert 'ZRESULT' in code

    # MERGE and SIMD should be present
    assert 'MERGE' in code
    assert 'SIMD' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_pipeline_driver_role_noop(frontend, horizontal):
    """
    When role='driver', all kernel transformations should be skipped.
    """
    fcode = """
subroutine test_driver(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zb(jl) /= 0.0) then
      zresult(jl) = za(jl) / zb(jl)
    else
      zresult(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    code_before = fgen(routine)

    for TrafoClass in (HoistWriteFromLoop, SafeDenominatorGuard,
                       ConditionalFPGuardToMerge, InsertSIMDPragmaDirectives):
        trafo = TrafoClass(horizontal=horizontal)
        trafo.apply(routine, role='driver')

    code_after = fgen(routine)

    # No MERGE, no SIMD, no flag variables introduced
    assert 'MERGE' not in code_after.upper()
    assert 'LLHOIST' not in code_after.upper()
    assert '!$OMP SIMD' not in code_after.upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_pipeline_t2_before_t1_order_matters(frontend, horizontal):
    """
    Demonstrate that T2 (SafeDenominator) must run before T1 (MERGE)
    for correct results. If T1 runs first, the conditional is already
    replaced with MERGE and T2 cannot find the guarded division.
    With T2 first, the denominator is safe-guarded, then T1 converts
    the conditional to MERGE.
    """
    fcode = """
subroutine test_order(kidia, kfdia, klon, za, zb, zresult)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon
  real, intent(in) :: za(klon), zb(klon)
  real, intent(inout) :: zresult(klon)
  integer :: jl

  do jl = kidia, kfdia
    if (zb(jl) /= 0.0) then
      zresult(jl) = za(jl) / zb(jl)
    else
      zresult(jl) = 0.0
    end if
  end do
end subroutine
""".strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Correct order: T2 then T1
    t2 = SafeDenominatorGuard(horizontal=horizontal)
    t2.apply(routine, role='kernel')

    t1 = ConditionalFPGuardToMerge(horizontal=horizontal)
    t1.apply(routine, role='kernel')

    code = fgen(routine).upper()

    # Both transformations should have had effect
    assert 'MERGE' in code
    # The denominator should be guarded (MAX/SIGN/TINY pattern)
    assert 'MAX' in code or 'SIGN' in code or 'TINY' in code

    # No IF/ELSE for the guard should remain in the loop
    loops = FindNodes(Loop).visit(routine.body)
    conds_in_loop = FindNodes(Conditional).visit(loops[0].body)
    assert len(conds_in_loop) == 0
