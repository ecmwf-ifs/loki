# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :mod:`loki.transformations.single_column.vertical_kcaching`.
"""

import re
import pytest

from loki import Subroutine, Dimension
from loki.frontend import available_frontends
from loki.ir import FindNodes, Assignment, Conditional
from loki.backend import fgen

from loki.transformations.single_column.vertical_kcaching import (
    SCCVerticalKCaching,
)
from loki.transformations.single_column.tests.conftest import (
    _count_jk_loops, _find_jk_loops,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',)
    )

@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk', aliases=('nlev',))


# --------------------------------------------------------------------------
# Test: basic two-loop merge with carries
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_basic_merge(frontend, horizontal, vertical):
    """
    Two vertical loops with a forward dependency should be merged
    into one loop.  The local array should be demoted.
    """
    fcode = """
  SUBROUTINE test_basic(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: temp(nlon, nz)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        temp(jl, jk) = pt(jl, jk) * 2.0
      END DO
    END DO

    DO jk = 1, nz
      DO jl = 1, nlon
        pout(jl, jk) = temp(jl, jk) + 1.0
      END DO
    END DO
  END SUBROUTINE test_basic
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert _count_jk_loops(routine) == 2

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    # Should have merged to 1 loop
    assert _count_jk_loops(routine) == 1

    # temp should be demoted (no KLEV dimension remaining)
    var_map = {v.name.lower(): v for v in routine.variables}
    temp_var = var_map.get('temp')
    if temp_var is not None:
        shape = getattr(temp_var.type, 'shape', None) or ()
        for s in shape:
            s_str = str(s).strip().lower()
            assert s_str != 'nz', 'temp should have nz dimension demoted'


# --------------------------------------------------------------------------
# Test: cumulative sum (Pattern A) carry
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_cumsum(frontend, horizontal, vertical):
    """
    Pattern A: cumulative sum with IF(JK==1) init.
    After transformation, should have single loop with carry variable.
    """
    fcode = """
  SUBROUTINE test_cumsum(nlon, nz, delta, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: delta(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: cumul(nlon, nz)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        IF (jk == 1) THEN
          cumul(jl, jk) = 0.0
        ELSE
          cumul(jl, jk) = cumul(jl, jk - 1)
        END IF
        cumul(jl, jk) = cumul(jl, jk) + delta(jl, jk)
      END DO
    END DO

    DO jk = 1, nz
      DO jl = 1, nlon
        pout(jl, jk) = cumul(jl, jk)
      END DO
    END DO
  END SUBROUTINE test_cumsum
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert _count_jk_loops(routine) == 2

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    # Single merged loop
    assert _count_jk_loops(routine) == 1

    # cumul_vc carry variable should exist
    var_names = [v.name.lower() for v in routine.variables]
    assert 'cumul_vc' in var_names


# --------------------------------------------------------------------------
# Test: forward propagation (Pattern B readback)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_pattern_b_readback(frontend, horizontal, vertical):
    """
    Pattern B-readback: write at JK+1, read at JK+1 in same iteration.
    Should produce two carry variables: _vc and _next.
    """
    fcode = """
  SUBROUTINE test_readback(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: flux(nlon, nz + 1)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        flux(jl, jk + 1) = flux(jl, jk) + pt(jl, jk)
        pout(jl, jk) = flux(jl, jk + 1) * 0.5
      END DO
    END DO
  END SUBROUTINE test_readback
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    var_names = [v.name.lower() for v in routine.variables]
    assert 'flux_vc' in var_names
    assert 'flux_next' in var_names


# --------------------------------------------------------------------------
# Test: dead loop elimination
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_dead_loop_removal(frontend, horizontal, vertical):
    """
    A loop that only writes to a local never read elsewhere should be
    removed before merging.
    """
    fcode = """
  SUBROUTINE test_dead(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: zdead(nlon)
    INTEGER :: jl, jk

    ! Dead loop
    DO jk = 1, nz
      DO jl = 1, nlon
        zdead(jl) = pt(jl, jk)
      END DO
    END DO

    ! Live loop
    DO jk = 1, nz
      DO jl = 1, nlon
        pout(jl, jk) = pt(jl, jk) * 2.0
      END DO
    END DO
  END SUBROUTINE test_dead
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert _count_jk_loops(routine) == 2

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    # Dead loop removed, only live loop remains (merged into 1)
    assert _count_jk_loops(routine) == 1


# --------------------------------------------------------------------------
# Test: different bounds merge with IF guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_different_bounds(frontend, horizontal, vertical):
    """
    Two loops with different bounds should merge into one loop with
    IF guards.
    """
    fcode = """
  SUBROUTINE test_diffbounds(nlon, nz, pt, pout, pout2)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL, INTENT(OUT)   :: pout2(nlon, nz + 1)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        pout(jl, jk) = pt(jl, jk) * 2.0
      END DO
    END DO

    DO jk = 1, nz + 1
      DO jl = 1, nlon
        pout2(jl, jk) = 1.0
      END DO
    END DO
  END SUBROUTINE test_diffbounds
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert _count_jk_loops(routine) == 2

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    assert _count_jk_loops(routine) == 1

    # Should have IF guards in the merged loop
    loops = _find_jk_loops(routine)
    conds = FindNodes(Conditional).visit(loops[0].body)
    assert len(conds) >= 2


# --------------------------------------------------------------------------
# Test: zero-init removal
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_zero_init_removal(frontend, horizontal, vertical):
    """
    Whole-array zero-init for a local array with no other outside-loop
    references should be removed.
    """
    fcode = """
  SUBROUTINE test_zeroinit(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: arr(nlon, nz)
    INTEGER :: jl, jk

    arr(:, :) = 0.0

    DO jk = 1, nz
      DO jl = 1, nlon
        arr(jl, jk) = pt(jl, jk) * 2.0
        pout(jl, jk) = arr(jl, jk)
      END DO
    END DO
  END SUBROUTINE test_zeroinit
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    # After transformation, the arr(:,:) = 0.0 should be gone
    code = fgen(routine).lower()
    # Check no whole-array zero init remains
    assert 'arr(:, :) = 0.0' not in code


# --------------------------------------------------------------------------
# Test: self-assignment removal (Phase 4a)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_no_self_assignments(frontend, horizontal, vertical):
    """
    After full transformation, no self-assignment no-ops should remain.
    """
    fcode = """
  SUBROUTINE test_noop(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: temp(nlon, nz)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        temp(jl, jk) = pt(jl, jk) * 2.0
      END DO
    END DO

    DO jk = 1, nz
      DO jl = 1, nlon
        pout(jl, jk) = temp(jl, jk) + 1.0
      END DO
    END DO
  END SUBROUTINE test_noop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    assigns = FindNodes(Assignment).visit(routine.body)
    for a in assigns:
        lhs_str = fgen(a.lhs).strip().lower()
        rhs_str = fgen(a.rhs).strip().lower()
        assert lhs_str != rhs_str, (
            f'Self-assignment no-op should have been removed: {fgen(a)}'
        )


# --------------------------------------------------------------------------
# Test: apply_to filtering
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_apply_to(frontend, horizontal, vertical):
    """
    When apply_to is set, the transformation should only process
    routines in the list.
    """
    fcode = """
  SUBROUTINE ignored_routine(nlon, nz)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL :: a(nlon, nz)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        a(jl, jk) = 1.0
      END DO
    END DO

    DO jk = 1, nz
      DO jl = 1, nlon
        a(jl, jk) = 2.0
      END DO
    END DO
  END SUBROUTINE ignored_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert _count_jk_loops(routine) == 2

    trafo = SCCVerticalKCaching(
        horizontal=horizontal, vertical=vertical,
        apply_to=['some_other_routine']
    )
    trafo.transform_subroutine(routine, role='kernel')

    # Should NOT have been transformed
    assert _count_jk_loops(routine) == 2


# --------------------------------------------------------------------------
# Test: balanced DO/END DO in generated code
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_balanced_fortran(frontend, horizontal, vertical):
    """
    Generated Fortran should have balanced DO/END DO counts.
    """
    fcode = """
  SUBROUTINE test_balanced(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: a(nlon, nz), b(nlon, nz)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        a(jl, jk) = pt(jl, jk)
      END DO
    END DO

    DO jk = 1, nz
      DO jl = 1, nlon
        b(jl, jk) = a(jl, jk) + 1.0
      END DO
    END DO

    DO jk = 1, nz
      DO jl = 1, nlon
        pout(jl, jk) = b(jl, jk)
      END DO
    END DO
  END SUBROUTINE test_balanced
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    code = fgen(routine)
    do_count = len(re.findall(r'^\s*DO\s', code, re.MULTILINE))
    enddo_count = len(re.findall(r'^\s*END\s+DO', code, re.MULTILINE))
    assert do_count == enddo_count, (
        f'Unbalanced DO/END DO: {do_count} vs {enddo_count}'
    )


# --------------------------------------------------------------------------
# Test: stencil pattern carry conversion
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_stencil_pattern(frontend, horizontal, vertical):
    """
    Stencil pattern: a local array written in one loop, then read at
    JK and JK-1 (but not written) in a second loop.  The backward
    offset read should be converted to a carry variable with an
    OOB-guarded init statement.
    """
    fcode = """
  SUBROUTINE test_stencil(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: za(nlon, nz)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        za(jl, jk) = pt(jl, jk)
      END DO
    END DO

    DO jk = 2, nz
      DO jl = 1, nlon
        pout(jl, jk) = za(jl, jk) - za(jl, jk - 1)
      END DO
    END DO
  END SUBROUTINE test_stencil
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert _count_jk_loops(routine) == 2

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    # Stencil carry variable should exist
    var_names = [v.name.lower() for v in routine.variables]
    assert 'za_vc' in var_names

    # Should have merged to 1 loop
    assert _count_jk_loops(routine) == 1

    # The stencil init should contain an OOB guard (IF ... > 1)
    # which appears before the merged loop.
    code = fgen(routine).lower()
    assert '> 1' in code or '>= 2' in code or '.gt. 1' in code

    # Balanced DO/END DO
    do_count = len(re.findall(r'^\s*DO\s', fgen(routine), re.MULTILINE))
    enddo_count = len(re.findall(r'^\s*END\s+DO', fgen(routine), re.MULTILINE))
    assert do_count == enddo_count


# --------------------------------------------------------------------------
# Test: B_readback with INTENT(OUT) argument (writeback insertion)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_argument_writeback(frontend, horizontal, vertical):
    """
    Pattern B_readback on an INTENT(OUT) *argument* array.  After carry
    conversion the original output array is never written to — all
    writes go through the _next carry variable.  The transformation
    must insert a write-back statement before the rotate so that the
    output array is populated correctly.
    """
    fcode = """
  SUBROUTINE test_arg_wb(nlon, nz, psrc, pflux)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: psrc(nlon, nz)
    REAL, INTENT(OUT)   :: pflux(nlon, nz + 1)
    INTEGER :: jl, jk

    DO jk = 1, nz
      DO jl = 1, nlon
        pflux(jl, jk + 1) = pflux(jl, jk) + psrc(jl, jk)
        ! readback at jk+1 to produce a B_readback pattern
        psrc(jl, jk) = pflux(jl, jk + 1) * 0.5
      END DO
    END DO
  END SUBROUTINE test_arg_wb
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    # B_readback carry variables should exist
    var_names = [v.name.lower() for v in routine.variables]
    assert 'pflux_vc' in var_names
    assert 'pflux_next' in var_names

    # The write-back statement should reference the original array
    # and the _next carry variable
    code = fgen(routine).lower()
    assert 'pflux_next' in code
    # pflux should still be written to (via write-back)
    assert 'pflux(' in code or 'pflux =' in code

    # No self-assignment no-ops
    assigns = FindNodes(Assignment).visit(routine.body)
    for a in assigns:
        lhs_str = fgen(a.lhs).strip().lower()
        rhs_str = fgen(a.rhs).strip().lower()
        assert lhs_str != rhs_str, (
            f'Self-assignment no-op should have been removed: {fgen(a)}'
        )


# --------------------------------------------------------------------------
# Test: B_simple pattern carry conversion
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_pattern_b_simple(frontend, horizontal, vertical):
    """
    Pattern B_simple: write at JK+1, read at JK (offset 0), but NO
    readback at JK+1 within the same iteration.  Should produce a
    single carry variable (_vc) and NO _next variable.
    """
    fcode = """
  SUBROUTINE test_b_simple(nlon, nz, pt, pout)
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(IN)    :: pt(nlon, nz)
    REAL, INTENT(OUT)   :: pout(nlon, nz)
    REAL :: za(nlon, nz + 1)
    INTEGER :: jl, jk

    ! First loop: write at JK+1, read at JK -- B_simple pattern
    ! (no readback at JK+1 in the same iteration)
    DO jk = 1, nz
      DO jl = 1, nlon
        za(jl, jk + 1) = za(jl, jk) + pt(jl, jk)
      END DO
    END DO

    ! Second loop references za so the first loop is not dead
    DO jk = 1, nz
      DO jl = 1, nlon
        pout(jl, jk) = za(jl, jk) + pt(jl, jk)
      END DO
    END DO
  END SUBROUTINE test_b_simple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert _count_jk_loops(routine) == 2

    trafo = SCCVerticalKCaching(horizontal=horizontal, vertical=vertical)
    trafo.process_kernel(routine)

    var_names = [v.name.lower() for v in routine.variables]

    # B_simple creates only _vc, not _next
    assert 'za_vc' in var_names
    assert 'za_next' not in var_names

    # Should have merged to 1 loop
    assert _count_jk_loops(routine) == 1
