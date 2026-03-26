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
from pathlib import Path

from loki import Subroutine, Dimension, Sourcefile
from loki.frontend import available_frontends
from loki.ir import FindNodes, Assignment, Conditional
from loki.backend import fgen

from loki.transformations.single_column.vertical_kcaching import (
    SCCVerticalKCaching,
)
from loki.transformations.single_column.vertical_complete import (
    _collect_vertical_loops,
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


@pytest.fixture(scope='module', name='cloudsc_vertical')
def fixture_cloudsc_vertical():
    """Vertical dimension matching the dwarf cloudsc kernel."""
    return Dimension(name='vertical', size='KLEV', index='JK')


@pytest.fixture(scope='module', name='cloudsc_horizontal')
def fixture_cloudsc_horizontal():
    """Horizontal dimension matching the dwarf cloudsc kernel."""
    return Dimension(
        name='horizontal', size='KLON', index='JL',
        bounds=('KIDIA', 'KFDIA'), aliases=('NPROMA',)
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _get_cloudsc_path():
    """
    Locate the dwarf cloudsc.F90 kernel file.

    Search order:
    1. Relative to this file's location (traversing up to the workspace root)
    2. Environment variable CLOUDSC_SRC_DIR
    """
    # Walk up from this file to find the workspace root
    workspace = Path(__file__).resolve()
    for _ in range(10):
        workspace = workspace.parent
        candidate = (workspace / 'source' / 'dwarf-p-cloudsc' /
                     'src' / 'cloudsc_loki' / 'cloudsc.F90')
        if candidate.exists():
            return candidate

    import os
    env_dir = os.environ.get('CLOUDSC_SRC_DIR')
    if env_dir:
        candidate = Path(env_dir) / 'cloudsc.F90'
        if candidate.exists():
            return candidate

    return None


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
# Integration test: cloudsc.F90 (the dwarf kernel)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends())
def test_kcaching_cloudsc_integration(frontend, cloudsc_horizontal,
                                       cloudsc_vertical):
    """
    Full integration test: apply SCCVerticalKCaching to the dwarf
    cloudsc.F90 kernel and verify structural properties.
    """
    cloudsc_path = _get_cloudsc_path()
    if cloudsc_path is None:
        pytest.skip('cloudsc.F90 not found')

    source = Sourcefile.from_file(str(cloudsc_path), frontend=frontend)
    routine = source['CLOUDSC']

    # Before: should have multiple JK loops
    loops_before = _count_jk_loops(routine)
    assert loops_before > 1, (
        f'Expected multiple JK loops before transformation, got {loops_before}'
    )

    trafo = SCCVerticalKCaching(
        horizontal=cloudsc_horizontal,
        vertical=cloudsc_vertical
    )
    trafo.process_kernel(routine)

    # After: should have exactly 1 top-level JK loop.
    # FindNodes recurses, so inner JK loops inside the merged body
    # also show up.  Use _collect_vertical_loops which finds only
    # top-level vertical loops.
    top_vloops = _collect_vertical_loops(routine.body, cloudsc_vertical.index)
    assert len(top_vloops) == 1, (
        f'Expected 1 top-level JK loop after transformation, got {len(top_vloops)}'
    )

    # Carry variables should exist
    var_names = [v.name.lower() for v in routine.variables]
    all_vc = [n for n in var_names if n.endswith('_vc')]
    all_next = [n for n in var_names if n.endswith('_next')]
    assert len(all_vc) > 0, 'Expected carry (_vc) variables'

    # The remaining KLEV-dimensioned locals should be only 3D arrays
    # with a species (NCLV) dimension that have computed-index subscripts
    # (e.g. JC(JCC)) preventing demotion.  All pure 2D (KLON,KLEV) temps
    # should have been demoted to scalars.
    # With auto-interchange of JM/JK loops, the 3D arrays (zqx, zqx0,
    # zpfplsx, zlneg, zqxn2d) are also demoted because their JK loops
    # are now inside the merged loop.  No KLEV-dimensioned locals should
    # remain.
    arg_names = {v.name.lower() for v in routine.arguments}
    klev_locals = []
    for var in routine.variables:
        vname = var.name.lower()
        if vname in arg_names:
            continue
        shape = getattr(var.type, 'shape', None) or getattr(var, 'shape', None)
        if not shape:
            continue
        for s in shape:
            s_str = str(s).strip().lower()
            if s_str == 'klev' or s_str.replace(' ', '').startswith('klev+'):
                klev_locals.append(vname)
                break

    # With auto-interchange, all KLEV locals should be demoted
    assert len(klev_locals) == 0, (
        f'Expected no remaining KLEV locals, got {sorted(klev_locals)}'
    )

    # Verify IF guards exist in the merged loop.
    # The merged loop is the top-level JK loop (not nested inside a species
    # loop).  Find it by picking the JK loop with the most conditionals.
    jk_loops = _find_jk_loops(routine)
    merged_loop = max(jk_loops,
                      key=lambda l: len(FindNodes(Conditional).visit(l.body)))
    conds = FindNodes(Conditional).visit(merged_loop.body)
    assert len(conds) >= 10, (
        f'Expected at least 10 IF guards in merged loop, got {len(conds)}'
    )

    # Verify balanced DO/END DO
    code = fgen(routine)
    do_count = len(re.findall(r'^\s*DO\s', code, re.MULTILINE))
    enddo_count = len(re.findall(r'^\s*END\s+DO', code, re.MULTILINE))
    assert do_count == enddo_count, (
        f'Unbalanced DO/END DO: {do_count} vs {enddo_count}'
    )

    # No self-assignment no-ops
    assigns = FindNodes(Assignment).visit(routine.body)
    for a in assigns:
        lhs_str = fgen(a.lhs).strip().lower()
        rhs_str = fgen(a.rhs).strip().lower()
        assert lhs_str != rhs_str, (
            f'Self-assignment no-op: {fgen(a)}'
        )
