# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Module
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, FindNodes, pragmas_attached, pragma_regions_attached,
    is_loki_pragma
)

from loki.transformations.parallel import (
    remove_openmp_regions, add_openmp_regions,
    remove_explicit_firstprivatisation,
    create_explicit_firstprivatisation
)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('insert_loki_parallel', (True, False))
def test_remove_openmp_regions(frontend, insert_loki_parallel):
    """
    A simple test for :any:`remove_openmp_regions`
    """
    fcode = """
subroutine test_driver_openmp(n, arr)
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: arr(n)
  integer :: i

  !$omp parallel private(i)
  !$omp do schedule dynamic(1)
  do i=1, n
    arr(i) = arr(i) + 1.0
  end do
  !$omp end do
  !$omp end parallel


  !$OMP PARALLEL PRIVATE(i)
  !$OMP DO SCHEDULE DYNAMIC(1)
  do i=1, n
    arr(i) = arr(i) + 1.0
  end do
  !$OMP END DO
  !$OMP END PARALLEL


  !$omp parallel do private(i)
  do i=1, n
    !$omp simd
    arr(i) = arr(i) + 1.0
  end do
  !$omp end parallel
end subroutine test_driver_openmp
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3
    assert len(FindNodes(ir.Pragma).visit(routine.body)) == 11

    with pragma_regions_attached(routine):
        # Without attaching Loop-pragmas, all are recognised as regions
        assert len(FindNodes(ir.PragmaRegion).visit(routine.body)) == 5

    remove_openmp_regions(routine, insert_loki_parallel=insert_loki_parallel)

    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3
    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == (6 if insert_loki_parallel else 0)

    if insert_loki_parallel:
        with pragma_regions_attached(routine):
            pragma_regions = FindNodes(ir.PragmaRegion).visit(routine.body)
            assert len(pragma_regions) == 3
            for region in pragma_regions:
                assert is_loki_pragma(region.pragma, starts_with='parallel')
                assert is_loki_pragma(region.pragma_post, starts_with='end parallel')


@pytest.mark.parametrize('frontend', available_frontends())
def test_add_openmp_regions(tmp_path, frontend):
    """
    A simple test for :any:`add_openmp_regions`
    """
    fcode_type = """
module geom_mod
  type geom_type
    integer :: nproma, ngptot
  end type geom_type
end module geom_mod
"""

    fcode = """
subroutine test_add_openmp_loop(ydgeom, arr)
  use geom_mod, only: geom_type
  implicit none
  type(geom_type), intent(in) :: ydgeom
  real(kind=8), intent(inout) :: arr(:,:,:)
  integer :: JKGLO, IBL, ICEND

  !$loki parallel

  DO JKGLO=1,YDGEOM%NGPTOT,YDGEOM%NPROMA
    ICEND = MIN(YDGEOM%NPROMA, YDGEOM%NGPTOT - JKGLO + 1)
    IBL = (JKGLO - 1) / YDGEOM%NPROMA + 1

    CALL MY_KERNEL(ARR(:,:,IBL))
  END DO

  !$loki end parallel

end subroutine test_add_openmp_loop
"""
    _ = Module.from_source(fcode_type, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    assert len(FindNodes(ir.Pragma).visit(routine.body)) == 2
    with pragma_regions_attached(routine):
        regions = FindNodes(ir.PragmaRegion).visit(routine.body)
        assert len(regions) == 1
        assert regions[0].pragma.keyword == 'loki' and regions[0].pragma.content == 'parallel'
        assert regions[0].pragma_post.keyword == 'loki' and regions[0].pragma_post.content == 'end parallel'

    add_openmp_regions(routine)

    # Ensure pragmas have been inserted
    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == 4
    assert all(p.keyword == 'OMP' for p in pragmas)

    with pragmas_attached(routine, node_type=ir.Loop):
        with pragma_regions_attached(routine):
            # Ensure pragma region has been created
            regions = FindNodes(ir.PragmaRegion).visit(routine.body)
            assert len(regions) == 1
            assert regions[0].pragma.keyword == 'OMP'
            assert regions[0].pragma.content.startswith('PARALLEL')
            assert regions[0].pragma_post.keyword == 'OMP'
            assert regions[0].pragma_post.content == 'END PARALLEL'

            # Ensure loops has been annotated
            loops = FindNodes(ir.Loop).visit(routine.body)
            assert len(loops) == 1
            assert loops[0].pragma[0].keyword == 'OMP'
            assert loops[0].pragma[0].content == 'DO SCHEDULE(DYNAMIC,1)'

    # TODO: Test field_group_types and known global variables


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full type definitions for derived types')]
))
def test_remove_explicit_firstprivatisation(frontend):
    """
    A simple test for :any:`remove_explicit_firstprivatisation`
    """
    fcode = """
subroutine test_add_openmp_loop(ydgeom, state, arr)
  use geom_mod, only: geom_type
  implicit none
  type(geom_type), intent(in) :: ydgeom
  real(kind=8), intent(inout) :: arr(:,:,:)
  type(state_type), intent(in) :: state
  type(state_type) :: ydstate
  integer :: jkglo, ibl, icend

  !$loki parallel

  ydstate = state

  do jkglo=1,ydgeom%ngptot,ydgeom%nproma
    icend = min(ydgeom%nproma, ydgeom%ngptot - jkglo + 1)
    ibl = (jkglo - 1) / ydgeom%nproma + 1

    call ydstate%update_view(ibl)

    call my_kernel(ydstate%u(:,:), arr(:,:,ibl))
  end do

  !$loki end parallel
end subroutine test_add_openmp_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    fprivate_map = {'ydstate' : 'state'}

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'ydstate' and assigns[0].rhs == 'state'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('ydstate%')
    assert calls[1].arguments[0].parent == 'ydstate'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1

    # Remove the explicit copy of `ydstate = state` and adjust symbols
    routine.body = remove_explicit_firstprivatisation(
        region=routine.body, fprivate_map=fprivate_map, scope=routine
    )

    # Check removal and symbol replacement
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].lhs == 'icend'
    assert assigns[1].lhs == 'ibl'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('state%')
    assert calls[1].arguments[0].parent == 'state'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full type definitions for derived types')]
))
def test_create_explicit_firstprivatisation(tmp_path, frontend):
    """
    A simple test for :any:`create_explicit_firstprivatisation`
    """
    
    fcode = """
subroutine test_add_openmp_loop(ydgeom, state, arr)
  use geom_mod, only: geom_type
  implicit none
  type(geom_type), intent(in) :: ydgeom
  real(kind=8), intent(inout) :: arr(:,:,:)
  type(state_type), intent(in) :: state
  integer :: jkglo, ibl, icend

  !$loki parallel

  do jkglo=1,ydgeom%ngptot,ydgeom%nproma
    icend = min(ydgeom%nproma, ydgeom%ngptot - jkglo + 1)
    ibl = (jkglo - 1) / ydgeom%nproma + 1

    call state%update_view(ibl)

    call my_kernel(state%u(:,:), arr(:,:,ibl))
  end do

  !$loki end parallel
end subroutine test_add_openmp_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    fprivate_map = {'ydstate' : 'state'}

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].lhs == 'icend'
    assert assigns[1].lhs == 'ibl'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('state%')
    assert calls[1].arguments[0].parent == 'state'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1
    
    # Put the explicit firstprivate copies back in
    create_explicit_firstprivatisation(
        routine=routine, fprivate_map=fprivate_map
    )
    
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'ydstate' and assigns[0].rhs == 'state'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('ydstate%')
    assert calls[1].arguments[0].parent == 'ydstate'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1
