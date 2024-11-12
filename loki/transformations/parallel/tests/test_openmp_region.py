# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Module, Dimension
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, FindNodes, pragmas_attached, pragma_regions_attached,
    is_loki_pragma
)

from loki.transformations.parallel import (
    do_remove_openmp_regions, add_openmp_regions,
    do_remove_firstprivate_copies, add_firstprivate_copies
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
    !$loki foo-bar
    arr(i) = arr(i) + 1.0
  end do
  !$omp end do
  !$omp end parallel


  !$OMP PARALLEL PRIVATE(i)
  !$OMP DO SCHEDULE DYNAMIC(1)
  do i=1, n
    !$loki foo-baz
    arr(i) = arr(i) + 1.0
    !$loki end foo-baz
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
    assert len(FindNodes(ir.Pragma).visit(routine.body)) == 14

    with pragma_regions_attached(routine):
        # Without attaching Loop-pragmas, all are recognised as regions
        assert len(FindNodes(ir.PragmaRegion).visit(routine.body)) == 6

    do_remove_openmp_regions(routine, insert_loki_parallel=insert_loki_parallel)

    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3
    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == (9 if insert_loki_parallel else 3)

    if insert_loki_parallel:
        with pragma_regions_attached(routine):
            pragma_regions = FindNodes(ir.PragmaRegion).visit(routine.body)
            assert len(pragma_regions) == 4
            assert is_loki_pragma(pragma_regions[0].pragma, starts_with='parallel')
            assert is_loki_pragma(pragma_regions[0].pragma_post, starts_with='end parallel')
            assert is_loki_pragma(pragma_regions[1].pragma, starts_with='parallel')
            assert is_loki_pragma(pragma_regions[1].pragma_post, starts_with='end parallel')
            assert is_loki_pragma(pragma_regions[2].pragma, starts_with='foo-baz')
            assert is_loki_pragma(pragma_regions[2].pragma_post, starts_with='end foo-baz')
            assert is_loki_pragma(pragma_regions[3].pragma, starts_with='parallel')
            assert is_loki_pragma(pragma_regions[3].pragma_post, starts_with='end parallel')


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI has trouble mixing Loop and Section pragmas')]
))
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
subroutine test_add_openmp_loop(ydgeom, ydfields, arr)
  use geom_mod, only: geom_type, fld_type
  use kernel_mod, only: my_kernel, my_non_kernel
  implicit none
  type(geom_type), intent(in) :: ydgeom
  type(fld_type), intent(inout) :: ydfields
  type(fld_type), intent(inout) :: ylfields
  real(kind=8), intent(inout) :: arr(:,:,:)
  integer :: JKGLO, IBL, ICEND

  !$loki parallel

  ylfields = ydfields

  DO JKGLO=1,YDGEOM%NGPTOT,YDGEOM%NPROMA
    ICEND = MIN(YDGEOM%NPROMA, YDGEOM%NGPTOT - JKGLO + 1)
    IBL = (JKGLO - 1) / YDGEOM%NPROMA + 1

    CALL YDFIELDS%UPDATE_STUFF()

    CALL MY_KERNEL(ARR(:,:,IBL))
  END DO

  !$loki end parallel

  !$loki not-so-parallel

  DO JKGLO=1,YDGEOM%NGPTOT,YDGEOM%NPROMA
    call my_non_kernel(arr(1,1,1))
  END DO

  !$loki end not-so-parallel

end subroutine test_add_openmp_loop
"""
    _ = Module.from_source(fcode_type, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    assert len(FindNodes(ir.Pragma).visit(routine.body)) == 4
    with pragma_regions_attached(routine):
        regions = FindNodes(ir.PragmaRegion).visit(routine.body)
        assert len(regions) == 2
        assert is_loki_pragma(regions[0].pragma, starts_with='parallel')
        assert is_loki_pragma(regions[0].pragma_post, starts_with='end parallel')
        assert is_loki_pragma(regions[1].pragma, starts_with='not-so-parallel')
        assert is_loki_pragma(regions[1].pragma_post, starts_with='end not-so-parallel')

    block_dim = Dimension(index='JKGLO', size='YDGEOM%NGPBLK')
    add_openmp_regions(
        routine, dimension=block_dim,
        field_group_types=('fld_type',),
        shared_variables=('ydfields',)
    )

    # Ensure pragmas have been inserted
    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == 6
    assert all(p.keyword == 'OMP' for p in pragmas[0:4])
    assert all(p.keyword == 'loki' for p in pragmas[5:6])

    with pragmas_attached(routine, node_type=ir.Loop):
        with pragma_regions_attached(routine):
            # Ensure pragma region has been created
            regions = FindNodes(ir.PragmaRegion).visit(routine.body)
            assert len(regions) == 2
            assert regions[0].pragma.keyword == 'OMP'
            assert regions[0].pragma.content.startswith('PARALLEL')
            assert regions[0].pragma_post.keyword == 'OMP'
            assert regions[0].pragma_post.content == 'END PARALLEL'
            assert is_loki_pragma(regions[1].pragma, starts_with='not-so-parallel')
            assert is_loki_pragma(regions[1].pragma_post, starts_with='end not-so-parallel')

            # Ensure shared, private and firstprivate have been set right
            assert 'PARALLEL DEFAULT(SHARED)' in regions[0].pragma.content
            assert 'PRIVATE(JKGLO, IBL, ICEND)' in regions[0].pragma.content
            assert 'FIRSTPRIVATE(ylfields)' in regions[0].pragma.content

            # Ensure loops has been annotated
            loops = FindNodes(ir.Loop).visit(routine.body)
            assert len(loops) == 2
            assert loops[0].pragma[0].keyword == 'OMP'
            assert loops[0].pragma[0].content == 'DO SCHEDULE(DYNAMIC,1)'
            assert not loops[1].pragma


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full type definitions for derived types')]
))
def test_remove_firstprivate_copies(frontend):
    """
    A simple test for :any:`remove_firstprivate_copies`
    """
    fcode = """
subroutine test_add_openmp_loop(ydgeom, state, arr)
  use geom_mod, only: geom_type
  use type_mod, only: state_type, flux_type, NewFlux
  implicit none
  type(geom_type), intent(in) :: ydgeom
  real(kind=8), intent(inout) :: arr(:,:,:)
  type(state_type), intent(in) :: state
  type(state_type) :: ydstate
  type(flux_type) :: ydflux
  integer :: jkglo, ibl, icend

  !$loki parallel

  ydstate = state

  ydflux = NewFlux()

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
    assert len(assigns) == 4
    assert assigns[0].lhs == 'ydstate' and assigns[0].rhs == 'state'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('ydstate%')
    assert calls[1].arguments[0].parent == 'ydstate'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1

    # Remove the explicit copy of `ydstate = state` and adjust symbols
    routine.body = do_remove_firstprivate_copies(
        region=routine.body, fprivate_map=fprivate_map, scope=routine
    )

    # Check removal and symbol replacement
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'ydflux'
    assert assigns[1].lhs == 'icend'
    assert assigns[2].lhs == 'ibl'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('state%')
    assert calls[1].arguments[0].parent == 'state'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full type definitions for derived types')]
))
def test_add_firstprivate_copies(frontend):
    """
    A simple test for :any:`add_firstprivate_copies`
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

  !$loki not-so-parallel

  do jkglo=1,ydgeom%ngptot,ydgeom%nproma
    icend = min(ydgeom%nproma, ydgeom%ngptot - jkglo + 1)
    ibl = (jkglo - 1) / ydgeom%nproma + 1
  end do

  !$loki end not-so-parallel
end subroutine test_add_openmp_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    fprivate_map = {'ydstate' : 'state'}

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 4
    assert assigns[0].lhs == 'icend'
    assert assigns[1].lhs == 'ibl'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('state%')
    assert calls[1].arguments[0].parent == 'state'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 2

    # Put the explicit firstprivate copies back in
    add_firstprivate_copies(
        routine=routine, fprivate_map=fprivate_map
    )

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 5
    assert assigns[0].lhs == 'ydstate' and assigns[0].rhs == 'state'
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert str(calls[0].name).startswith('ydstate%')
    assert calls[1].arguments[0].parent == 'ydstate'
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 2
