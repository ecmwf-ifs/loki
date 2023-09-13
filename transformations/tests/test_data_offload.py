# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    Sourcefile, FindNodes, Pragma, PragmaRegion, Loop,
    CallStatement, pragma_regions_attached, get_pragma_parameters
)
from conftest import available_frontends
from transformations import DataOffloadTransformation


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('assume_deviceptr', [True, False])
def test_data_offload_region_openacc(frontend, assume_deviceptr):
    """
    Test the creation of a simple device data offload region
    (`!$acc update`) from a `!$loki data` region with a single
    kernel call.
    """

    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(INOUT)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,nlev)

    !$loki data
    call kernel_routine(nlon, nlev, a, b, c)
    !$loki end data

  END SUBROUTINE driver_routine
"""
    fcode_kernel = """
  SUBROUTINE kernel_routine(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(IN)      :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(OUT)     :: c(nlon,nlev)
    INTEGER :: i, j

    do j=1, nlon
      do i=1, nlev
        b(i,j) = a(i,j) + 0.1
        c(i,j) = 0.1
      end do
    end do
  END SUBROUTINE kernel_routine
"""
    driver = Sourcefile.from_source(fcode_driver, frontend=frontend)['driver_routine']
    kernel = Sourcefile.from_source(fcode_kernel, frontend=frontend)['kernel_routine']
    driver.enrich_calls(kernel)

    driver.apply(DataOffloadTransformation(assume_deviceptr=assume_deviceptr), role='driver', targets=['kernel_routine'])

    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 2
    assert all(p.keyword == 'acc' for p in pragmas)
    if assume_deviceptr:
        assert 'deviceptr' in pragmas[0].content
        params = get_pragma_parameters(pragmas[0], only_loki_pragmas=False)
        assert all(var in params['deviceptr'] for var in ('a', 'b', 'c'))
    else:
        transformed = driver.to_fortran()
        assert 'copyin( a )' in transformed
        assert 'copy( b )' in transformed
        assert 'copyout( c )' in transformed


@pytest.mark.parametrize('frontend', available_frontends())
def test_data_offload_region_complex_remove_openmp(frontend):
    """
    Test the creation of a data offload region (OpenACC) with
    driver-side loops and CPU-style OpenMP pragmas to be removed.
    """

    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b, c, flag)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(INOUT)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,nlev)
    logical, intent(in) :: flag
    INTEGER :: j

    !$loki data
    call my_custom_timer()

    if(flag)then
       !$omp parallel do private(j)
       do j=1, nlev
         call kernel_routine(nlon, j, a(:,j), b(:,j), c(:,j))
       end do
       !$omp end parallel do
    else
       !$omp parallel do private(j)
       do j=1, nlev
          a(:,j) = 0.
          b(:,j) = 0.
          c(:,j) = 0.
       end do
       !$omp end parallel do
    endif
    call my_custom_timer()

    !$loki end data
  END SUBROUTINE driver_routine
"""
    fcode_kernel = """
  SUBROUTINE kernel_routine(nlon, j, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, j
    REAL, INTENT(IN)      :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon)
    REAL, INTENT(INOUT)   :: c(nlon)
    INTEGER :: i

    do j=1, nlon
      b(i) = a(i) + 0.1
      c(i) = 0.1
    end do
  END SUBROUTINE kernel_routine
"""
    driver = Sourcefile.from_source(fcode_driver, frontend=frontend)['driver_routine']
    kernel = Sourcefile.from_source(fcode_kernel, frontend=frontend)['kernel_routine']
    driver.enrich_calls(kernel)

    offload_transform = DataOffloadTransformation(remove_openmp=True)
    driver.apply(offload_transform, role='driver', targets=['kernel_routine'])

    assert len(FindNodes(Pragma).visit(driver.body)) == 2
    assert all(p.keyword == 'acc' for p in FindNodes(Pragma).visit(driver.body))

    with pragma_regions_attached(driver):
        # Ensure that loops in the region are preserved
        regions = FindNodes(PragmaRegion).visit(driver.body)
        assert len(regions) == 1
        assert len(FindNodes(Loop).visit(regions[0])) == 2

        # Ensure all activa and inactive calls are there
        calls = FindNodes(CallStatement).visit(regions[0])
        assert len(calls) == 3
        assert calls[0].name == 'my_custom_timer'
        assert calls[1].name == 'kernel_routine'
        assert calls[2].name == 'my_custom_timer'

        # Ensure OpenMP loop pragma is taken out
        assert len(FindNodes(Pragma).visit(regions[0])) == 0

    transformed = driver.to_fortran()
    assert 'copyin( a )' in transformed
    assert 'copy( b, c )' in transformed
    assert '!$omp' not in transformed


@pytest.mark.parametrize('frontend', available_frontends())
def test_data_offload_region_multiple(frontend):
    """
    Test the creation of a device data offload region (`!$acc update`)
    from a `!$loki data` region with multiple kernel calls.
    """

    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b, c, d)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(INOUT)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,nlev)
    REAL, INTENT(INOUT)   :: d(nlon,nlev)

    !$loki data
    call kernel_routine(nlon, nlev, a, b, c)

    call kernel_routine(nlon, nlev, d, b, a)
    !$loki end data

  END SUBROUTINE driver_routine
"""
    fcode_kernel = """
  SUBROUTINE kernel_routine(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(IN)      :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(OUT)     :: c(nlon,nlev)
    INTEGER :: i, j

    do j=1, nlon
      do i=1, nlev
        b(i,j) = a(i,j) + 0.1
        c(i,j) = 0.1
      end do
    end do
  END SUBROUTINE kernel_routine
"""
    driver = Sourcefile.from_source(fcode_driver, frontend=frontend)['driver_routine']
    kernel = Sourcefile.from_source(fcode_kernel, frontend=frontend)['kernel_routine']
    driver.enrich_calls(kernel)

    driver.apply(DataOffloadTransformation(), role='driver', targets=['kernel_routine'])

    assert len(FindNodes(Pragma).visit(driver.body)) == 2
    assert all(p.keyword == 'acc' for p in FindNodes(Pragma).visit(driver.body))

    # Ensure that the copy direction is the union of the two calls, ie.
    # "a" is "copyin" in first call and "copyout" in second, so it should be "copy"
    transformed = driver.to_fortran()
    assert 'copyin( d )' in transformed
    assert 'copy( b, a )' in transformed
    assert 'copyout( c )' in transformed
