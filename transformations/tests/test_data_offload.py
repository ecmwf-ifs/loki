# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    Sourcefile, FindNodes, Pragma, PragmaRegion, Loop,
    CallStatement, pragma_regions_attached
)
from conftest import available_frontends
from transformations import DataOffloadTransformation


@pytest.mark.parametrize('frontend', available_frontends())
def test_data_offload_region_openacc(frontend):
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

    driver.apply(DataOffloadTransformation(), role='driver', targets=['kernel_routine'])

    assert len(FindNodes(Pragma).visit(driver.body)) == 2
    assert all(p.keyword == 'acc' for p in FindNodes(Pragma).visit(driver.body))
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
  SUBROUTINE driver_routine(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev
    REAL, INTENT(INOUT)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,nlev)
    INTEGER :: j

    !$loki data
    call my_custom_timer()

    !$omp parallel do private(j)
    do j=1, nlev
      call kernel_routine(nlon, j, a(:,j), b(:,j), c(:,j))
    end do
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
        assert len(FindNodes(Loop).visit(regions[0])) == 1

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
