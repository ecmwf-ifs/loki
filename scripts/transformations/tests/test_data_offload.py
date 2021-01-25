import pytest
import sys
from pathlib import Path

from loki import OFP, OMNI, FP, Sourcefile, FindNodes, Pragma

# Bootstrap the local transformations directory for custom transformations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# pylint: disable=wrong-import-position,wrong-import-order
from transformations import DataOffloadTransformation


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    REAL, INTENT(IN)   :: a(nlon,nlev)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(OUT)   :: c(nlon,nlev)
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
    assert 'copyin(a)' in transformed
    assert 'copy(b)' in transformed
    assert 'copyout(c)' in transformed
