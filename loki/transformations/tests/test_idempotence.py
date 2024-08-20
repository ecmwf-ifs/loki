# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import pytest

from loki import Subroutine, fgen
from loki.frontend import available_frontends

from loki.transformations import IdemTransformation


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_idempotence(frontend, tmp_path):
    """ Test the do-nothing equivalence of :any:`IdemTransformations` """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nproma, nlev, nz, q, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    INTEGER, INTENT(IN)   :: nproma, nlev  ! Aliases of horizontal and vertical sizes
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nproma, nz, q(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nproma, nlev, nz, q)
    INTEGER, INTENT(IN) :: start, end   ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz     ! Size of the horizontal and vertical
    INTEGER, INTENT(IN) :: nproma, nlev ! Aliases of horizontal and vertical sizes
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: t(nlon,nz)
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO
  END SUBROUTINE compute_column
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    driver_before = copy.deepcopy(driver)
    kernel_before = copy.deepcopy(kernel)

    idempotence = IdemTransformation()
    idempotence.apply(driver, role='driver')
    idempotence.apply(kernel, role='kernel')

    assert not id(driver_before.ir) == id(driver.ir)
    assert not id(kernel_before.ir) == id(kernel.ir)
    assert driver_before.ir == driver.ir
    assert kernel_before.ir == kernel.ir
    assert fgen(driver_before) == fgen(driver)
    assert fgen(kernel_before) == fgen(kernel)
