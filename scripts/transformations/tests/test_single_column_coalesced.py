import sys
from pathlib import Path
import pytest

from loki import (
    OFP, OMNI, FP, Subroutine, Dimension, FindNodes, Loop, Assignment,
    CallStatement, fgen
)

# Bootstrap the local transformations directory for custom transformations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# pylint: disable=wrong-import-position,wrong-import-order
from transformations import SingleColumnCoalescedTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'))


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_single_column_coalesced_simple(frontend, horizontal):
    """
    Test removal of vector loops in kernel and re-insertion of the
    horizontal loop in the "driver".
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, t, nb)
    INTEGER, INTENT(IN)   :: nlon, nz  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b), t(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * k
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich_calls(kernel)  # Attach kernel source to driver call

    scc_transform = SingleColumnCoalescedTransformation(horizontal=horizontal)
    scc_transform.apply(driver, role='driver', targets=['compute_column'])
    scc_transform.apply(kernel, role='kernel')

    # Ensure we only have one vertical loop left
    assert len(FindNodes(Loop).visit(kernel.body)) == 1
    assert FindNodes(Loop).visit(kernel.body)[0].variable == 'jk'
    assert FindNodes(Loop).visit(kernel.body)[0].bounds == '2:nz'

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*k'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure two nested loops found in driver
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 2
    assert driver_loops[1] in driver_loops[0].body
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'
    assert driver_loops[1].variable == 'jl'
    assert driver_loops[1].bounds == 'start:end'

    # Ensure we have a kernel call in the new loop nest
    kernel_calls = FindNodes(CallStatement).visit(driver_loops[1])
    assert len(kernel_calls) == 1
    assert kernel_calls[0].name == 'compute_column'
    assert ('jl', 'jl') in kernel_calls[0].kwarguments
