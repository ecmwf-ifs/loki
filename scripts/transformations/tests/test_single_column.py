import pytest
import sys
from pathlib import Path

from loki import OFP, OMNI, FP, SourceFile

# Bootstrap the local transformations directory for custom transformations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# pylint: disable=wrong-import-position,wrong-import-order
from transformations import Dimension, ExtractSCATransformation


@pytest.fixture(scope='module', name='dimension')
def fixture_dimension():
    return Dimension(name='horizontal', variable='jl', iteration=('jstart', 'jend'))


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_extract_sca_horizontal(frontend):
    """
    Test removal of loops that traverse the full horizontal dimension.
    """

    fcode = """
  SUBROUTINE compute_column(nlon, nz, q, t)
    INTEGER, INTENT(IN)   :: nlon, nz  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = 1, nlon
        t(jl, jk) = c * k
        q(jl, jk) = q(jl, jk-1)  + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = 1, NLON
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    source = SourceFile.from_source(fcode, frontend=frontend)
    dimension = Dimension(name='nlon', variable='jl', iteration=('jstart', 'jend'), aliases=[])
    sca_transform = ExtractSCATransformation(dimension=dimension)
    source.apply(sca_transform, role='kernel')

    exp_loop_nest = """
  c = 5.345
  DO jk=2,nz
    t(jk) = c*k
    q(jk) = q(jk - 1) + t(jk)*c
  END DO
"""
    assert exp_loop_nest in source.to_fortran()
    # OMNI automatically lower-cases everything
    assert "q(nz) = q(nz)*c" in source.to_fortran().lower()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_extract_sca_iteration(frontend):
    """
    Test removal of loops that traverse a defined iterations space.
    """

    fcode = """
  SUBROUTINE compute_column(jstart, jend, nlon, nz, q, t)
    INTEGER, INTENT(IN)   :: jstart, jend  ! Explicit iteration bounds of the kernel
    INTEGER, INTENT(IN)   :: nlon, nz      ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = jstart, jend
        t(jl, jk) = c * k
        q(jl, jk) = q(jl, jk-1)  + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = JSTART, JEND
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    source = SourceFile.from_source(fcode, frontend=frontend)
    dimension = Dimension(name='nlon', variable='jl', iteration=('jstart', 'jend'), aliases=[])
    sca_transform = ExtractSCATransformation(dimension=dimension)
    source.apply(sca_transform, role='kernel')

    exp_loop_nest = """
  c = 5.345
  DO jk=2,nz
    t(jk) = c*k
    q(jk) = q(jk - 1) + t(jk)*c
  END DO
"""
    assert exp_loop_nest in source.to_fortran()
    # OMNI automatically lower-cases everything
    assert "q(nz) = q(nz)*c" in source.to_fortran().lower()
