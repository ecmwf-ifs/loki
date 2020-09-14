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
    REAL, INTENT(INOUT)   :: t(jend-jstart+1,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = jstart, jend
        t(jl, jk) = c * jk
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
    t(jk) = c*jk
    q(jk) = q(jk - 1) + t(jk)*c
  END DO
"""
    assert exp_loop_nest in source.to_fortran()
    # OMNI automatically lower-cases everything
    assert "q(nz) = q(nz)*c" in source.to_fortran().lower()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_extract_sca_nested_level_zero(frontend):
    """
    Test nested subroutine call outside the vertical loop.
    """
    source = SourceFile.from_source(source="""
  SUBROUTINE compute_column(jstart, jend, nlon, nz, q, t)
    INTEGER, INTENT(IN)   :: jstart, jend  ! Explicit iteration bounds of the kernel
    INTEGER, INTENT(IN)   :: nlon, nz      ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jk
    REAL :: c

    c = 5.345
    call compute_level_zero(jstart, jend, nlon, nz, c, q(:,:), t)

    DO JL = JSTART, JEND
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
""", frontend=frontend)

    level_zero = SourceFile.from_source(source="""
  SUBROUTINE compute_level_zero(jstart, jend, nlon, nz, c, q, t)
    INTEGER, INTENT(IN)   :: jstart, jend
    INTEGER, INTENT(IN)   :: nlon, nz
    REAL, INTENT(IN) :: c
    REAL, INTENT(INOUT)   :: t(nlon,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jl, jk

    DO jk = 2, nz
      DO jl = jstart, jend
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1)  + t(jl, jk) * c
      END DO
    END DO
  END SUBROUTINE compute_level_zero
""", frontend=frontend)

    source['compute_column'].enrich_calls(routines=level_zero.all_subroutines)

    # Apply single-column extraction trasnformation in topological order
    dimension = Dimension(name='nlon', variable='jl', iteration=('jstart', 'jend'), aliases=[])
    sca_transform = ExtractSCATransformation(dimension=dimension)
    source.apply(sca_transform, role='kernel', targets=['compute_utility'])
    level_zero.apply(sca_transform, role='kernel')

    exp_loop_nest = """
  DO jk=2,nz
    t(jk) = c*jk
    q(jk) = q(jk - 1) + t(jk)*c
  END DO
"""
    assert exp_loop_nest in level_zero.to_fortran()

    # OMNI automatically lower-cases everything
    assert "call compute_level_zero(nz, c, q(:), t(:))" in source.to_fortran().lower()
    assert "q(nz) = q(nz)*c" in source.to_fortran().lower()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_extract_sca_nested_level_one(frontend):
    """
    Test nested subroutine call inside vertical loop.
    """
    source = SourceFile.from_source(source="""
  SUBROUTINE compute_column(jstart, jend, nlon, nz, q, t)
    INTEGER, INTENT(IN)   :: jstart, jend  ! Explicit iteration bounds of the kernel
    INTEGER, INTENT(IN)   :: nlon, nz      ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      call compute_level_one(jstart, jend, nlon, jk, c, q(:,jk), t(:,jk))
    END DO

    DO JL = JSTART, JEND
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
""", frontend=frontend)

    level_one = SourceFile.from_source(source="""
  SUBROUTINE compute_level_one(jstart, jend, nlon, jk, c, q, t)
    INTEGER, INTENT(IN)   :: jstart, jend
    INTEGER, INTENT(IN)   :: nlon, jk
    REAL, INTENT(IN) :: c
    REAL, INTENT(INOUT)   :: t(nlon,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jl, jk

    DO jl = jstart, jend
      t(jl, jk) = c * jk
      q(jl, jk) = q(jl, jk-1)  + t(jl, jk) * c
    END DO
  END SUBROUTINE compute_level_one
""", frontend=frontend)

    source['compute_column'].enrich_calls(routines=level_one.all_subroutines)

    # Apply single-column extraction trasnformation in topological order
    dimension = Dimension(name='nlon', variable='jl', iteration=('jstart', 'jend'), aliases=[])
    sca_transform = ExtractSCATransformation(dimension=dimension)
    source.apply(sca_transform, role='kernel', targets=['compute_utility'])
    level_one.apply(sca_transform, role='kernel')

    exp_loop_nest = """
  t(jk) = c*jk
  q(jk) = q(jk - 1) + t(jk)*c
"""
    assert exp_loop_nest in level_one.to_fortran()

    # OMNI automatically lower-cases everything
    assert "call compute_level_one(jk, c, q(jk), t(jk))" in source.to_fortran().lower()
    assert "q(nz) = q(nz)*c" in source.to_fortran().lower()
