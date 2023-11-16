# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Sourcefile, Dimension
from conftest import available_frontends
from transformations import ExtractSCATransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('jstart', 'jend'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_sca_horizontal(frontend, horizontal):
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
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1)  + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = 1, NLON
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    source = Sourcefile.from_source(fcode, frontend=frontend)
    sca_transform = ExtractSCATransformation(horizontal=horizontal)
    source['compute_column'].apply(sca_transform, role='kernel')

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


@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_sca_iteration(frontend, horizontal):
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
    source = Sourcefile.from_source(fcode, frontend=frontend)
    sca_transform = ExtractSCATransformation(horizontal=horizontal)
    source['compute_column'].apply(sca_transform, role='kernel')

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


@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_sca_nested_level_zero(frontend, horizontal):
    """
    Test nested subroutine call outside the vertical loop.
    """
    source = Sourcefile.from_source(source="""
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

    level_zero = Sourcefile.from_source(source="""
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

    source['compute_column'].enrich(level_zero.all_subroutines)

    # Apply single-column extraction trasnformation in topological order
    sca_transform = ExtractSCATransformation(horizontal=horizontal)
    source['compute_column'].apply(sca_transform, role='kernel', targets=['compute_utility'])
    level_zero['compute_level_zero'].apply(sca_transform, role='kernel')

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


@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_sca_nested_level_one(frontend, horizontal):
    """
    Test nested subroutine call inside vertical loop.
    """
    source = Sourcefile.from_source(source="""
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

    level_one = Sourcefile.from_source(source="""
  SUBROUTINE compute_level_one(jstart, jend, nlon, jk, c, q, t)
    INTEGER, INTENT(IN)   :: jstart, jend
    INTEGER, INTENT(IN)   :: nlon, jk
    REAL, INTENT(IN) :: c
    REAL, INTENT(INOUT)   :: t(nlon,nz)
    REAL, INTENT(INOUT)   :: q(nlon,nz)
    INTEGER :: jl

    DO jl = jstart, jend
      t(jl, jk) = c * jk
      q(jl, jk) = q(jl, jk-1)  + t(jl, jk) * c
    END DO
  END SUBROUTINE compute_level_one
""", frontend=frontend)

    source['compute_column'].enrich(level_one.all_subroutines)

    # Apply single-column extraction trasnformation in topological order
    sca_transform = ExtractSCATransformation(horizontal=horizontal)
    source['compute_column'].apply(sca_transform, role='kernel', targets=['compute_utility'])
    level_one['compute_level_one'].apply(sca_transform, role='kernel')

    exp_loop_nest = """
  t(jk) = c*jk
  q(jk) = q(jk - 1) + t(jk)*c
"""
    assert exp_loop_nest in level_one.to_fortran()

    # OMNI automatically lower-cases everything
    assert "call compute_level_one(jk, c, q(jk), t(jk))" in source.to_fortran().lower()
    assert "q(nz) = q(nz)*c" in source.to_fortran().lower()
