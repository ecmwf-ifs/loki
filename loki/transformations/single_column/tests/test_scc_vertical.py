# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Dimension
from loki.frontend import available_frontends
from loki.ir import FindNodes, Loop
from loki.expression import FindVariables
from loki.transformations.single_column import SCCFuseVerticalLoops


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',)
    )

@pytest.fixture(scope='module', name='horizontal_bounds_aliases')
def fixture_horizontal_bounds_aliases():
    return Dimension(
        name='horizontal_bounds_aliases', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',),
        bounds_aliases=('bnds%start', 'bnds%end')
    )

@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk', aliases=('nlev',))

@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.mark.parametrize('frontend', available_frontends())
def test_simple_scc_fuse_verticals_transformation(frontend, horizontal, vertical):
    """
    Test simple example of vertical loop fusion and demotion of temporaries.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: temp_t(nlon, nz)
    REAL :: temp_q(nlon, nz)
    INTEGER :: jl, JK
    REAL :: c

    c = 5.345
    !$loki loop-fusion group(1)
    DO JK = 1, nz
      DO jl = start, end
        temp_t(jl, jk) = c
        temp_q(jl, JK) = c
      END DO
    END DO

    !$loki loop-fusion group(1)
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = temp_t(jl, jk) * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * temp_q(jl, jk)
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 5

    SCCFuseVerticalLoops(vertical=vertical).apply(kernel, role='kernel')

    # Ensure the two vertical loops are fused
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 4
    assert kernel_loops[0].variable.name.lower() == 'jk'
    assert kernel_loops[-1].variable.name.lower() == 'jl'
    assert len([loop for loop in kernel_loops if loop.variable.name.lower() == 'jk']) == 1
    kernel_var_map = kernel.variable_map
    assert kernel_var_map['temp_t'].shape == (horizontal.size,)
    assert kernel_var_map['temp_q'].shape == (horizontal.size,)
    kernel_vars = [var for var in FindVariables().visit(kernel.body) if var.name.lower() in ['temp_t', 'temp_q']]
    for var in kernel_vars:
        assert var.shape == (horizontal.size,)
        assert var.dimensions == (horizontal.index,)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('ignore', (False, True))
def test_scc_fuse_verticals_transformation(frontend, horizontal, vertical, ignore):
    """
    Test somewhat more sophisticated example of vertical loop fusion
    and demotion of temporaries.
    """

    fcode_kernel = f"""
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL :: temp_t(nlon, nz)
    REAL :: temp_t2(nlon, nz)
    REAL :: temp_q(nlon, nz)
    REAL :: temp_q2(nlon, nz)
    REAL :: temp_cld(nlon, nz, 5)
    INTEGER :: jl, jk, jm
    REAL :: c

    {'!$loki k-caching ignore(temp_q2)' if ignore else ''}

    c = 5.345
    !$loki loop-fusion group(1-init)
    DO jk = 1, nz
      DO jl = start, end
        temp_t(jl, jk) = c
        temp_q(jl, jk) = c
        temp_t2(jl, jk) = 2*c
      END DO
    END DO

    !$loki loop-fusion group(1)
    !$loki loop-interchange
    DO jm=1,5
      DO jk = 1, nz
        DO jl = start, end
          temp_cld(jl, jk, jm) = 3.1415
        END DO
      END DO
    END DO

    DO jl = start, end
      q(jl, jk) = 0.
    END DO

    !$loki loop-fusion group(1) insert
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = temp_t(jl, jk) * temp_t2(jl, jk-1) * temp_cld(jl, jk, 1)
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * temp_q(jl, jk)
      END DO
    END DO

    CALL nested_kernel(start, end, nlon, nz, q)

    !$loki loop-fusion group(2)
    DO jk = 2, nz
      DO jl = start, end
        temp_q2(jl, jk) = 3.1415
      END DO
    END DO

    !$loki loop-fusion group(2)
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = t(jl, jk) + 3.1415
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * temp_q(jl, jk) + temp_q2(jl, jk)
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""


    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 13
    SCCFuseVerticalLoops(vertical=vertical).apply(kernel, role='kernel')

    # Ensure the two vertical loops are fused
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 12
    vertical_loops = [loop for loop in kernel_loops if loop.variable.name.lower() == vertical.index]
    assert len(vertical_loops) == 3

    shape1D = (horizontal.size,)
    shape2D = (horizontal.size, vertical.size)
    dimension1D = (horizontal.index,)
    dimension2D = (horizontal.index,vertical.index)
    dimension2DI1 = (horizontal.index, f'{vertical.index}-1')

    vertical_loop_0_vars = FindVariables().visit(vertical_loops[0].body)
    vertical_loop_0_var_names = [var.name.lower() for var in vertical_loop_0_vars]
    vertical_loop_0_var_dict = dict(zip(vertical_loop_0_var_names, vertical_loop_0_vars))
    assert 'temp_t2' in vertical_loop_0_var_names
    assert 'temp_t' not in vertical_loop_0_var_names
    assert 'temp_q' not in vertical_loop_0_var_names
    assert 'temp_q2' not in vertical_loop_0_var_names
    assert 'temp_cld' not in vertical_loop_0_var_names
    assert vertical_loop_0_var_dict['temp_t2'].shape == shape2D
    assert vertical_loop_0_var_dict['temp_t2'].dimensions == dimension2D

    vertical_loop_1_vars = FindVariables().visit(vertical_loops[1].body)
    vertical_loop_1_var_names = [var.name.lower() for var in vertical_loop_1_vars]
    vertical_loop_1_var_dict = dict(zip(vertical_loop_1_var_names, vertical_loop_1_vars))
    assert 'temp_t2' in vertical_loop_1_var_names
    assert 'temp_t' in vertical_loop_1_var_names
    assert 'temp_q' in vertical_loop_1_var_names
    assert 'temp_q2' not in vertical_loop_1_vars
    assert 'temp_cld' in vertical_loop_1_var_names
    assert vertical_loop_1_var_dict['temp_t2'].shape == shape2D
    assert vertical_loop_1_var_dict['temp_t2'].dimensions == dimension2DI1
    assert vertical_loop_1_var_dict['temp_t'].shape == shape1D
    assert vertical_loop_1_var_dict['temp_t'].dimensions == dimension1D
    assert vertical_loop_1_var_dict['temp_q'].shape == shape2D
    assert vertical_loop_1_var_dict['temp_q'].dimensions == dimension2D
    assert vertical_loop_1_var_dict['temp_cld'].shape == shape1D + (5,)
    assert vertical_loop_1_var_dict['temp_cld'].dimensions in (dimension1D + (1,), dimension1D + ('jm',))

    vertical_loop_2_vars = FindVariables().visit(vertical_loops[2].body)
    vertical_loop_2_var_names = [var.name.lower() for var in vertical_loop_2_vars]
    vertical_loop_2_var_dict = dict(zip(vertical_loop_2_var_names, vertical_loop_2_vars))
    assert 'temp_t2' not in vertical_loop_2_var_names
    assert 'temp_t' not in vertical_loop_2_var_names
    assert 'temp_q' in vertical_loop_2_var_names
    assert 'temp_q2' in vertical_loop_2_var_names
    assert 'temp_cld' not in vertical_loop_2_var_names
    assert vertical_loop_2_var_dict['temp_q'].shape == shape2D
    assert vertical_loop_2_var_dict['temp_q'].dimensions == dimension2D
    assert vertical_loop_2_var_dict['temp_q2'].shape == shape2D if ignore else shape1D
    assert vertical_loop_2_var_dict['temp_q2'].dimensions == dimension2D if ignore else dimension1D

    kernel_var_map = kernel.variable_map
    assert kernel_var_map['temp_t'].shape == shape1D
    assert kernel_var_map['temp_t2'].shape == shape2D
    assert kernel_var_map['temp_q'].shape == shape2D
    assert kernel_var_map['temp_q2'].shape == shape2D if ignore else shape1D
