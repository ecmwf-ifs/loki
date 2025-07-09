# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Module, Subroutine, Dimension
from loki.jit_build import jit_compile, jit_compile_lib, Builder, Obj
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes, FindVariables

from loki.transformations.array_indexing.vector_notation import (
    resolve_vector_notation, resolve_vector_dimension,
    remove_explicit_array_dimensions, add_explicit_array_dimensions
)


@pytest.fixture(scope='function', name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_resolve_vector_notation(tmp_path, frontend):
    """
    Apply and test resolve vector notation utility.
    """
    fcode = """
subroutine transform_resolve_vector_notation(ret1, ret2)
  implicit none
  integer, parameter :: param1 = 3
  integer, parameter :: param2 = 5
  integer, intent(out) :: ret1(param1, param1), ret2(param1, param2)
  integer :: tmp, jk

  ret1(:, :) = 11
  ret2(:, :) = 42

end subroutine transform_resolve_vector_notation
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine)

    loops = FindNodes(ir.Loop).visit(routine.body)
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]

    assert len(loops) == 4
    assert loops[0].variable == 'i_ret1_1'
    assert loops[0].bounds == '1:param1' if frontend != OMNI else '1:3:1'
    assert loops[1].variable == 'i_ret1_0'
    assert loops[1].bounds == '1:param1' if frontend != OMNI else '1:3:1'
    assert loops[2].variable == 'i_ret2_1'
    assert loops[2].bounds == '1:param2' if frontend != OMNI else '1:5:1'
    assert loops[3].variable == 'i_ret2_0'
    assert loops[3].bounds == '1:param1' if frontend != OMNI else '1:3:1'

    assert len(arrays) == 2
    assert arrays[0].dimensions == ('i_ret1_0', 'i_ret1_1')
    assert arrays[1].dimensions == ('i_ret2_0', 'i_ret2_1')

    ret1 = np.zeros(shape=(3, 3), order='F', dtype=np.int32)
    ret2 = np.zeros(shape=(3, 5), order='F', dtype=np.int32)

    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    function(ret1, ret2)

    assert np.all(ret1 == 11)
    assert np.all(ret2 == 42)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('kidia_loop', (True, False))
def test_transform_resolve_vector_notation_common_loops(tmp_path, frontend, kidia_loop):
    """
    Apply and test resolve vector notation utility with already
    available/appropriate loops.
    """
    fcode = f"""
subroutine transform_resolve_vector_notation_common_loops(scalar, vector, vector_2, matrix, n, m, l, kidia, kfdia)
  implicit none
  integer, intent(in) :: n, m, l, kidia, kfdia
  integer, intent(inout) :: scalar, vector(n), vector_2(n), matrix(l, n)
  integer :: tmp_scalar, tmp_vector(n, m), tmp_matrix(l, m, n), tmp_dummy(n, 0:4)
  integer :: jl, jk, jm

  tmp_dummy(:,:) = 0
  tmp_vector(:, 1) = tmp_dummy(:, 1)
  tmp_vector(:, :) = 0
  tmp_matrix(:, :, :) = 0
  matrix(:, :) = 0

  do jl={'kidia,kfdia' if kidia_loop else '1,n'}
    do jm=1,m
      tmp_vector(jl, jm) = scalar + jl
    end do
  end do

  do jm=1,m
    do jl=1,n
      scalar = jl
      vector(jl) = tmp_vector(jl, jm) + tmp_vector(jl, jm)

      do jk=1,l
        tmp_matrix(jk, jm, jl) = vector(jl) + jk
      end do
    end do
  end do


  do jk=1,l
    matrix(jk, :) = 0
    do jm=1,m
      do jl=1,n
        matrix(jk, jl) = tmp_matrix(jk, jm, jl)
      end do
    end do
  end do

  vector_2(:) = 1
  vector_2(kidia:kfdia) = 2

end subroutine transform_resolve_vector_notation_common_loops
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 3
    m = 2
    l = 3
    kidia = 1
    kfdia = n
    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    vector_2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    function(scalar, vector, vector_2, matrix, n, m, l, kidia, kfdia)

    assert scalar == 3
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

    resolve_vector_notation(routine)
    loops = FindNodes(ir.Loop).visit(routine.body)
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
    assert len(loops) == 21
    assert loops[0].variable == 'i_tmp_dummy_1' and loops[0].bounds == '0:4'
    assert loops[1].variable == 'jl' and loops[1].bounds == '1:n'
    assert loops[2].variable == 'jl' and loops[2].bounds == '1:n'
    assert loops[3].variable == 'jm' and loops[3].bounds == '1:m'
    assert loops[4].variable == 'jl' and loops[4].bounds == '1:n'
    assert loops[5].variable == 'jl' and loops[5].bounds == '1:n'
    assert loops[6].variable == 'jm' and loops[6].bounds == '1:m'
    assert loops[7].variable == 'jk' and loops[7].bounds == '1:l'
    assert loops[8].variable == 'jl' and loops[8].bounds == '1:n'
    assert loops[9].variable == 'jk' and loops[9].bounds == '1:l'
    assert loops[10].variable == 'jl'
    if kidia_loop:
        assert loops[10].bounds == 'kidia:kfdia'
    else:
        assert loops[10].bounds == '1:n'
    assert loops[11].variable == 'jm' and loops[11].bounds == '1:m'
    assert loops[12].variable == 'jm' and loops[12].bounds == '1:m'
    assert loops[13].variable == 'jl' and loops[13].bounds == '1:n'
    assert loops[14].variable == 'jk' and loops[14].bounds == '1:l'
    assert loops[15].variable == 'jk' and loops[15].bounds == '1:l'
    assert loops[16].variable == 'jl' and loops[16].bounds == '1:n'
    assert loops[17].variable == 'jm' and loops[17].bounds == '1:m'
    assert loops[18].variable == 'jl' and loops[18].bounds == '1:n'
    assert loops[19].variable == 'jl' and loops[19].bounds == '1:n'
    if kidia_loop:
        assert loops[20].variable == 'jl'
        assert loops[20].bounds == 'kidia:kfdia'
    else:
        assert loops[20].variable == 'i_vector_2_0'
        assert loops[20].bounds == 'kidia:kfdia'

    assert len(arrays) == 17
    assert arrays[0].name.lower() == 'tmp_dummy' and arrays[0].dimensions == ('jl', 'i_tmp_dummy_1')
    assert arrays[1].name.lower() == 'tmp_vector' and arrays[1].dimensions == ('jl', 1)
    assert arrays[2].name.lower() == 'tmp_dummy' and arrays[2].dimensions == ('jl', 1)
    assert arrays[3].name.lower() == 'tmp_vector' and arrays[3].dimensions == ('jl', 'jm')
    assert arrays[4].name.lower() == 'tmp_matrix' and arrays[4].dimensions == ('jk', 'jm', 'jl')
    assert arrays[15].name.lower() == 'vector_2' and arrays[15].dimensions == ('jl',)
    assert arrays[16].name.lower() == 'vector_2'
    if kidia_loop:
        assert arrays[16].dimensions == ('jl',)
    else:
        assert arrays[16].dimensions == ('i_vector_2_0',)

    # Test promoted routine
    resolved_filepath = tmp_path/(f'{routine.name}_resolved_{frontend}.f90')
    resolved_function = jit_compile(routine, filepath=resolved_filepath, objname=routine.name)

    n = 3
    m = 2
    l = 3
    kidia = 1
    kfdia = n
    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    vector_2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    resolved_function(scalar, vector, vector_2, matrix, n, m, l, kidia, kfdia)

    assert scalar == 3
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('calls_only', (False, True))
def test_transform_explicit_dimensions(tmp_path, frontend, builder, calls_only):
    """
    Test making dimensions of arrays explicit and undoing this,
    thus removing colon notation from array dimensions either for all
    or for arrays within (inline) calls only.
    """
    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b)
    use kernel_explicit_dimensions_mod, only: kernel_routine
    INTEGER, INTENT(IN)    :: nlon, nlev
    INTEGER, INTENT(INOUT) :: a(nlon,nlev)
    INTEGER, INTENT(INOUT)  :: b(nlon,nlev)

    call kernel_routine(nlon, a, b=b, nlev=nlev)

  END SUBROUTINE driver_routine
    """

    fcode_kernel = """
  module kernel_explicit_dimensions_mod
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE kernel_routine(nlon, a, b, nlev)
    INTEGER, INTENT(IN)    :: nlon, nlev
    INTEGER, INTENT(INOUT) :: a(nlon,nlev)
    INTEGER, INTENT(INOUT) :: b(nlon,nlev)

    A = MYADD(A, B=B)
  END SUBROUTINE kernel_routine

  PURE ELEMENTAL FUNCTION MYADD(A, B)
    INTEGER :: MYADD
    INTEGER, INTENT(IN) :: A, B

    MYADD = A + B
  END FUNCTION
  end module kernel_explicit_dimensions_mod
    """

    def init_arguments(nlon, nlev):
        a = 2*np.ones(shape=(nlon,nlev,), order='F', dtype=np.int32)
        b = 3*np.ones(shape=(nlon,nlev,), order='F', dtype=np.int32)
        return a, b

    kernel_module = Module.from_source(fcode_kernel, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path],
                                     definitions=[kernel_module])
    kernel = kernel_module.subroutines[0]

    # compile and test reference
    refname = f'ref_explicit_dims_{driver.name}_{frontend}'
    reference = jit_compile_lib([kernel_module, driver], path=tmp_path, name=refname, builder=builder)
    ref_function = reference.driver_routine

    nlon = 10
    nlev = 12
    a_ref, b_ref = init_arguments(nlon, nlev)
    ref_function(nlon, nlev, a_ref, b_ref)
    builder.clean()

    # add explicit array dimensions
    add_explicit_array_dimensions(driver)
    add_explicit_array_dimensions(kernel)
    kernel_call = FindNodes(ir.CallStatement).visit(driver.body)[0]
    kernel_call_array_args = [arg for arg in kernel_call.arguments if isinstance(arg, sym.Array)]
    assert all(len(arg.dimensions) == 2 for arg in kernel_call_array_args)

    # remove explicit array dimensions (possibly only for calls)
    remove_explicit_array_dimensions(driver, calls_only=calls_only)
    remove_explicit_array_dimensions(kernel, calls_only=calls_only)

    kernel_call = FindNodes(ir.CallStatement).visit(driver.body)[0]
    kernel_call_array_args = [arg for arg in kernel_call.arguments if isinstance(arg, sym.Array)]
    assert all(not arg.dimensions for arg in kernel_call_array_args)
    if calls_only:
        assignments = FindNodes(ir.Assignment).visit(kernel.body)
        assert len(assignments) == 1
        assert len(assignments[0].lhs.dimensions) == 2
        parameters = (assignments[0].rhs.parameters[0],)
        parameters += (assignments[0].rhs.kwarguments[0][1],)
        assert not parameters[0].dimensions
        assert not parameters[1].dimensions
    else:
        kernel_arrays = FindVariables().visit(kernel.body)
        assert all(not arr.dimensions for arr in kernel_arrays)

    # compile and test the resulting code
    testname = f'test_explicit_dims_{"calls_only_" if calls_only else ""}_{driver.name}_{frontend}'
    test = jit_compile_lib([kernel_module, driver], path=tmp_path, name=testname, builder=builder)
    test_function = test.driver_routine

    a_test, b_test = init_arguments(nlon, nlev)
    test_function(nlon, nlev, a_test, b_test)

    # check whether reference and flattened variant(s) produce same result
    assert (a_test == a_ref).all()
    assert (b_test == b_ref).all()

    builder.clean()


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_dimension(frontend):
    """ Test vector resolution utility for a single dimension. """

    fcode = """
subroutine kernel(start, end, nlon, nlev, z, work, play, sleep, repeat)
  integer, intent(in) :: start, end, nlon, nlev
  real, intent(in) :: z
  real, intent(out) :: work(nlon), play(nlon, nlev), sleep(nlev,nlev), repeat(nlev,nlon)
  integer :: jl
  real :: work_maxval

  work(start:end) = 0.
  work_maxval = maxval(work(start:end))

  play(:,1:nlev) = 42.
  sleep(:, :) = z * z * z
  repeat(:,start:end) = 6.66
end subroutine kernel
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    horizontal = Dimension(name='horizontal', index='jl', lower='start', upper='end')
    resolve_vector_dimension(routine, dimension=horizontal)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 2

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 5

    # Check that the first expression has been wrapped
    assert assigns[0] in loops[0].body
    assert assigns[0].lhs == 'work(jl)'

    # Ensure that none of the other sections has been wrapped
    assert not assigns[1] in loops[0].body
    assert not assigns[1] in loops[1].body
    assert 'maxval' == assigns[1].rhs.name.lower()
    assert 'start:end' in assigns[1].rhs.parameters[0].dimensions

    assert not assigns[2] in loops[0].body
    assert not assigns[2] in loops[1].body
    assert assigns[2].lhs == 'play(:,1:nlev)'

    assert not assigns[3] in loops[0].body
    assert not assigns[3] in loops[1].body
    assert assigns[3].lhs == 'sleep(:,:)'

    # Check that the last expression has been partially wrapped
    assert assigns[4] in loops[1].body
    assert assigns[4].lhs == 'repeat(:,jl)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_masked_statements(frontend):
    """
    Test resolving of masked statements in kernel.
    """

    fcode = """
subroutine test_resolve_where(start, end, nlon, nz, q, t)
  INTEGER, INTENT(IN) :: start, end  ! Iteration indices
  INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
  REAL, INTENT(INOUT) :: t(nlon,nz)
  REAL, INTENT(INOUT) :: q(nlon,nz)
  INTEGER :: jk

  DO jk = 2, nz
    WHERE (q(start:end, jk) > 1.234)
      q(start:end, jk) = q(start:end, jk-1) + t(start:end, jk)
    ELSEWHERE
      q(start:end, jk) = t(start:end, jk)
    END WHERE
  END DO
end subroutine test_resolve_where
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    horizontal = Dimension(
        name='horizontal', index='jl', lower='start', upper='end'
    )
    resolve_vector_dimension(routine, dimension=horizontal)

    # Ensure horizontal loop variable has been declared
    assert 'jl' in routine.variables

    # Ensure we have three loops in the kernel,
    # horizontal loops should be nested within vertical
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 2
    assert loops[1] in FindNodes(ir.Loop).visit(loops[0].body)
    assert loops[1].variable == 'jl'
    assert loops[1].bounds == 'start:end'
    assert loops[0].variable == 'jk'
    assert loops[0].bounds == '2:nz'

    # Ensure that the respective conditional has been inserted correctly
    conds = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds) == 1
    assert conds[0] in FindNodes(ir.Conditional).visit(loops[1])
    assert conds[0].condition == 'q(jl, jk) > 1.234'

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0] in conds[0].body
    assert assigns[0].lhs == 'q(jl, jk)' and assigns[0].rhs == 'q(jl, jk - 1) + t(jl, jk)'
    assert assigns[1] in conds[0].else_body
    assert assigns[1].lhs == 'q(jl, jk)' and assigns[1].rhs == 't(jl, jk)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_masked_inferred_bounds(frontend):
    """ Test the resolution of WHERE stmts with inferred bounds """

    fcode = """
subroutine test_masked_inferred(n, m, x, y, z)
  implicit none
  integer, intent(in) :: n, m
  real(kind=8), intent(inout) :: x(n), y(n), z(m)
  integer :: i

  do i=1,n
    x(i) = i
  end do
  y(:) = 0.0
  z(:) = 0.0

  where( (x > 5.0) )
    x = y
  end where
end subroutine test_masked_inferred
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    dim = Dimension(name='n', index='i', lower='1', upper='n')
    resolve_vector_dimension(
        routine, dimension=dim, derive_qualified_ranges=True
    )

    # Check only assignments over ``n`` have been resolved
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 4
    assert assigns[0].lhs == 'x(i)' and assigns[0].rhs == 'i'
    assert assigns[1].lhs == 'y(i)' and assigns[1].rhs == '0.0'
    assert assigns[2].lhs == 'z(1:m)' and assigns[2].rhs == '0.0'
    assert assigns[3].lhs == 'x(i)' and assigns[3].rhs == 'y(i)'

    # Check the WHERE has been resolved to IF
    conds = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds) == 1
    assert conds[0].condition == 'x(i) > 5.0'
    assert assigns[3] in conds[0].body

    # Check that new loops have been inserted
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 3
    assert assigns[0] in loops[0].body
    assert assigns[1] in loops[1].body
    assert conds[0] in loops[2].body
