# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from shutil import rmtree
import pytest
import numpy as np

from loki import Module, Subroutine, fgen
from loki.build import jit_compile, jit_compile_lib, clean_test, Builder
from loki.expression import symbols as sym, FindVariables
from loki.frontend import available_frontends
from loki.ir import FindNodes, CallStatement
from loki.tools import gettempdir

from loki.transformations.array_indexing import (
    promote_variables, demote_variables, normalize_range_indexing,
    invert_array_indices, flatten_arrays,
    normalize_array_shape_and_access, shift_to_zero_indexing,
)
from loki.transformations.transpile import FortranCTransformation


@pytest.fixture(scope='function', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='function', name='tempdir')
def fixture_tempdir(request):
    basedir = gettempdir()/request.function.__name__
    basedir.mkdir(exist_ok=True)
    yield basedir
    if basedir.exists():
        rmtree(basedir)


@pytest.fixture(scope='function', name='builder')
def fixture_builder(here, tempdir):
    return Builder(source_dirs=tempdir, build_dir=tempdir)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variable_scalar(here, frontend):
    """
    Apply variable promotion for a single scalar variable.
    """
    fcode = """
subroutine transform_promote_variable_scalar(ret)
  implicit none
  integer, intent(out) :: ret
  integer :: tmp, jk

  ret = 0
  do jk=1,10
    tmp = jk
    ret = ret + tmp
  end do
end subroutine transform_promote_variable_scalar
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Test the original implementation
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    ret = function()
    assert ret == 55

    # Apply and test the transformation
    assert isinstance(routine.variable_map['tmp'], sym.Scalar)
    promote_variables(routine, ['TMP'], pos=0, index=routine.variable_map['JK'], size=sym.Literal(10))
    assert isinstance(routine.variable_map['tmp'], sym.Array)
    assert routine.variable_map['tmp'].shape == (sym.Literal(10),)

    promoted_filepath = here/(f'{routine.name}_promoted_{frontend}.f90')
    promoted_function = jit_compile(routine, filepath=promoted_filepath, objname=routine.name)
    ret = promoted_function()
    assert ret == 55

    clean_test(filepath)
    clean_test(promoted_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variables(here, frontend):
    """
    Apply variable promotion for scalar and array variables.
    """
    fcode = """
subroutine transform_promote_variables(scalar, vector, n)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: scalar, vector(n)
  integer :: tmp_scalar, tmp_vector(n), tmp_matrix(n,n)
  integer :: jl, jk

  do jl=1,n
    ! a bit of a hack to create initialized meaningful output
    tmp_vector(:) = 0
  end do

  do jl=1,n
    tmp_scalar = jl
    tmp_vector(jl) = jl

    do jk=1,n
      tmp_matrix(jk, jl) = jl + jk
    end do
  end do

  scalar = 0
  do jl=1,n
    scalar = scalar + tmp_scalar
    vector = tmp_matrix(:,jl) + tmp_vector(:)
  end do
end subroutine transform_promote_variables
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_range_indexing(routine) # Fix OMNI nonsense

    # Test the original implementation
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 10
    scalar = np.zeros(shape=(1,), order='F', dtype=np.int32)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    function(scalar, vector, n)
    assert scalar == n*n
    assert np.all(vector == np.array(list(range(1, 2*n+1, 2)), order='F', dtype=np.int32) + n + 1)

    # Verify dimensions before promotion
    assert isinstance(routine.variable_map['tmp_scalar'], sym.Scalar)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])

    # Promote scalar and vector and verify dimensions
    promote_variables(routine, ['tmp_scalar', 'tmp_vector'], pos=-1, index=routine.variable_map['JL'],
                      size=routine.variable_map['n'])

    assert isinstance(routine.variable_map['tmp_scalar'], sym.Array)
    assert routine.variable_map['tmp_scalar'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'], routine.variable_map['n'])
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])

    # Promote matrix and verify dimensions
    promote_variables(routine, ['tmp_matrix'], pos=1, index=routine.variable_map['JL'],
                      size=routine.variable_map['n'])

    assert isinstance(routine.variable_map['tmp_scalar'], sym.Array)
    assert routine.variable_map['tmp_scalar'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'], routine.variable_map['n'])
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], ) * 3

    # Test promoted routine
    promoted_filepath = here/(f'{routine.name}_promoted_{frontend}.f90')
    promoted_function = jit_compile(routine, filepath=promoted_filepath, objname=routine.name)

    scalar = np.zeros(shape=(1,), order='F', dtype=np.int32)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    promoted_function(scalar, vector, n)
    assert scalar == n*(n+1)//2
    assert np.all(vector[:-1] == np.array(list(range(n + 1, 2*n)), order='F', dtype=np.int32))
    assert vector[-1] == 3*n

    clean_test(filepath)
    clean_test(promoted_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_demote_variables(here, frontend):
    """
    Apply variable demotion to a range of array variables.
    """
    fcode = """
subroutine transform_demote_variables(scalar, vector, matrix, n, m)
  implicit none
  integer, intent(in) :: n, m
  integer, intent(inout) :: scalar, vector(n), matrix(n, n)
  integer :: tmp_scalar, tmp_vector(n, m), tmp_matrix(n, m, n)
  integer :: jl, jk, jm

  do jl=1,n
    do jm=1,m
      tmp_vector(jl, jm) = scalar + jl
    end do
  end do

  do jm=1,m
    do jl=1,n
      scalar = jl
      vector(jl) = tmp_vector(jl, jm) + tmp_vector(jl, jm)

      do jk=1,n
        tmp_matrix(jk, jm, jl) = vector(jl) + jk
      end do
    end do
  end do

  do jk=1,n
    do jm=1,m
      do jl=1,n
        matrix(jk, jl) = tmp_matrix(jk, jm, jl)
      end do
    end do
  end do
end subroutine transform_demote_variables
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_range_indexing(routine) # Fix OMNI nonsense

    # Test the original implementation
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 3
    m = 2
    scalar = np.zeros(shape=(1,), order='F', dtype=np.int32)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    function(scalar, vector, matrix, n, m)

    assert all(scalar == 3)
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

    # Do the variable demotion for all relevant array variables
    demote_variables(routine, ['tmp_vector', 'tmp_matrix'], ['m'])

    assert isinstance(routine.variable_map['scalar'], sym.Scalar)
    assert isinstance(routine.variable_map['vector'], sym.Array)
    assert routine.variable_map['vector'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['tmp_vector'], sym.Array)
    assert routine.variable_map['tmp_vector'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['matrix'], sym.Array)
    assert routine.variable_map['matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])
    assert isinstance(routine.variable_map['tmp_matrix'], sym.Array)
    assert routine.variable_map['tmp_matrix'].shape == (routine.variable_map['n'], routine.variable_map['n'])

    # Test promoted routine
    demoted_filepath = here/(f'{routine.name}_demoted_{frontend}.f90')
    demoted_function = jit_compile(routine, filepath=demoted_filepath, objname=routine.name)

    n = 3
    m = 2
    scalar = np.zeros(shape=(1,), order='F', dtype=np.int32)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    demoted_function(scalar, vector, matrix, n, m)

    assert all(scalar == 3)
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

    # Test that the transformation doesn't fail for scalar arguments and leaves the
    # IR unchanged
    demoted_fcode = routine.to_fortran()
    demote_variables(routine, ['jl'], ['m'])
    assert routine.to_fortran() == demoted_fcode


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_demote_dimension_arguments(here, frontend):
    """
    Apply variable demotion to array arguments defined with DIMENSION
    keywords.
    """
    fcode = """
subroutine transform_demote_dimension_arguments(vec1, vec2, matrix, n, m)
  implicit none
  integer, intent(in) :: n, m
  integer, dimension(n), intent(inout) :: vec1, vec2
  integer, dimension(n, m), intent(inout) :: matrix
  integer, dimension(n) :: vec_tmp
  integer :: i, j

  do i=1,n
    do j=1,m
      vec_tmp(i) = vec1(i) + vec2(i)
      matrix(i, j) = matrix(i, j) + vec_tmp(i)
    end do
  end do
end subroutine transform_demote_dimension_arguments
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_range_indexing(routine) # Fix OMNI nonsense

    # Test the original implementation
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    assert isinstance(routine.variable_map['vec1'], sym.Array)
    assert routine.variable_map['vec1'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['vec2'], sym.Array)
    assert routine.variable_map['vec2'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['matrix'], sym.Array)
    assert routine.variable_map['matrix'].shape == (routine.variable_map['n'], routine.variable_map['m'])

    n = 3
    m = 2
    vec1 = np.zeros(shape=(n,), order='F', dtype=np.int32) + 3
    vec2 = np.zeros(shape=(n,), order='F', dtype=np.int32) + 2
    matrix = np.zeros(shape=(n, m), order='F', dtype=np.int32) + 1
    function(vec1, vec2, matrix, n, m)

    assert np.all(vec1 == 3) and np.sum(vec1) == 9
    assert np.all(vec2 == 2) and np.sum(vec2) == 6
    assert np.all(matrix == 6) and np.sum(matrix) == 36

    demote_variables(routine, ['vec1', 'vec_tmp', 'matrix'], ['n'])

    assert isinstance(routine.variable_map['vec1'], sym.Scalar)
    assert isinstance(routine.variable_map['vec2'], sym.Array)
    assert routine.variable_map['vec2'].shape == (routine.variable_map['n'],)
    assert isinstance(routine.variable_map['matrix'], sym.Array)
    assert routine.variable_map['matrix'].shape == (routine.variable_map['m'],)

    # Test promoted routine
    demoted_filepath = here/(f'{routine.name}_demoted_{frontend}.f90')
    demoted_function = jit_compile(routine, filepath=demoted_filepath, objname=routine.name)

    n = 3
    m = 2
    vec1 = np.zeros(shape=(1,), order='F', dtype=np.int32) + 3
    vec2 = np.zeros(shape=(n,), order='F', dtype=np.int32) + 2
    matrix = np.zeros(shape=(m, ), order='F', dtype=np.int32) + 1
    demoted_function(vec1, vec2, matrix, n, m)

    assert np.all(vec1 == 3) and np.sum(vec1) == 3
    assert np.all(vec2 == 2) and np.sum(vec2) == 6
    assert np.all(matrix == 16) and np.sum(matrix) == 32

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('start_index', (0, 1, 5))
def test_transform_normalize_array_shape_and_access(here, frontend, start_index):
    """
    Test normalization of array shape and access, thus changing arrays with start 
    index different than "1" to have start index "1".

    E.g., ``x1(5:len)`` -> ```x1(1:len-4)`` 
    """
    fcode = f"""
    module transform_normalize_array_shape_and_access_mod
    implicit none
    
    contains

    subroutine transform_normalize_array_shape_and_access(x1, x2, x3, x4, assumed_x1, l1, l2, l3, l4)
        ! use nested_routine_mod, only : nested_routine
        implicit none
        integer :: i1, i2, i3, i4, c1, c2, c3, c4
        integer, intent(in) :: l1, l2, l3, l4
        integer, intent(inout) :: x1({start_index}:l1+{start_index}-1)
        integer, intent(inout) :: x2({start_index}:l2+{start_index}-1, &
         & {start_index}:l1+{start_index}-1)
        integer, intent(inout) :: x3({start_index}:l3+{start_index}-1, &
         & {start_index}:l2+{start_index}-1, {start_index}:l1+{start_index}-1)
        integer, intent(inout) :: x4({start_index}:l4+{start_index}-1, &
         & {start_index}:l3+{start_index}-1, {start_index}:l2+{start_index}-1, &
         & {start_index}:l1+{start_index}-1)
        integer, intent(inout) :: assumed_x1(l1)
        c1 = 1
        c2 = 1
        c3 = 1
        c4 = 1
        do i1=1,l1
            assumed_x1(i1) = c1
            call nested_routine(assumed_x1, l1, c1)
        end do
        x1({start_index}:l4+{start_index}-1) = 0
        do i1={start_index},l1+{start_index}-1
            x1(i1) = c1
            do i2={start_index},l2+{start_index}-1
                x2(i2, i1) = c2*10 + c1
                do i3={start_index},l3+{start_index}-1
                    x3(i3, i2, i1) = c3*100 + c2*10 + c1
                    do i4={start_index},l4+{start_index}-1
                        x4(i4, i3, i2, i1) = c4*1000 + c3*100 + c2*10 + c1
                        c4 = c4 + 1
                    end do
                    c3 = c3 + 1
                end do
                c2 = c2 + 1
            end do
            c1 = c1 + 1
        end do
    end subroutine transform_normalize_array_shape_and_access

    subroutine nested_routine(nested_x1, l1, c1)
        implicit none
        integer, intent(in) :: l1, c1
        integer, intent(inout) :: nested_x1(:)
        integer :: i1
        do i1=1,l1
            nested_x1(i1) = c1
        end do
    end subroutine nested_routine

    end module transform_normalize_array_shape_and_access_mod
    """

    def init_arguments(l1, l2, l3, l4):
        x1 = np.zeros(shape=(l1,), order='F', dtype=np.int32)
        assumed_x1 = np.zeros(shape=(l1,), order='F', dtype=np.int32)
        x2 = np.zeros(shape=(l2,l1,), order='F', dtype=np.int32)
        x3 = np.zeros(shape=(l3,l2,l1,), order='F', dtype=np.int32)
        x4 = np.zeros(shape=(l4,l3,l2,l1,), order='F', dtype=np.int32)
        return x1, x2, x3, x4, assumed_x1

    def validate_routine(routine):
        arrays = [var for var in FindVariables().visit(routine.body) if isinstance(var, sym.Array)]
        for arr in arrays:
            assert all(not isinstance(shape, sym.RangeIndex) for shape in arr.shape)

    l1 = 2
    l2 = 3
    l3 = 4
    l4 = 5
    module = Module.from_source(fcode, frontend=frontend)
    for routine in module.routines:
        normalize_range_indexing(routine) # Fix OMNI nonsense
    filepath = here/(f'transform_normalize_array_shape_and_access_{frontend}.f90')
    # compile and test "original" module/function
    mod = jit_compile(module, filepath=filepath, objname='transform_normalize_array_shape_and_access_mod')
    function = getattr(mod, 'transform_normalize_array_shape_and_access')
    orig_x1, orig_x2, orig_x3, orig_x4, orig_assumed_x1 = init_arguments(l1, l2, l3, l4)
    function(orig_x1, orig_x2, orig_x3, orig_x4, orig_assumed_x1, l1, l2, l3, l4)
    clean_test(filepath)

    # apply `normalize_array_shape_and_access`
    for routine in module.routines:
        normalize_array_shape_and_access(routine)

    filepath = here/(f'transform_normalize_array_shape_and_access_normalized_{frontend}.f90')
    # compile and test "normalized" module/function
    mod = jit_compile(module, filepath=filepath, objname='transform_normalize_array_shape_and_access_mod')
    function = getattr(mod, 'transform_normalize_array_shape_and_access')
    x1, x2, x3, x4, assumed_x1 = init_arguments(l1, l2, l3, l4)
    function(x1, x2, x3, x4, assumed_x1, l1, l2, l3, l4)
    clean_test(filepath)
    # validate the routine "transform_normalize_array_shape_and_access"
    validate_routine(module.subroutines[0])
    # validate the nested routine to see whether the assumed size array got correctly handled
    assert module.subroutines[1].variable_map['nested_x1'] == 'nested_x1(:)'

    # check whether results generated by the "original" and "normalized" version agree
    assert (x1 == orig_x1).all()
    assert (assumed_x1 == orig_assumed_x1).all()
    assert (x2 == orig_x2).all()
    assert (x3 == orig_x3).all()
    assert (x4 == orig_x4).all()


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('start_index', (0, 1, 5))
def test_transform_flatten_arrays(here, frontend, builder, start_index):
    """
    Test flattening or arrays, meaning converting multi-dimensional
    arrays to one-dimensional arrays including corresponding
    index arithmetic.
    """
    fcode = f"""
    subroutine transform_flatten_arrays(x1, x2, x3, x4, l1, l2, l3, l4)
        implicit none
        integer :: i1, i2, i3, i4, c1, c2, c3, c4
        integer, intent(in) :: l1, l2, l3, l4
        integer, intent(inout) :: x1({start_index}:l1+{start_index}-1)
        integer, intent(inout) :: x2({start_index}:l2+{start_index}-1, &
         & {start_index}:l1+{start_index}-1)
        integer, intent(inout) :: x3({start_index}:l3+{start_index}-1, &
         & {start_index}:l2+{start_index}-1, {start_index}:l1+{start_index}-1)
        integer, intent(inout) :: x4({start_index}:l4+{start_index}-1, &
         & {start_index}:l3+{start_index}-1, {start_index}:l2+{start_index}-1, &
         & {start_index}:l1+{start_index}-1)
        c1 = 1
        c2 = 1
        c3 = 1
        c4 = 1
        do i1={start_index},l1+{start_index}-1
            x1(i1) = c1
            do i2={start_index},l2+{start_index}-1
                x2(i2, i1) = c2*10 + c1
                do i3={start_index},l3+{start_index}-1
                    x3(i3, i2, i1) = c3*100 + c2*10 + c1
                    do i4={start_index},l4+{start_index}-1
                        x4(i4, i3, i2, i1) = c4*1000 + c3*100 + c2*10 + c1
                        c4 = c4 + 1
                    end do
                    c3 = c3 + 1
                end do
                c2 = c2 + 1
            end do
            c1 = c1 + 1
        end do

    end subroutine transform_flatten_arrays
    """
    def init_arguments(l1, l2, l3, l4, flattened=False):
        x1 = np.zeros(shape=(l1,), order='F', dtype=np.int32)
        x2 = np.zeros(shape=(l2*l1) if flattened else (l2,l1,), order='F', dtype=np.int32)
        x3 = np.zeros(shape=(l3*l2*l1) if flattened else (l3,l2,l1,), order='F', dtype=np.int32)
        x4 = np.zeros(shape=(l4*l3*l2*l1) if flattened else (l4,l3,l2,l1,), order='F', dtype=np.int32)
        return x1, x2, x3, x4

    def validate_routine(routine):
        arrays = [var for var in FindVariables().visit(routine.body) if isinstance(var, sym.Array)]
        assert all(len(arr.dimensions) == 1 for arr in arrays)
        assert all(len(arr.shape) == 1 for arr in arrays)

    l1 = 2
    l2 = 3
    l3 = 4
    l4 = 5
    # Test the original implementation
    routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_range_indexing(routine) # Fix OMNI nonsense
    filepath = here/(f'{routine.name}_{start_index}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    orig_x1, orig_x2, orig_x3, orig_x4 = init_arguments(l1, l2, l3, l4)
    function(orig_x1, orig_x2, orig_x3, orig_x4, l1, l2, l3, l4)
    clean_test(filepath)

    # Test flattening order='F'
    f_routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_array_shape_and_access(f_routine)
    normalize_range_indexing(f_routine) # Fix OMNI nonsense
    flatten_arrays(routine=f_routine, order='F', start_index=1)
    filepath = here/(f'{f_routine.name}_{start_index}_flattened_F_{frontend}.f90')
    function = jit_compile(f_routine, filepath=filepath, objname=routine.name)
    f_x1, f_x2, f_x3, f_x4 = init_arguments(l1, l2, l3, l4, flattened=True)
    function(f_x1, f_x2, f_x3, f_x4, l1, l2, l3, l4)
    validate_routine(f_routine)
    clean_test(filepath)

    assert (f_x1 == orig_x1.flatten(order='F')).all()
    assert (f_x2 == orig_x2.flatten(order='F')).all()
    assert (f_x3 == orig_x3.flatten(order='F')).all()
    assert (f_x4 == orig_x4.flatten(order='F')).all()

    # Test flattening order='C'
    c_routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_array_shape_and_access(c_routine)
    normalize_range_indexing(c_routine) # Fix OMNI nonsense
    invert_array_indices(c_routine)
    flatten_arrays(routine=c_routine, order='C', start_index=1)
    filepath = here/(f'{c_routine.name}_{start_index}_flattened_C_{frontend}.f90')
    function = jit_compile(c_routine, filepath=filepath, objname=routine.name)
    c_x1, c_x2, c_x3, c_x4 = init_arguments(l1, l2, l3, l4, flattened=True)
    function(c_x1, c_x2, c_x3, c_x4, l1, l2, l3, l4)
    validate_routine(c_routine)
    clean_test(filepath)

    assert f_routine.body == c_routine.body

    assert (c_x1 == orig_x1.flatten(order='F')).all()
    assert (c_x2 == orig_x2.flatten(order='F')).all()
    assert (c_x3 == orig_x3.flatten(order='F')).all()
    assert (c_x4 == orig_x4.flatten(order='F')).all()

    # Test C transpilation (which includes flattening)
    f2c_routine = Subroutine.from_source(fcode, frontend=frontend)
    f2c = FortranCTransformation()
    f2c.apply(source=f2c_routine, path=here)
    libname = f'fc_{f2c_routine.name}_{start_index}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transform_flatten_arrays_fc_mod.transform_flatten_arrays_fc
    f2c_x1, f2c_x2, f2c_x3, f2c_x4 = init_arguments(l1, l2, l3, l4, flattened=True)
    fc_function(f2c_x1, f2c_x2, f2c_x3, f2c_x4, l1, l2, l3, l4)
    validate_routine(c_routine)

    assert (f2c_x1 == orig_x1.flatten(order='F')).all()
    assert (f2c_x2 == orig_x2.flatten(order='F')).all()
    assert (f2c_x3 == orig_x3.flatten(order='F')).all()
    assert (f2c_x4 == orig_x4.flatten(order='F')).all()

    builder.clean()
    clean_test(filepath)
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('ignore', ((), ('i2',), ('i4', 'i1')))
def test_shift_to_zero_indexing(frontend, ignore):
    """
    Test shifting array dimensions to zero (or rather shift dimension `dim` 
    to `dim - 1`). This does not produce valid Fortran, but is part of the 
    F2C transpilation logic.
    """
    fcode = """
    subroutine transform_shift_indexing(x1, x2, x3, x4, l1, l2, l3, l4)
        implicit none
        integer :: i1, i2, i3, i4, c1, c2, c3, c4
        integer, intent(in) :: l1, l2, l3, l4
        integer, intent(inout) :: x1(l1)
        integer, intent(inout) :: x2(l2, l1)
        integer, intent(inout) :: x3(l3, l2, l1)
        integer, intent(inout) :: x4(l4, l3, l2, l1)
        c1 = 1
        c2 = 1
        c3 = 1
        c4 = 1
        do i1=1,l1
            x1(i1) = c1
            do i2=1,l2
                x2(i2, i1) = c2*10 + c1
                do i3=1,l3
                    x3(i3, i2, i1) = c3*100 + c2*10 + c1
                    do i4=1,l4
                        x4(i4, i3, i2, i1) = c4*1000 + c3*100 + c2*10 + c1
                        c4 = c4 + 1
                    end do
                    c3 = c3 + 1
                end do
                c2 = c2 + 1
            end do
            c1 = c1 + 1
        end do

    end subroutine transform_shift_indexing
    """

    expected_dims = {'x1': ('i1',), 'x2': ('i2', 'i1'),
            'x3': ('i3', 'i2', 'i1'), 'x4': ('i4', 'i3', 'i2', 'i1')}
    routine = Subroutine.from_source(fcode, frontend=frontend)
    arrays = [var for var in FindVariables().visit(routine.body) if isinstance(var, sym.Array)]
    for array in arrays:
        assert array.dimensions == expected_dims[array.name]

    shift_to_zero_indexing(routine, ignore=ignore)

    arrays = [var for var in FindVariables().visit(routine.body) if isinstance(var, sym.Array)]
    for array in arrays:
        dimensions = tuple(sym.Sum((sym.Scalar(name=dim), sym.Product((-1, sym.IntLiteral(1)))))
                if dim not in ignore else dim for dim in expected_dims[array.name])
        assert fgen(array.dimensions) == fgen(dimensions)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('explicit_dimensions', [True, False])
def test_transform_flatten_arrays_call(tempdir, frontend, builder, explicit_dimensions):
    """
    Test flattening or arrays, meaning converting multi-dimensional
    arrays to one-dimensional arrays including corresponding
    index arithmetic (for calls).
    """
    array_dims = '(:,:)' if explicit_dimensions else ''
    fcode_driver = f"""
  SUBROUTINE driver_routine(nlon, nlev, a, b)
    use kernel_mod, only: kernel_routine
    INTEGER, INTENT(IN)    :: nlon, nlev
    INTEGER, INTENT(INOUT) :: a(nlon,nlev)
    INTEGER, INTENT(INOUT)  :: b(nlon,nlev)

    call kernel_routine(nlon, nlev, a{array_dims}, b{array_dims})

  END SUBROUTINE driver_routine
    """
    fcode_kernel = """
  module kernel_mod
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE kernel_routine(nlon, nlev, a, b)
    INTEGER, INTENT(IN)    :: nlon, nlev
    INTEGER, INTENT(INOUT) :: a(nlon,nlev)
    INTEGER, INTENT(INOUT) :: b(nlon,nlev)
    INTEGER :: i, j

    do j=1, nlon
      do i=1, nlev
        a(i,j) = i*10 + j
        b(i,j) = i*10 + j + 1
      end do
    end do
  END SUBROUTINE kernel_routine
  end module kernel_mod
    """
    def init_arguments(nlon, nlev, flattened=False):
        a = np.zeros(shape=(nlon*nlev) if flattened else (nlon,nlev,), order='F', dtype=np.int32)
        b = np.zeros(shape=(nlon*nlev) if flattened else (nlon,nlev,), order='F', dtype=np.int32)
        return a, b

    def validate_routine(routine):
        arrays = [var for var in FindVariables().visit(routine.body) if isinstance(var, sym.Array)]
        assert all(len(arr.dimensions) == 1 or not arr.dimensions for arr in arrays)
        assert all(len(arr.shape) == 1 for arr in arrays)

    kernel_module = Module.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = kernel_module.subroutines[0]

    # check for a(:,:) and b(:,:) if "explicit_dimensions"
    call = FindNodes(CallStatement).visit(driver.body)[0]
    if explicit_dimensions:
        assert call.arguments[-2].dimensions == (sym.RangeIndex((None, None)), sym.RangeIndex((None, None)))
        assert call.arguments[-1].dimensions == (sym.RangeIndex((None, None)), sym.RangeIndex((None, None)))
    else:
        assert call.arguments[-2].dimensions == ()
        assert call.arguments[-1].dimensions == ()

    # compile and test reference
    refname = f'ref_{driver.name}_{frontend}'
    reference = jit_compile_lib([kernel_module, driver], path=tempdir, name=refname, builder=builder)
    ref_function = reference.driver_routine

    nlon = 10
    nlev = 12
    a_ref, b_ref = init_arguments(nlon, nlev)
    ref_function(nlon, nlev, a_ref, b_ref)
    builder.clean()

    normalize_range_indexing(driver)
    normalize_range_indexing(kernel)

    # flatten all the arrays in the kernel and driver
    flatten_arrays(routine=kernel, order='F', start_index=1)
    flatten_arrays(routine=driver, order='F', start_index=1)

    # check whether all the arrays are 1-dimensional
    validate_routine(kernel)
    validate_routine(driver)

    # compile and test the flattened variant
    flattenedname = f'flattened_{driver.name}_{frontend}'
    flattened = jit_compile_lib([kernel_module, driver], path=tempdir, name=flattenedname, builder=builder)
    flattened_function = flattened.driver_routine

    a_flattened, b_flattened = init_arguments(nlon, nlev, flattened=True)
    flattened_function(nlon, nlev, a_flattened, b_flattened)

    # check whether reference and flattened variant(s) produce same result
    assert (a_flattened == a_ref.flatten(order='F')).all()
    assert (b_flattened == b_ref.flatten(order='F')).all()

    builder.clean()
    (tempdir/f'{driver.name}.f90').unlink()
    (tempdir/f'{kernel_module.name}.f90').unlink()
