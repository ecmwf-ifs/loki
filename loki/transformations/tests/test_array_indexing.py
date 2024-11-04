# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import platform
import pytest
import numpy as np

from loki import Module, Subroutine, fgen
from loki.build import jit_compile, jit_compile_lib, clean_test, Builder, Obj
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, CallStatement, Loop, FindVariables, Assignment

from loki.transformations.array_indexing import (
    promote_variables, demote_variables, invert_array_indices,
    flatten_arrays, normalize_array_shape_and_access,
    shift_to_zero_indexing, resolve_vector_notation,
    LowerConstantArrayIndices, remove_explicit_array_dimensions,
    add_explicit_array_dimensions
)
from loki.transformations.transpile import FortranCTransformation


@pytest.fixture(scope='function', name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variable_scalar(tmp_path, frontend):
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
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    ret = function()
    assert ret == 55

    # Apply and test the transformation
    assert isinstance(routine.variable_map['tmp'], sym.Scalar)
    promote_variables(routine, ['TMP'], pos=0, index=routine.variable_map['JK'], size=sym.Literal(10))
    assert isinstance(routine.variable_map['tmp'], sym.Array)
    assert routine.variable_map['tmp'].shape == (sym.Literal(10),)

    promoted_filepath = tmp_path/(f'{routine.name}_promoted_{frontend}.f90')
    promoted_function = jit_compile(routine, filepath=promoted_filepath, objname=routine.name)
    ret = promoted_function()
    assert ret == 55


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_variables(tmp_path, frontend):
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

    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
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
    promoted_filepath = tmp_path/(f'{routine.name}_promoted_{frontend}.f90')
    promoted_function = jit_compile(routine, filepath=promoted_filepath, objname=routine.name)

    scalar = np.zeros(shape=(1,), order='F', dtype=np.int32)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    promoted_function(scalar, vector, n)
    assert scalar == n*(n+1)//2
    assert np.all(vector[:-1] == np.array(list(range(n + 1, 2*n)), order='F', dtype=np.int32))
    assert vector[-1] == 3*n


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_demote_variables(tmp_path, frontend):
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

    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
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
    demoted_filepath = tmp_path/(f'{routine.name}_demoted_{frontend}.f90')
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
def test_transform_demote_dimension_arguments(tmp_path, frontend):
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

    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
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
    demoted_filepath = tmp_path/(f'{routine.name}_demoted_{frontend}.f90')
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


@pytest.mark.skipif(platform.system() == 'Darwin', reason='Unclear issue causing problems on MacOS (#352)')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('start_index', (0, 1, 5))
def test_transform_normalize_array_shape_and_access(tmp_path, frontend, start_index):
    """
    Test normalization of array shape and access, thus changing arrays with start
    index different than "1" to have start index "1".

    E.g., ``x1(5:len)`` -> ```x1(1:len-4)``
    """
    fcode = f"""
    module norm_arr_shape_access_mod
    implicit none

    contains

    subroutine norm_arr_shape_access(x1, x2, x3, x4, assumed_x1, l1, l2, l3, l4)
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
    end subroutine norm_arr_shape_access

    subroutine nested_routine(nested_x1, l1, c1)
        implicit none
        integer, intent(in) :: l1, c1
        integer, intent(inout) :: nested_x1(:)
        integer :: i1
        do i1=1,l1
            nested_x1(i1) = c1
        end do
    end subroutine nested_routine

    end module norm_arr_shape_access_mod
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    filepath = tmp_path/(f'norm_arr_shape_access_{frontend}.f90')
    # compile and test "original" module/function
    mod = jit_compile(module, filepath=filepath, objname='norm_arr_shape_access_mod')
    function = getattr(mod, 'norm_arr_shape_access')
    orig_x1, orig_x2, orig_x3, orig_x4, orig_assumed_x1 = init_arguments(l1, l2, l3, l4)
    function(orig_x1, orig_x2, orig_x3, orig_x4, orig_assumed_x1, l1, l2, l3, l4)
    clean_test(filepath)

    # apply `normalize_array_shape_and_access`
    for routine in module.routines:
        normalize_array_shape_and_access(routine)

    filepath = tmp_path/(f'norm_arr_shape_access_normalized_{frontend}.f90')
    # compile and test "normalized" module/function
    mod = jit_compile(module, filepath=filepath, objname='norm_arr_shape_access_mod')
    function = getattr(mod, 'norm_arr_shape_access')
    x1, x2, x3, x4, assumed_x1 = init_arguments(l1, l2, l3, l4)
    function(x1, x2, x3, x4, assumed_x1, l1, l2, l3, l4)
    clean_test(filepath)
    # validate the routine "norm_arr_shape_access"
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
def test_transform_flatten_arrays(tmp_path, frontend, builder, start_index):
    """
    Test flattening or arrays, meaning converting multi-dimensional
    arrays to one-dimensional arrays including corresponding
    index arithmetic.
    """
    fcode = f"""
    subroutine transf_flatten_arr(x1, x2, x3, x4, l1, l2, l3, l4)
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

    end subroutine transf_flatten_arr
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
    filepath = tmp_path/(f'{routine.name}_{start_index}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    orig_x1, orig_x2, orig_x3, orig_x4 = init_arguments(l1, l2, l3, l4)
    function(orig_x1, orig_x2, orig_x3, orig_x4, l1, l2, l3, l4)
    clean_test(filepath)

    # Test flattening order='F'
    f_routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_array_shape_and_access(f_routine)
    flatten_arrays(routine=f_routine, order='F', start_index=1)
    filepath = tmp_path/(f'{f_routine.name}_{start_index}_flattened_F_{frontend}.f90')
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
    invert_array_indices(c_routine)
    flatten_arrays(routine=c_routine, order='C', start_index=1)
    filepath = tmp_path/(f'{c_routine.name}_{start_index}_flattened_C_{frontend}.f90')
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
    f2c.apply(source=f2c_routine, path=tmp_path)
    libname = f'fc_{f2c_routine.name}_{start_index}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.transf_flatten_arr_fc_mod.transf_flatten_arr_fc
    f2c_x1, f2c_x2, f2c_x3, f2c_x4 = init_arguments(l1, l2, l3, l4, flattened=True)
    fc_function(f2c_x1, f2c_x2, f2c_x3, f2c_x4, l1, l2, l3, l4)
    validate_routine(c_routine)

    assert (f2c_x1 == orig_x1.flatten(order='F')).all()
    assert (f2c_x2 == orig_x2.flatten(order='F')).all()
    assert (f2c_x3 == orig_x3.flatten(order='F')).all()
    assert (f2c_x4 == orig_x4.flatten(order='F')).all()


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
def test_transform_flatten_arrays_call(tmp_path, frontend, builder, explicit_dimensions):
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

    kernel_module = Module.from_source(fcode_kernel, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path],
            definitions=kernel_module)
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
    reference = jit_compile_lib([kernel_module, driver], path=tmp_path, name=refname, builder=builder)
    ref_function = reference.driver_routine

    nlon = 10
    nlev = 12
    a_ref, b_ref = init_arguments(nlon, nlev)
    ref_function(nlon, nlev, a_ref, b_ref)
    builder.clean()

    # flatten all the arrays in the kernel and driver
    flatten_arrays(routine=kernel, order='F', start_index=1)
    flatten_arrays(routine=driver, order='F', start_index=1)

    # check whether all the arrays are 1-dimensional
    validate_routine(kernel)
    validate_routine(driver)

    # compile and test the flattened variant
    flattenedname = f'flattened_{driver.name}_{frontend}'
    flattened = jit_compile_lib([kernel_module, driver], path=tmp_path, name=flattenedname, builder=builder)
    flattened_function = flattened.driver_routine

    a_flattened, b_flattened = init_arguments(nlon, nlev, flattened=True)
    flattened_function(nlon, nlev, a_flattened, b_flattened)

    # check whether reference and flattened variant(s) produce same result
    assert (a_flattened == a_ref.flatten(order='F')).all()
    assert (b_flattened == b_ref.flatten(order='F')).all()

    builder.clean()

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('recurse_to_kernels', (False, True))
@pytest.mark.parametrize('inline_external_only', (False, True))
@pytest.mark.parametrize('pass_as_kwarg', (False, True,))
def test_lower_constant_array_indices(tmp_path, frontend, recurse_to_kernels, inline_external_only, pass_as_kwarg):
    """
    Test lowering constant array indices
    """
    fcode_driver = f"""
subroutine driver(nlon,nlev,nb,var)
  use kernel_mod, only: kernel
  implicit none
  integer, parameter :: param_1 = 1
  integer, parameter :: param_2 = 2
  integer, parameter :: param_3 = 5
  integer, intent(in) :: nlon,nlev,nb
  real, intent(inout) :: var(nlon,nlev,param_3,nb)
  integer :: ibl
  integer :: offset
  integer :: some_val
  integer :: loop_start, loop_end
  loop_start = 2
  loop_end = nb
  some_val = 0
  offset = 1
  !$omp test
  do ibl=loop_start, loop_end
    call kernel(nlon,nlev,{'var=' if pass_as_kwarg else ''}var(:,:,param_1,ibl), {'another_var=' if pass_as_kwarg else ''}var(:,:,param_2:param_3,ibl), {'icend=' if pass_as_kwarg else ''}offset, {'lstart=' if pass_as_kwarg else ''}loop_start, {'lend=' if pass_as_kwarg else ''}loop_end)
    call kernel(nlon,nlev,{'var=' if pass_as_kwarg else ''}var(:,:,param_1,ibl), {'another_var=' if pass_as_kwarg else ''}var(:,:,param_2:param_3,ibl), {'icend=' if pass_as_kwarg else ''}offset, {'lstart=' if pass_as_kwarg else ''}loop_start, {'lend=' if pass_as_kwarg else ''}loop_end)
    ! call kernel(nlon,nlev,var(:,:,param_1,ibl), var(:,:,param_2:param_3,ibl), offset, loop_start, loop_end)
  enddo
end subroutine driver
"""

    fcode_kernel = """
module kernel_mod
implicit none
contains
subroutine kernel(nlon,nlev,var,another_var,icend,lstart,lend)
  use compute_mod, only: compute
  implicit none
  integer, intent(in) :: nlon,nlev,icend,lstart,lend
  real, intent(inout) :: var(nlon,nlev)
  real, intent(inout) :: another_var(nlon,nlev,4)
  integer :: jk, jl, jt
  var(:,:) = 0.
  do jk = 1,nlev
    do jl = 1, nlon
      var(jl, jk) = 0.
      do jt= 1,4
        another_var(jl, jk, jt) = 0.0
      end do
    end do
  end do
  call compute(nlon,nlev,var)
  call compute(nlon,nlev,var)
end subroutine kernel
end module kernel_mod
"""

    fcode_nested_kernel = """
module compute_mod
implicit none
contains
subroutine compute(nlon,nlev,var)
  implicit none
  integer, intent(in) :: nlon,nlev
  real, intent(inout) :: var(nlon,nlev)
  var(:,:) = 0.
end subroutine compute
end module compute_mod
"""

    nested_kernel_mod = Module.from_source(fcode_nested_kernel, frontend=frontend, xmods=[tmp_path])
    kernel_mod = Module.from_source(fcode_kernel, frontend=frontend, definitions=nested_kernel_mod, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=kernel_mod, xmods=[tmp_path])

    kwargs = {'recurse_to_kernels': recurse_to_kernels, 'inline_external_only': inline_external_only}
    LowerConstantArrayIndices(**kwargs).apply(driver, role='driver', targets=('kernel',))
    LowerConstantArrayIndices(**kwargs).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    LowerConstantArrayIndices(**kwargs).apply(nested_kernel_mod['compute'], role='kernel')

    # driver
    kernel_calls = FindNodes(CallStatement).visit(driver.body)
    for kernel_call in kernel_calls:
        if pass_as_kwarg:
            arg1 = kernel_call.kwarguments[0][1]
            arg2 = kernel_call.kwarguments[1][1]
        else:
            arg1 = kernel_call.arguments[2]
            arg2 = kernel_call.arguments[3]
        if inline_external_only and frontend != OMNI:
            assert arg1.dimensions == (':', ':', 'param_1', 'ibl')
            assert arg2.dimensions == (':', ':', 'param_2:param_3', 'ibl')
        else:
            assert arg1.dimensions == (':', ':', ':', 'ibl')
            assert arg2.dimensions == (':', ':', ':', 'ibl')
    # kernel
    kernel_vars = kernel_mod['kernel'].variable_map
    if inline_external_only and frontend != OMNI:
        assert kernel_vars['var'].shape == ('nlon', 'nlev')
        assert kernel_vars['var'].dimensions == ('nlon', 'nlev')
        assert kernel_vars['another_var'].shape == ('nlon', 'nlev', 4)
        assert kernel_vars['another_var'].dimensions == ('nlon', 'nlev', 4)
    else:
        assert kernel_vars['var'].shape == ('nlon', 'nlev', 5)
        assert kernel_vars['var'].dimensions == ('nlon', 'nlev', 5)
        assert kernel_vars['another_var'].shape == ('nlon', 'nlev', 5)
        assert kernel_vars['another_var'].dimensions == ('nlon', 'nlev', 5)
    if inline_external_only and frontend != OMNI:
        for var in FindVariables().visit(kernel_mod['kernel'].body):
            if var.name.lower() == 'var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert var.dimensions == ('jl', 'jk')
            if var.name.lower() == 'another_var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert tuple(str(dim) for dim in var.dimensions) == ('jl', 'jk', 'jt')
    else:
        for var in FindVariables().visit(kernel_mod['kernel'].body):
            if var.name.lower() == 'var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert var.dimensions == ('jl', 'jk', 1)
            if var.name.lower() == 'another_var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert tuple(str(dim) for dim in var.dimensions) == ('jl', 'jk', 'jt + 2 + -1')
    compute_calls = FindNodes(CallStatement).visit(kernel_mod['kernel'].body)
    for compute_call in compute_calls:
        for arg in compute_call.arguments:
            if arg.name.lower() == 'var':
                if inline_external_only and frontend != OMNI:
                    assert arg.dimensions == (':', ':')
                elif recurse_to_kernels:
                    assert arg.dimensions == (':', ':', ':')
                else:
                    assert arg.dimensions == (':', ':', '1')
    # nested kernel
    nested_kernel_var = nested_kernel_mod['compute'].variable_map['var']
    if recurse_to_kernels and (not inline_external_only or frontend == OMNI):
        assert nested_kernel_var.shape == ('nlon', 'nlev', 5)
        assert nested_kernel_var.dimensions == ('nlon', 'nlev', 5)
        for var in FindVariables().visit(nested_kernel_mod['compute'].body):
            if var.name.lower() == 'var':
                assert var.dimensions == (':', ':', 1)
    else:
        assert nested_kernel_var.shape == ('nlon', 'nlev')
        assert nested_kernel_var.dimensions == ('nlon', 'nlev')
        for var in FindVariables().visit(nested_kernel_mod['compute'].body):
            if var.name.lower() == 'var':
                assert var.dimensions == (':', ':')


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('recurse_to_kernels', (False, True,))
@pytest.mark.parametrize('inline_external_only', (False, True,))
def test_lower_constant_array_indices_academic(tmp_path, frontend, recurse_to_kernels, inline_external_only):
    """
    Test lowering constant array indices for a valid but somewhat academic example ...

    The transformation is capable to handle that, but let's just hope we'll never see
    something like that out there in the wild ...
    """
    fcode_driver = """
subroutine driver(nlon,nlev,nb,var)
  use kernel_mod, only: kernel
  implicit none
  integer, parameter :: param_1 = 1
  integer, parameter :: param_2 = 2
  integer, parameter :: param_3 = 5
  integer, intent(in) :: nlon,nlev,nb
  real, intent(inout) :: var(nlon,4,3,nlev,param_3,nb)
  ! real, intent(inout) :: var(nlon,3,nlev,param_3,nb)
  integer :: ibl, j
  integer :: offset
  integer :: some_val
  integer :: loop_start, loop_end
  loop_start = 2
  loop_end = nb
  some_val = 0
  offset = 1
  !$omp test
  do ibl=loop_start, loop_end
    do j=1,4
      call kernel(nlon,nlev,var(:,j,1,:,param_1,ibl), var(:,j,2:3,:,param_2:param_3,ibl), offset, loop_start, loop_end)
      call kernel(nlon,nlev,var(:,j,1,:,param_1,ibl), var(:,j,2:3,:,param_2:param_3,ibl), offset, loop_start, loop_end)
    end do
  enddo
end subroutine driver
"""

    fcode_kernel = """
module kernel_mod
implicit none
contains
subroutine kernel(nlon,nlev,var,another_var,icend,lstart,lend)
  use compute_mod, only: compute
  implicit none
  integer, intent(in) :: nlon,nlev,icend,lstart,lend
  real, intent(inout) :: var(nlon,nlev)
  real, intent(inout) :: another_var(nlon,2,nlev,4)
  integer :: jk, jl, jt
  var(:,:) = 0.
  do jk = 1,nlev
    do jl = 1, nlon
      var(jl, jk) = 0.
      do jt= 1,4
        another_var(jl, 1, jk, jt) = 0.0
      end do
    end do
  end do
  call compute(nlon,nlev,var)
  call compute(nlon,nlev,var)
end subroutine kernel
end module kernel_mod
"""

    fcode_nested_kernel = """
module compute_mod
implicit none
contains
subroutine compute(nlon,nlev,var)
  implicit none
  integer, intent(in) :: nlon,nlev
  real, intent(inout) :: var(nlon,nlev)
  var(:,:) = 0.
end subroutine compute
end module compute_mod
"""

    nested_kernel_mod = Module.from_source(fcode_nested_kernel, frontend=frontend, xmods=[tmp_path])
    kernel_mod = Module.from_source(fcode_kernel, frontend=frontend, definitions=nested_kernel_mod, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=kernel_mod, xmods=[tmp_path])

    kwargs = {'recurse_to_kernels': recurse_to_kernels, 'inline_external_only': inline_external_only}
    LowerConstantArrayIndices(**kwargs).apply(driver, role='driver', targets=('kernel',))
    LowerConstantArrayIndices(**kwargs).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    LowerConstantArrayIndices(**kwargs).apply(nested_kernel_mod['compute'], role='kernel')

    # driver
    kernel_calls = FindNodes(CallStatement).visit(driver.body)
    for kernel_call in kernel_calls:
        if inline_external_only and frontend != OMNI:
            assert kernel_call.arguments[2].dimensions == (':', 'j', ':', ':', 'param_1', 'ibl')
            assert kernel_call.arguments[3].dimensions == (':', 'j', ':', ':', 'param_2:param_3', 'ibl')
        else:
            assert kernel_call.arguments[2].dimensions == (':', 'j', ':', ':', ':', 'ibl')
            assert kernel_call.arguments[3].dimensions == (':', 'j', ':', ':', ':', 'ibl')
    # kernel
    kernel_vars = kernel_mod['kernel'].variable_map
    if inline_external_only and frontend != OMNI:
        assert kernel_vars['var'].shape == ('nlon', 3, 'nlev')
        assert kernel_vars['var'].dimensions == ('nlon', 3, 'nlev')
        assert kernel_vars['another_var'].shape == ('nlon', 3, 'nlev', 4)
        assert kernel_vars['another_var'].dimensions == ('nlon', 3, 'nlev', 4)
    else:
        assert kernel_vars['var'].shape == ('nlon', '3', 'nlev', 5)
        assert kernel_vars['var'].dimensions == ('nlon', '3', 'nlev', 5)
        assert kernel_vars['another_var'].shape == ('nlon', '3', 'nlev', 5)
        assert kernel_vars['another_var'].dimensions == ('nlon', '3', 'nlev', 5)
    if inline_external_only and frontend != OMNI:
        for var in FindVariables().visit(kernel_mod['kernel'].body):
            if var.name.lower() == 'var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert var.dimensions == ('jl', 1, 'jk')
            if var.name.lower() == 'another_var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert tuple(str(dim) for dim in var.dimensions) == ('jl', '1 + 2 + -1', 'jk', 'jt')
    else:
        for var in FindVariables().visit(kernel_mod['kernel'].body):
            if var.name.lower() == 'var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert var.dimensions == ('jl', 1, 'jk', 1)
            if var.name.lower() == 'another_var' and not any(isinstance(dim, sym.RangeIndex) for dim in var.dimensions):
                assert tuple(str(dim) for dim in var.dimensions) == ('jl', '1 + 2 + -1', 'jk', 'jt + 2 + -1')
    compute_calls = FindNodes(CallStatement).visit(kernel_mod['kernel'].body)
    for compute_call in compute_calls:
        for arg in compute_call.arguments:
            if arg.name.lower() == 'var':
                if inline_external_only and frontend != OMNI:
                    if recurse_to_kernels:
                        assert arg.dimensions == (':', ':', ':')
                    else:
                        assert arg.dimensions == (':', 1, ':')
                elif recurse_to_kernels:
                    assert arg.dimensions == (':', ':', ':', ':')
                else:
                    assert arg.dimensions == (':', 1, ':', '1')
    # nested kernel
    nested_kernel_var = nested_kernel_mod['compute'].variable_map['var']
    if recurse_to_kernels and (not inline_external_only or frontend == OMNI):
        assert nested_kernel_var.shape == ('nlon', 3, 'nlev', 5)
        assert nested_kernel_var.dimensions == ('nlon', 3, 'nlev', 5)
        for var in FindVariables().visit(nested_kernel_mod['compute'].body):
            if var.name.lower() == 'var':
                assert var.dimensions == (':', 1, ':', 1)
    else:
        if recurse_to_kernels:
            assert nested_kernel_var.shape == ('nlon', 3, 'nlev')
            assert nested_kernel_var.dimensions == ('nlon', 3, 'nlev')
        else:
            assert nested_kernel_var.shape == ('nlon', 'nlev')
            assert nested_kernel_var.dimensions == ('nlon', 'nlev')
        for var in FindVariables().visit(nested_kernel_mod['compute'].body):
            if var.name.lower() == 'var':
                if recurse_to_kernels:
                    assert var.dimensions == (':', 1, ':')
                else:
                    assert var.dimensions == (':', ':')


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

    loops = FindNodes(Loop).visit(routine.body)
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]

    assert len(loops) == 4
    assert loops[0].variable == 'i_ret1_1'
    assert loops[0].bounds.children == (1, 'param1', 1) if frontend != OMNI else (1, 3, 1)
    assert loops[1].variable == 'i_ret1_0'
    assert loops[1].bounds.children == (1, 'param1', 1) if frontend != OMNI else (1, 3, 1)
    assert loops[2].variable == 'i_ret2_1'
    assert loops[2].bounds.children == (1, 'param2', 1) if frontend != OMNI else (1, 5, 1)
    assert loops[3].variable == 'i_ret2_0'
    assert loops[3].bounds.children == (1, 'param1', 1) if frontend != OMNI else (1, 3, 1)

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
def test_transform_resolve_vector_notation_common_loops(tmp_path, frontend):
    """
    Apply and test resolve vector notation utility with already
    available/appropriate loops.
    """
    fcode = """
subroutine transform_resolve_vector_notation_common_loops(scalar, vector, matrix, n, m, l)
  implicit none
  integer, intent(in) :: n, m, l
  integer, intent(inout) :: scalar, vector(n), matrix(l, n)
  integer :: tmp_scalar, tmp_vector(n, m), tmp_matrix(l, m, n), tmp_dummy(n, 0:4)
  integer :: jl, jk, jm

  tmp_dummy(:,:) = 0
  tmp_vector(:, 1) = tmp_dummy(:, 1)
  tmp_vector(:, :) = 0
  tmp_matrix(:, :, :) = 0
  matrix(:, :) = 0

  do jl=1,n
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

end subroutine transform_resolve_vector_notation_common_loops
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 3
    m = 2
    l = 3
    scalar = np.zeros(shape=(1,), order='F', dtype=np.int32)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    function(scalar, vector, matrix, n, m, l)

    assert all(scalar == 3)
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

    resolve_vector_notation(routine)

    loops = FindNodes(Loop).visit(routine.body)
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]

    assert len(loops) == 19
    assert loops[0].variable == 'i_tmp_dummy_1' and loops[0].bounds.children == (0, 4, None)
    assert loops[1].variable == 'jl' and loops[1].bounds.children == (1, 'n', 1)
    assert loops[2].variable == 'jl' and loops[2].bounds.children == (1, 'n', 1)
    assert loops[3].variable == 'jm' and loops[3].bounds.children == (1, 'm', 1)
    assert loops[4].variable == 'jl' and loops[4].bounds.children == (1, 'n', 1)
    assert loops[5].variable == 'jl' and loops[5].bounds.children == (1, 'n', 1)
    assert loops[6].variable == 'jm' and loops[6].bounds.children == (1, 'm', 1)
    assert loops[7].variable == 'jk' and loops[7].bounds.children == (1, 'l', 1)
    assert loops[8].variable == 'jl' and loops[8].bounds.children == (1, 'n', 1)
    assert loops[9].variable == 'jk' and loops[9].bounds.children == (1, 'l', 1)
    assert loops[10].variable == 'jl' and loops[10].bounds.children == (1, 'n', None)
    assert loops[11].variable == 'jm' and loops[11].bounds.children == (1, 'm', None)
    assert loops[12].variable == 'jm' and loops[12].bounds.children == (1, 'm', None)
    assert loops[13].variable == 'jl' and loops[13].bounds.children == (1, 'n', None)
    assert loops[14].variable == 'jk' and loops[14].bounds.children == (1, 'l', None)
    assert loops[15].variable == 'jk' and loops[15].bounds.children == (1, 'l', None)
    assert loops[16].variable == 'jl' and loops[16].bounds.children == (1, 'n', 1)
    assert loops[17].variable == 'jm' and loops[17].bounds.children == (1, 'm', None)
    assert loops[18].variable == 'jl' and loops[18].bounds.children == (1, 'n', None)

    assert len(arrays) == 15
    assert arrays[0].name.lower() == 'tmp_dummy' and arrays[0].dimensions == ('jl', 'i_tmp_dummy_1')
    assert arrays[1].name.lower() == 'tmp_vector' and arrays[1].dimensions == ('jl', 1)
    assert arrays[2].name.lower() == 'tmp_dummy' and arrays[2].dimensions == ('jl', 1)
    assert arrays[3].name.lower() == 'tmp_vector' and arrays[3].dimensions == ('jl', 'jm')
    assert arrays[4].name.lower() == 'tmp_matrix' and arrays[4].dimensions == ('jk', 'jm', 'jl')

    # Test promoted routine
    resolved_filepath = tmp_path/(f'{routine.name}_resolved_{frontend}.f90')
    resolved_function = jit_compile(routine, filepath=resolved_filepath, objname=routine.name)

    n = 3
    m = 2
    l = 3
    scalar = np.zeros(shape=(1,), order='F', dtype=np.int32)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    resolved_function(scalar, vector, matrix, n, m, l)

    assert all(scalar == 3)
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
    kernel_call = FindNodes(CallStatement).visit(driver.body)[0]
    kernel_call_array_args = [arg for arg in kernel_call.arguments if isinstance(arg, sym.Array)]
    assert all(len(arg.dimensions) == 2 for arg in kernel_call_array_args)

    # remove explicit array dimensions (possibly only for calls)
    remove_explicit_array_dimensions(driver, calls_only=calls_only)
    remove_explicit_array_dimensions(kernel, calls_only=calls_only)

    kernel_call = FindNodes(CallStatement).visit(driver.body)[0]
    kernel_call_array_args = [arg for arg in kernel_call.arguments if isinstance(arg, sym.Array)]
    assert all(not arg.dimensions for arg in kernel_call_array_args)
    if calls_only:
        assignments = FindNodes(Assignment).visit(kernel.body)
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
