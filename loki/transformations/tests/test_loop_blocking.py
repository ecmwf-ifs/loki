# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import numpy as np

from loki import available_frontends, Subroutine, pragmas_attached, find_driver_loops, Loop, fgen, \
   ir, FindNodes, jit_compile, clean_test, FindVariables, Array
from loki.transformations.loop_blocking import split_loop, block_loop_arrays


"""
Splitting tests. These tests check that loop
"""
LOKI_LOOP_SLIT_VAR_ADDITION = 7

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [10, 117])
@pytest.mark.parametrize('n', [50, 1200])
def test_1d_splitting(tmp_path, frontend, block_size, n):
    """
    Apply loop blocking of simple loops into two loops
    """
    fcode = """
subroutine test_1d_splitting(a, b, n)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a(n)
  real(kind=8), intent(inout) :: b(n)
  integer :: i
  !$loki driver-loop
  do i=1,n
    a(i) = real(i, kind=8)
  end do
end subroutine test_1d_splitting
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    num_loops = len(loops)
    num_vars = len(routine.variable_map)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)
    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)

    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    a = np.zeros(n, order='F')
    b = np.zeros(n, order='F')
    function(a, b, n)
    assert np.all(b == 0.), "b array should not be modified."
    a_ref = np.linspace(1, n, n)
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_1d_splitting_multi_var(tmp_path, frontend, block_size, n):
    """
    Apply loop blocking of simple loops into two loops
    """
    fcode = """
subroutine test_1d_splitting_multi_var(a, b, n)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a(n)
  real(kind=8), intent(inout) :: b(n)
  real(kind=8) :: c(n)
  integer :: i
  !$loki driver-loop
  do i=1,n
    c(1) = c(1) + i
    a(i) = real(i)
  end do
end subroutine test_1d_splitting_multi_var
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    num_loops = len(loops)
    num_vars = len(routine.variable_map)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)
    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)

    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    a = np.zeros(n, order='F')
    b = np.zeros(n, order='F')
    function(a, b, n)
    assert np.all(b == 0.), "b array should not be modified."
    a_ref = np.linspace(1, n, n)
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_2d_splitting(tmp_path, frontend, block_size, n):
    fcode = """
    subroutine test_2d_splitting(a, b, n)
      implicit none
      integer, intent(in) :: n
      real(kind=8), intent(inout) :: a(n)
      real(kind=8), intent(inout) :: b(n,n)
      real(kind=8) :: c(n)
      integer :: i
      !$loki driver-loop
      do i=1,n
        a(i) = i
        c(1) = a(i)
        b(:,i) = a(i)
      end do
    end subroutine test_2d_splitting
        """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    num_loops = len(loops)
    num_vars = len(routine.variable_map)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)
    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)

    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    a = np.zeros(n, order='F')
    b = np.zeros((n,n), order='F')
    function(a, b, n)
    a_ref = np.linspace(1, n, n)
    b_ref = np.tile(a_ref, (n,1))
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"
    assert np.array_equal(b, b_ref), "b should equal b_ref"

    clean_test(filepath)



@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_3d_splitting(tmp_path, frontend, block_size, n):
    fcode = """
    subroutine test_3d_splitting(a, b, c, n)
      implicit none
      integer, intent(in) :: n
      real(kind=8), intent(inout) :: a(n)
      real(kind=8), intent(inout) :: b(2,n)
      real(kind=8), intent(inout) :: c(2,2,n)
      integer :: i
      !$loki driver-loop
      do i=1,n
        a(i) = i
        b(:,i) = a(i)
        c(:,:,i) = a(i)
      end do
    end subroutine test_3d_splitting
        """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    num_loops = len(loops)
    num_vars = len(routine.variable_map)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)
    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)

    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    a = np.zeros(n, order='F')
    b = np.zeros((2,n), order='F')
    c = np.zeros((2,2,n), order='F')
    function(a, b, c, n)
    a_ref = np.linspace(1, n, n)
    b_ref = np.tile(a_ref, (2, 1))
    c_ref = np.tile(a_ref, (2,2,1))
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"
    assert np.array_equal(b, b_ref), "b should equal b_ref"
    assert np.array_equal(c, c_ref), "c should equal c_ref"

    clean_test(filepath)

"""
--------------------------------------------------------------------------------
Blocking tests

Tests that variables are correctly blocked, and that blocked loops produce
the correct output.
--------------------------------------------------------------------------------
"""

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_1d_blocking(tmp_path, frontend, block_size, n):
    """
    Apply loop blocking of simple loops into two loops
    """
    fcode = """
subroutine test_1d_blocking(a, b, n)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: a(n)
  real(kind=8), intent(inout) :: b(n)
  integer :: i
  !$loki driver-loop
  do i=1,n
    a(i) = real(i)
  end do
end subroutine test_1d_blocking
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)

    num_loops = len(loops)
    num_vars = len(routine.variable_map)

    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"
    num_vars = len(routine.variable_map)

    blocking_indices = ['i']
    block_loop_arrays(routine, splitting_vars, inner_loop, outer_loop, blocking_indices)
    for var in FindVariables().visit(inner_loop.body):
        if isinstance(var, Array):
            for idx in blocking_indices:
                assert idx not in var.dimensions, "The variable should be blocked and the local variable used"

    assert len(routine.variable_map) == num_vars+1, "Expected 1 loop blocking to be added"

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    a = np.zeros(n, order='F')
    b = np.zeros(n, order='F')
    a_ref = np.linspace(1, n, n)
    function(a, b, n)
    assert np.all(b == 0.), "b array should not be modified."
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_1d_blocking_multi_intent(tmp_path, frontend, block_size, n):
    """
    Apply loop blocking of simple loops into two loops
    """
    fcode = """
subroutine test_1d_blocking_multi_intent(a, b, n)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n)
  real(kind=8), intent(inout) :: b(n)
  integer :: i
  !$loki driver-loop
  do i=1,n
    b(i) = b(i) + a(i)*a(i)
  end do
end subroutine test_1d_blocking_multi_intent
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)

    num_loops = len(loops)
    num_vars = len(routine.variable_map)
    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"


    num_vars = len(routine.variable_map)
    blocking_indices = ['i']
    block_loop_arrays(routine, splitting_vars, inner_loop, outer_loop, blocking_indices)

    assert len(routine.variable_map) == num_vars+2, "Expected 2 loop blocking to be added"
    for var in FindVariables().visit(inner_loop.body):
        if isinstance(var, Array):
            for idx in blocking_indices:
                assert idx not in var.dimensions, "The variable should be blocked and the local variable used"
    block_loop_arrays(routine, splitting_vars, inner_loop, outer_loop, blocking_indices=['i'])


    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    a = np.linspace(1, n, n)
    b = np.ones(n, order='F')
    a_ref = np.linspace(1, n, n)
    b_ref = b + a*a
    function(a, b, n)
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"
    assert np.array_equal(b, b_ref), "b should equal to (2, 5, ..., 1 + n^2)"
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_2d_blocking(tmp_path, frontend, block_size, n):
    fcode = """
    subroutine test_2d_blocking(a, b, n)
      implicit none
      integer, intent(in) :: n
      real(kind=8), intent(inout) :: a(n)
      real(kind=8), intent(inout) :: b(n,n)
      real(kind=8) :: c(n)
      integer :: i
      !$loki driver-loop
      do i=1,n
        a(i) = i
        c(1) = a(i)
        b(:,i) = a(i)
      end do
    end subroutine test_2d_blocking
        """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    num_loops = len(loops)
    num_vars = len(routine.variable_map)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)
    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)

    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"

    num_vars = len(routine.variable_map)
    blocking_indices = ['i']
    block_loop_arrays(routine, splitting_vars, inner_loop, outer_loop, blocking_indices)

    assert len(routine.variable_map) == num_vars + 2, "Expected 2 loop blocking to be added"
    for var in FindVariables().visit(inner_loop.body):
        if isinstance(var, Array):
            for idx in blocking_indices:
                assert idx not in var.dimensions, "The variable should be blocked and the local variable used"

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    a = np.zeros(n, order='F')
    b = np.zeros((n,n), order='F')
    function(a, b, n)
    a_ref = np.linspace(1, n, n)
    b_ref = np.tile(a_ref, (n,1))
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"
    assert np.array_equal(b, b_ref), "b[:,1] should equal a and a_ref"

    clean_test(filepath)



@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_3d_blocking(tmp_path, frontend, block_size, n):
    fcode = """
    subroutine test_3d_blocking(a, b, c, n)
      implicit none
      integer, intent(in) :: n
      real(kind=8), intent(inout) :: a(n)
      real(kind=8), intent(inout) :: b(2,n)
      real(kind=8), intent(inout) :: c(2,2,n)
      integer :: i
      !$loki driver-loop
      do i=1,n
        a(i) = i
        b(:,i) = a(i)
        c(:,:,i) = a(i)
      end do
    end subroutine test_3d_blocking
        """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    num_loops = len(loops)
    num_vars = len(routine.variable_map)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)
    splitting_vars, inner_loop, outer_loop = split_loop(routine, loops[0], block_size)
    loops = FindNodes(ir.Loop).visit(routine.ir)

    assert len(
        loops) == num_loops + 1, f"Total number of loops transformation is: {len(loops)} but expected {num_loops + 1}"
    assert len(
        routine.variable_map) == num_vars + LOKI_LOOP_SLIT_VAR_ADDITION, f"Total number of variables after loop splitting is: {len(routine.variable_map)} but expected {num_vars + LOKI_LOOP_SLIT_VAR_ADDITION}"

    num_vars = len(routine.variable_map)
    blocking_indices = ['i']
    block_loop_arrays(routine, splitting_vars, inner_loop, outer_loop, blocking_indices)

    assert len(routine.variable_map) == num_vars + 3, "Expected 3 loop blocking to be added"
    for var in FindVariables().visit(inner_loop.body):
        if isinstance(var, Array):
            for idx in blocking_indices:
                assert idx not in var.dimensions, "The variable should be blocked and the local variable used"

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)


    a = np.zeros(n, order='F')
    b = np.zeros((2,n), order='F')
    c = np.zeros((2,2,n), order='F')
    function(a, b, c, n)
    a_ref = np.linspace(1, n, n)
    b_ref = np.tile(a_ref, (2, 1))
    c_ref = np.tile(a_ref, (2,2,1))
    assert np.array_equal(a, a_ref), "a should be equal to a_ref=(1, 2, ..., n)"
    assert np.array_equal(b, b_ref), "b should equal b_ref"
    assert np.array_equal(c, c_ref), "c should equal c_ref"

    clean_test(filepath)

