# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from math import floor

import pytest
import numpy as np

from loki import available_frontends, Subroutine, pragmas_attached, find_driver_loops, Loop, fgen, \
   ir, FindNodes, LoopRange, IntLiteral, jit_compile, clean_test, FindVariables, Array
from loki.expression.parser import LokiEvaluationMapper
from loki.transformations.loop_blocking import split_loop, block_loop_arrays, \
    normalized_loop_range, iteration_number, iteration_index, LoopBockFieldAPITransformation


def get_pyrange(loop_range: LoopRange):
    """
    Converts a LoopRange of IntLiterals to a python range.
    """
    LEM = LokiEvaluationMapper()
    if loop_range.step is None:
        return range(LEM(loop_range.start), floor(LEM(loop_range.stop))+1)
    else:
        return range(LEM(loop_range.start), floor(LEM(loop_range.stop))+1, LEM(loop_range.step))


@pytest.mark.parametrize('frontend', available_frontends())
def test_normalized_loop_range(tmp_path, frontend):
    """
    Tests the num_iterations and normalized_loop_range functions.
    """
    for start in range(-10, 11):
        for stop in range(start + 1, 50 + 1, 4):
            for step in range(1, stop - start):
                loop_range = LoopRange((start, stop, step))
                pyrange = range(start, stop + 1, step)

                normalized_range = normalized_loop_range(loop_range)
                assert normalized_range.step is None, "LoopRange.step should be None in a normalized range"

                normalized_start = LokiEvaluationMapper()(normalized_range.start)
                assert normalized_start == 1, "LoopRange.start should be equal to 1 in a normalized range"

                normalized_stop = floor(LokiEvaluationMapper()(normalized_range.stop))
                assert normalized_stop == len(
                    pyrange), "LoopRange.stop should be equal to the total number of iterations of the original LoopRange"


@pytest.mark.parametrize('frontend', available_frontends())
def test_iteration_number(tmp_path, frontend):
    for start in range(-10, 11):
        for stop in range(start + 1, 50, 4):
            for step in range(1, stop - start):
                loop_range = LoopRange((start, stop, step))
                pyrange = range(start, stop + 1, step)
                normalized_range = get_pyrange(normalized_loop_range(loop_range))
                assert len(normalized_range) == len(
                    pyrange), "Length of normalized loop range should equal length of python loop range"

                LEM = LokiEvaluationMapper()
                assert all(n == LEM(iteration_number(IntLiteral(i), loop_range)) for i, n in
                           zip(pyrange, normalized_range))


@pytest.mark.parametrize('frontend', available_frontends())
def test_iteration_index(tmp_path, frontend):
    for start in range(-10, 11):
        for stop in range(start + 1, 50, 4):
            for step in range(1, stop - start):
                loop_range = LoopRange((start, stop, step))
                pyrange = range(start, stop + 1, step)
                normalized_range = get_pyrange(normalized_loop_range(loop_range))
                assert len(normalized_range) == len(
                    pyrange), "Length of normalized loop range should equal length of python loop range"

                LEM = LokiEvaluationMapper()
                assert all(i == LEM(iteration_index(IntLiteral(n), loop_range)) for i, n in
                           zip(pyrange, normalized_range))






"""
Splitting tests. These tests check that loop
"""
LOKI_LOOP_SLIT_VAR_ADDITION = 7

@pytest.mark.parametrize('frontend', available_frontends())
# @pytest.mark.parametrize('block_size', [10, 117, 250])
# @pytest.mark.parametrize('n', [50, 193, 500, 1200])
@pytest.mark.parametrize('block_size', [10])
@pytest.mark.parametrize('n', [50])
def test_1d_splitting(tmp_path, frontend, block_size, n):
    """
    Apply loop blocking of simple loops into two loops
    """
    fcode = """
subroutine test_1d_splitting(a, b, n)
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


"""
--------------------------------------------------------------------------------
Field API blocking tests

Tests that variables are correctly blocked, and that blocked loops produce
the correct output.
--------------------------------------------------------------------------------
"""


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('block_size', [117])
@pytest.mark.parametrize('n', [500])
def test_1d_field_blocking(tmp_path, frontend, block_size, n):
    """
    Apply loop blocking of simple loops into two loops
    """
    fcode = """
subroutine test_1d_field_blocking(a, b, n)
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(6,37)
  integer, intent(in) :: n
  real(kind=JPRB), intent(inout) :: a(n)
  real(kind=JPRB), intent(inout) :: b(n)
  integer :: i
  !$loki driver-loop
  do i=1,n
    a(i) = real(i)
  end do
end subroutine test_1d_field_blocking
    """
#     expected_code = """
# subroutine test_1d_field_blocking(a, b, n)
#   integer, intent(in) :: n
#   real(kind=jprb), intent(inout) :: a(n)
#   real(kind=jprb), intent(inout) :: b(n)
#   integer :: i
#
#   class(field_1rb), pointer :: a_field_block
#   real(kind=jprb), pointer, contiguous, dimension(:) :: a_block
#
#   !$loki driver-loop
#   do i=1,n
#
#     call field_new(a_field_block, data=a(block_start:block_end))
#     call pt_field_block%get_device_data_rdonly(pt_block)
#
#     !$acc data present(pt_block)
#     DO
#     a_block(i) = real(i)
#     end do
#     !$acc end data
#     call a_field_block%sync_host_rdwr()
#     call field_delete(a_field_block)
#
#   end do
# end subroutine test_1d_blocking
#     """

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.ir)
    with pragmas_attached(routine, Loop):
        loops = find_driver_loops(routine,
                                  targets=None)

    blocking_transformer = LoopBockFieldAPITransformation()
    blocking_transformer.apply(routine, loop=loops[0])
