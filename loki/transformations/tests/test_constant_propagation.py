# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    Subroutine, FindNodes, Loop, Assignment, Conditional
)
from loki.build import jit_compile
from loki.frontend import available_frontends

from loki.transformations.constant_propagation import ConstantPropagationTransformer


# Basic Types
@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_literals(tmp_path, frontend):
    fcode = """
subroutine const_prop_literals
  integer :: a, a1
  real :: b, b1
  character (len = 3) :: c, c1
  logical :: d, d1

  a1 = 1
  a = a1

  b1 = 1.5
  b = b1

  c1 = "foo"
  c = c1

  d1 = .true.
  d = d1

end subroutine const_prop_literals
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Proof that it compiles & runs, although no runtime testing here
    function()

    # Apply transformation
    routine.body = ConstantPropagationTransformer().visit(routine.body)

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert 'Assignment:: a = 1' in assignments
    assert 'Assignment:: b = 1.5' in assignments
    assert 'Assignment:: c = \'foo\'' in assignments
    assert 'Assignment:: d = True' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_ops_int(tmp_path, frontend):
    fcode = """
subroutine const_prop_ops_int(a_add, a_sub, a_mul, a_pow, a_div, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt)
  integer :: a = {a_val}
  integer :: b = {b_val}
  integer, intent(out) :: a_add, a_sub, a_mul, a_pow, a_div
  logical, intent(out) :: a_lt, a_leq, a_eq, a_neq, a_geq, a_gt

  a_add = a + b
  a_sub = a - b
  a_mul = a * b
  a_pow = a ** b
  a_div = a / b
  a_lt = a < b
  a_leq = a <= b
  a_eq = a == b
  a_neq = a /= b
  a_geq = a >= b
  a_gt = a > b

end subroutine const_prop_ops_int
"""

    a_val = 1
    b_val = 2
    routine = Subroutine.from_source(fcode.format(a_val=a_val, b_val=b_val), frontend=frontend)

    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a_add, a_sub, a_mul, a_pow, a_div, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt = function()

    assert a_add == a_val + b_val
    assert a_sub == a_val - b_val
    assert a_mul == a_val * b_val
    assert a_pow == a_val ** b_val
    # Fortran uses integer division by default
    assert a_div == a_val // b_val
    assert a_lt == a_val < b_val
    assert a_leq == a_val <= b_val
    assert a_eq == (a_val == b_val)
    assert a_neq == (a_val != b_val)
    assert a_geq == (a_val >= b_val)
    assert a_gt == (a_val > b_val)

    assert len(FindNodes(Assignment).visit(routine.body)) == 11

    # Apply transformation
    routine = ConstantPropagationTransformer().visit(routine)
    assert len(FindNodes(Assignment).visit(routine.body)) == 11

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: a_add = {a_val + b_val}' in assignments
    assert f'Assignment:: a_sub = {a_val - b_val}' in assignments
    assert f'Assignment:: a_mul = {a_val * b_val}' in assignments
    assert f'Assignment:: a_pow = {a_val ** b_val}' in assignments
    assert f'Assignment:: a_div = {a_val // b_val}' in assignments
    assert f'Assignment:: a_lt = {a_val < b_val}' in assignments
    assert f'Assignment:: a_leq = {a_val <= b_val}' in assignments
    assert f'Assignment:: a_eq = {a_val == b_val}' in assignments
    assert f'Assignment:: a_neq = {a_val != b_val}' in assignments
    assert f'Assignment:: a_geq = {a_val >= b_val}' in assignments
    assert f'Assignment:: a_gt = {a_val > b_val}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    a_add, a_sub, a_mul, a_pow, a_div, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt = new_function()

    assert a_add == a_val + b_val
    assert a_sub == a_val - b_val
    assert a_mul == a_val * b_val
    assert a_pow == a_val ** b_val
    # Fortran uses integer division by default
    assert a_div == a_val // b_val
    assert a_lt == a_val < b_val
    assert a_leq == a_val <= b_val
    assert a_eq == (a_val == b_val)
    assert a_neq == (a_val != b_val)
    assert a_geq == (a_val >= b_val)
    assert a_gt == (a_val > b_val)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_ops_float(tmp_path, frontend):
    fcode = """
subroutine const_prop_ops_float(a_add, a_sub, a_mul, a_pow, a_div, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt)
  real :: a = {a_val}
  real :: b = {b_val}
  real, intent(out) :: a_add, a_sub, a_mul, a_pow, a_div
  logical, intent(out) :: a_lt, a_leq, a_eq, a_neq, a_geq, a_gt

  a_add = a + b
  a_sub = a - b
  a_mul = a * b
  a_pow = a ** b
  a_div = a / b
  a_lt = a < b
  a_leq = a <= b
  a_eq = a == b
  a_neq = a /= b
  a_geq = a >= b
  a_gt = a > b

end subroutine const_prop_ops_float
"""

    a_val = 1.5
    b_val = 2.5
    routine = Subroutine.from_source(fcode.format(a_val=a_val, b_val=b_val), frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution

    a_add, a_sub, a_mul, a_pow, a_div, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt = function()

    assert a_add == a_val + b_val
    assert a_sub == a_val - b_val
    assert a_mul == a_val * b_val
    assert a_pow - a_val ** b_val < 1e-6
    assert a_div - a_val / b_val < 1e-6
    assert bool(a_lt) == (a_val < b_val)
    assert bool(a_leq) == (a_val <= b_val)
    assert bool(a_eq) == (a_val == b_val)
    assert bool(a_neq) == (a_val != b_val)
    assert bool(a_geq) == (a_val >= b_val)
    assert bool(a_gt) == (a_val > b_val)

    assert len(FindNodes(Assignment).visit(routine.body)) == 11

    # Apply transformation
    body = ConstantPropagationTransformer().visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 11

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: a_add = {a_val + b_val}' in assignments
    assert f'Assignment:: a_sub = {a_val - b_val}' in assignments
    assert f'Assignment:: a_mul = {a_val * b_val}' in assignments
    assert f'Assignment:: a_pow = {a_val ** b_val}' in assignments
    assert f'Assignment:: a_div = {a_val / b_val}' in assignments
    assert f'Assignment:: a_lt = {a_val < b_val}' in assignments
    assert f'Assignment:: a_leq = {a_val <= b_val}' in assignments
    assert f'Assignment:: a_eq = {a_val == b_val}' in assignments
    assert f'Assignment:: a_neq = {a_val != b_val}' in assignments
    assert f'Assignment:: a_geq = {a_val >= b_val}' in assignments
    assert f'Assignment:: a_gt = {a_val > b_val}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    a_add, a_sub, a_mul, a_pow, a_div, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt = new_function()

    assert a_add == a_val + b_val
    assert a_sub == a_val - b_val
    assert a_mul == a_val * b_val
    assert a_pow - a_val ** b_val < 1e-6
    assert a_div - a_val / b_val < 1e-6
    assert bool(a_lt) == (a_val < b_val)
    assert bool(a_leq) == (a_val <= b_val)
    assert bool(a_eq) == (a_val == b_val)
    assert bool(a_neq) == (a_val != b_val)
    assert bool(a_geq) == (a_val >= b_val)
    assert bool(a_gt) == (a_val > b_val)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_ops_string(tmp_path, frontend):
    fcode = """
subroutine const_prop_ops_string(a_concat, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt)
  character (len = {a_len}) :: a = '{a_val}'
  character (len = {b_len}) :: b = '{b_val}'
  character (len = {concat_len}), intent(out) :: a_concat
  logical, intent(out) :: a_lt, a_leq, a_eq, a_neq, a_geq, a_gt

  a_concat = a // b
  a_lt = a < b
  a_leq = a <= b
  a_eq = a == b
  a_neq = a /= b
  a_geq = a >= b
  a_gt = a > b

end subroutine const_prop_ops_string
"""

    a_val = 'foo'
    b_val = 'bar'
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, a_len=len(a_val), b_val=b_val, b_len=len(b_val), concat_len=len(a_val)+len(b_val)),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a_concat, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt = function()

    assert a_concat.decode('UTF-8') == a_val + b_val
    assert bool(a_lt) == (a_val < b_val)
    assert bool(a_leq) == (a_val <= b_val)
    assert bool(a_eq) == (a_val == b_val)
    assert bool(a_neq) == (a_val != b_val)
    assert bool(a_geq) == (a_val >= b_val)
    assert bool(a_gt) == (a_val > b_val)

    assert len(FindNodes(Assignment).visit(routine.body)) == 7

    # Apply transformation
    routine = ConstantPropagationTransformer().visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 7

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: a_concat = \'{a_val + b_val}\'' in assignments
    assert f'Assignment:: a_lt = {a_val < b_val}' in assignments
    assert f'Assignment:: a_leq = {a_val <= b_val}' in assignments
    assert f'Assignment:: a_eq = {a_val == b_val}' in assignments
    assert f'Assignment:: a_neq = {a_val != b_val}' in assignments
    assert f'Assignment:: a_geq = {a_val >= b_val}' in assignments
    assert f'Assignment:: a_gt = {a_val > b_val}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    a_concat, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt = new_function()

    assert a_concat.decode('UTF-8') == a_val + b_val
    assert bool(a_lt) == (a_val < b_val)
    assert bool(a_leq) == (a_val <= b_val)
    assert bool(a_eq) == (a_val == b_val)
    assert bool(a_neq) == (a_val != b_val)
    assert bool(a_geq) == (a_val >= b_val)
    assert bool(a_gt) == (a_val > b_val)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_ops_bool(tmp_path, frontend):
    fcode = """
subroutine const_prop_ops_bool(a_and, a_or, a_not, a_eqv, a_neqv)
  logical :: a = {a_val}
  logical :: b = {b_val}
  logical, intent(out) :: a_and, a_or, a_not, a_eqv, a_neqv

  a_and = a .and. b
  a_or = a .or. b
  a_not = .not. a
  a_eqv = a .eqv. b
  a_neqv = a .neqv. b

end subroutine const_prop_ops_bool
"""

    a = True
    b = False
    a_val = '.True.' if a else '.False.'
    b_val = '.True.' if b else '.False.'
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    print(routine.to_fortran())
    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a_and, a_or, a_not, a_eqv, a_neqv = function()

    assert bool(a_and) == (a and b)
    assert bool(a_or) == (a or b)
    assert bool(a_not) == (not a)
    assert bool(a_eqv) == (a == b)
    assert bool(a_neqv) == (a != b)

    assert len(FindNodes(Assignment).visit(routine.body)) == 5

    # Apply transformation
    routine = ConstantPropagationTransformer().visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 5

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: a_and = {a and b}' in assignments
    assert f'Assignment:: a_or = {a or b}' in assignments
    assert f'Assignment:: a_not = {not a}' in assignments
    assert f'Assignment:: a_eqv = {a == b}' in assignments
    assert f'Assignment:: a_neqv = {a != b}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    a_and, a_or, a_not, a_eqv, a_neqv = new_function()

    assert bool(a_and) == (a and b)
    assert bool(a_or) == (a or b)
    assert bool(a_not) == (not a)
    assert bool(a_eqv) == (a == b)
    assert bool(a_neqv) == (a != b)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_ops_bool_short_circuiting(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_ops_bool_short_circuiting(a_and, a_or)
  logical :: a = {a_val}
  logical :: b
  logical, intent(out) :: a_and, a_or

  integer :: n
  integer :: i
  real :: r
  integer, allocatable :: seed(:)

  call random_seed(size = n)
  allocate(seed(n))
  seed(:) = 1
  call random_seed(put=seed)
  call random_number(r)

  ! floor(r) will be 0, but this is only known at runtime. Statically, it is unknown
  b = floor(r) == 0

  a_and = .not. a .and. b
  a_or = a .or. b

end subroutine test_transform_region_const_prop_ops_bool_short_circuiting
"""

    a = True
    a_val = '.True.' if a else '.False.'
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a_and, a_or = function()

    assert (a_and == 1) == (a and False)
    assert (a_or  == 1) == (a or True)

    # Apply transformation
    routine = ConstantPropagationTransformer().visit(routine)

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert len(assignments) == 4
    assert f'Assignment:: a_and = False' in assignments
    assert f'Assignment:: a_or = True' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    a_and, a_or = new_function()

    assert (a_and == 1) == (a and False)
    assert (a_or  == 1) == (a or True)


# For loops
@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_basic(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_basic(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  integer :: i
  integer, intent(out) :: c

  c = 0
  do i = 1, a
    c = c + b
  end do

end subroutine test_transform_region_const_prop_for_loop_basic
"""

    a_val = 5
    b_val = 3
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == a_val * b_val

    # Apply transformation
    routine = ConstantPropagationTransformer().visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == a_val + 1
    assert len(FindNodes(Loop).visit(routine.body)) == 0

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    for i in range(1, a_val+1):
        assert f'Assignment:: c = {b_val*i}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == a_val * b_val


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_basic_no_unroll(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_basic_no_unroll(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  integer :: i, d
  integer, intent(out) :: c

  c = 0
  d = 0
  do i = 1, a
    c = a * b
    d = a * i
  end do
  c = c * 2
  d = d * 2

end subroutine test_transform_region_const_prop_for_loop_basic_no_unroll
"""

    a_val = 5
    b_val = 3
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == a_val * b_val * 2

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 6
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: c = {a_val * b_val}' in assignments
    assert f'Assignment:: d = {a_val}*i' in assignments

    assert f'Assignment:: c = {a_val * b_val * 2}' in assignments
    assert f'Assignment:: d = d*2' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == a_val * b_val * 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_neg_range_no_unroll(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_neg_range_no_unroll(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  integer :: i, d
  integer, intent(out) :: c

  c = 0
  do i = 1, a
    c = b
  end do
  c = c

end subroutine test_transform_region_const_prop_for_loop_neg_range_no_unroll
"""

    a_val = -1
    b_val = 3
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == 0

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: c = {b_val}' in assignments
    assert len([a for a in assignments if f'Assignment:: c = 0' == a]) == 2

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_never_taken_no_unroll(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_never_taken_no_unroll(c)
  integer :: a = {a_val}
  integer :: n, b
  integer :: i
  real :: r
  integer, allocatable :: seed(:)
  integer, intent(out) :: c

  call random_seed(size = n)
  allocate(seed(n))
  seed(:) = 1
  call random_seed(put=seed)
  call random_number(r)

  b = 0
  c = 0
  ! floor(r) will be 0, but this is only known at runtime. Statically, it is unknown
  do i = 1, floor(r)
    b = a
  end do

  c = b

end subroutine test_transform_region_const_prop_for_loop_never_taken_no_unroll
"""

    a_val = 5
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == 0

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert len(assignments) == 5
    assert f'Assignment:: b = {a_val}' in assignments

    assert len(FindNodes(Loop).visit(routine.body)) == 1

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_never_taken_nested_no_unroll(tmp_path, frontend):
    fcode = """
subroutine test_const_prop_for_loop_never_taken_nested_no_unroll(c)
  integer :: a = {a_val}
  integer :: n, b
  integer :: i, j
  real :: r
  integer, allocatable :: seed(:)
  integer, intent(out) :: c

  call random_seed(size = n)
  allocate(seed(n))
  seed(:) = 1
  call random_seed(put=seed)
  call random_number(r)

  b = 0
  c = 0
  ! floor(r) will be 0, but this is only known at runtime. Statically, it is unknown
  do i = 1, floor(r)
    b = a
    do j = 1,5
      b = 6
    end do
    b = b
  end do

  c = b

end subroutine test_const_prop_for_loop_never_taken_nested_no_unroll
"""

    a_val = 5
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == 0

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert len(assignments) == 7
    assert f'Assignment:: b = {a_val}' in assignments
    assert f'Assignment:: b = {6}' in assignments

    assert len(FindNodes(Loop).visit(routine.body)) == 2

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_double_never_taken_nested_no_unroll(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_never_taken_no_unroll(c)
  integer :: a = {a_val}
  integer :: n, b
  integer :: i
  real :: r
  integer, allocatable :: seed(:)
  integer, intent(out) :: c

  call random_seed(size = n)
  allocate(seed(n))
  seed(:) = 1
  call random_seed(put=seed)
  call random_number(r)

  b = 0
  c = 0
  ! floor(r) will be 0, but this is only known at runtime. Statically, it is unknown
  do i = 1, floor(r)
    b = a
    do j = 1, floor(r)
      b = 6
    end do
    b = b
  end do

  c = b

end subroutine test_transform_region_const_prop_for_loop_never_taken_no_unroll
"""

    a_val = 5
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == 0

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert len(assignments) == 7
    assert f'Assignment:: b = {a_val}' in assignments
    assert f'Assignment:: b = b' in assignments

    assert len(FindNodes(Loop).visit(routine.body)) == 2

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_nested(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_nested(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  integer :: i, j
  integer, intent(out) :: c

  c = 0
  do i = 1, a
      do j = 1, b
        c = c + i + j
      end do
  end do

end subroutine test_transform_region_const_prop_for_loop_nested
"""

    a_val = 5
    b_val = 3
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == b_val*(a_val*(a_val+1))/2 + a_val*(b_val*(b_val+1))/2

    # Apply transformation
    routine = ConstantPropagationTransformer().visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == a_val * b_val + 1
    assert len(FindNodes(Loop).visit(routine.body)) == 0

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    tmp = 0
    for i in range(1, a_val+1):
        for j in range(1, b_val+1):
            tmp = tmp + i + j
            assert f'Assignment:: c = {tmp}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == b_val*(a_val*(a_val+1))/2 + a_val*(b_val*(b_val+1))/2


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_nested_no_unroll(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_nested_no_unroll(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  integer :: i, j
  integer, intent(out) :: c

  c = 0
  do i = 1, a
      do j = 1, b
        c =  b
      end do
      c = c
  end do

  c = c

end subroutine test_transform_region_const_prop_for_loop_nested_no_unroll
"""

    a_val = 5
    b_val = 3
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == b_val

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert len(FindNodes(Loop).visit(routine.body)) == 2

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert len([a for a in assignments if f'Assignment:: c = {b_val}' == a]) == 3

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == b_val


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_nested_siblings(tmp_path, frontend):
    fcode = """
    subroutine test_transform_region_const_prop_loop_nested_siblings(c)
      integer :: a = {a_val}
      integer :: b = {b_val}
      integer :: i, j ,k
      integer, intent(out) :: c

      c = 0
      do i = 1, a
          do j = 1, b
            c = c + i + j
          end do
          do k = 1, b
            c = c + i + k
          end do
      end do

    end subroutine test_transform_region_const_prop_loop_nested_siblings
    """

    a_val = 5
    b_val = 3
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == b_val*(a_val*(a_val+1)) + a_val*(b_val*(b_val+1))

    # Apply transformation
    routine = ConstantPropagationTransformer().visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 2 * b_val * a_val + 1
    assert len(FindNodes(Loop).visit(routine.body)) == 0

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    tmp = 0
    for i in range(1, a_val+1):
        for j in range(1, b_val+1):
            tmp = tmp + i + j
            assert f'Assignment:: c = {tmp}' in assignments
        for j in range(1, b_val+1):
            tmp = tmp + i + j
            assert f'Assignment:: c = {tmp}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == b_val*(a_val*(a_val+1)) + a_val*(b_val*(b_val+1))


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_nested_siblings_no_unroll(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_loop_nested_siblings_no_unroll(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  integer :: i, j ,k
  integer, intent(out) :: c

  c = 0
  do i = 1, a
      do j = 1, b
        c = a
      end do
      c = c
      do k = 1, b
        c = b
      end do
      c = c
  end do
  c = c
end subroutine test_transform_region_const_prop_loop_nested_siblings_no_unroll
"""

    a_val = 5
    b_val = 3
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == b_val

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 6
    assert len(FindNodes(Loop).visit(routine.body)) == 3

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: c = {a_val}' in assignments
    assert f'Assignment:: c = {b_val}' in assignments

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == b_val


# Conditionals
@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_conditional_basic(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_conditional_basic(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  logical :: cond = {cond_val}
  integer, intent(out) :: c

  if (cond) then
    c = a
  else
    c = b
  endif

  c = c

end subroutine test_transform_region_const_prop_conditional_basic
"""

    a_val = 5
    b_val = 3
    cond = True

    cond_val = '.True.' if cond else '.False.'
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val, cond_val=cond_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == a_val if cond else b_val

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assert len(FindNodes(Conditional).visit(routine.body)) == 1

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: c = {a_val}' in assignments
    assert f'Assignment:: c = {b_val}' in assignments
    assert len([a for a in assignments if a == f'Assignment:: c = {a_val if cond else b_val}']) == 2

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == a_val if cond else b_val


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_conditional_dynamic_condition(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_conditional_dynamic_condition(c)
  integer :: a = {a_val}
  integer :: b = {b_val}
  logical :: cond
  integer, intent(out) :: c

  integer :: n
  integer :: i
  real :: r
  integer, allocatable :: seed(:)

  call random_seed(size = n)
  allocate(seed(n))
  seed(:) = 1
  call random_seed(put=seed)
  call random_number(r)

  ! floor(r) will be 0, but this is only known at runtime. Statically, it is unknown
  cond = floor(r) == 0

  if (cond) then
    c = a
  else
    c = b
  endif

  c = c

end subroutine test_transform_region_const_prop_conditional_dynamic_condition
"""

    a_val = 5
    b_val = 3
    cond = True

    cond_val = '.True.' if cond else '.False.'
    routine = Subroutine.from_source(fcode.format(
        a_val=a_val, b_val=b_val, cond_val=cond_val),
        frontend=frontend)

    filepath = tmp_path / (f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    c = function()

    assert c == a_val if cond else b_val

    # Apply transformation
    routine = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 5
    assert len(FindNodes(Conditional).visit(routine.body)) == 1

    assignments = [str(a) for a in FindNodes(Assignment).visit(routine.body)]
    assert f'Assignment:: c = {a_val}' in assignments
    assert f'Assignment:: c = {b_val}' in assignments
    assert f'Assignment:: c = c' in assignments
    assert len([a for a in assignments if a == f'Assignment:: c = {a_val if cond else b_val}']) == 1

    # Test transformation

    new_filepath = tmp_path / f'{routine.name}_proped_{frontend}.f90'
    new_function = jit_compile(routine, filepath=new_filepath, objname=routine.name)

    c = new_function()

    assert c == a_val if cond else b_val

# TODO: Conditionals & Loop interactions
# TODO: Assignments