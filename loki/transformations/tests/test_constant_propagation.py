# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import FindNodes, Subroutine
from loki.frontend import available_frontends
from loki.ir import Assignment, Conditional, Loop
from loki.jit_build import jit_compile
from loki.transformations import ConstantPropagationTransformer


def test_constant_propagation_transformer_export():
    assert ConstantPropagationTransformer is not None


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_literals_expected_future(tmp_path, frontend):
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
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    function()

    transformed = ConstantPropagationTransformer().visit(routine)
    assignments = [str(a) for a in FindNodes(Assignment).visit(transformed.body)]

    assert 'Assignment:: a = 1' in assignments
    assert 'Assignment:: b = 1.5' in assignments
    assert 'Assignment:: c = \'foo\'' in assignments
    assert 'Assignment:: d = True' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_ops_int_expected_future(tmp_path, frontend):
    fcode = """
subroutine const_prop_ops_int(a_add, a_sub, a_mul, a_pow, a_div, a_lt, a_leq, a_eq, a_neq, a_geq, a_gt)
  integer :: a = 1
  integer :: b = 2
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

    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    outputs = function()

    assert outputs[0] == 3
    assert outputs[1] == -1
    assert outputs[2] == 2
    assert outputs[3] == 1
    assert outputs[4] == 0

    transformed = ConstantPropagationTransformer().visit(routine)
    assignments = [str(a) for a in FindNodes(Assignment).visit(transformed.body)]

    assert 'Assignment:: a_add = 3' in assignments
    assert 'Assignment:: a_sub = -1' in assignments
    assert 'Assignment:: a_mul = 2' in assignments
    assert 'Assignment:: a_pow = 1' in assignments
    assert 'Assignment:: a_div = 0' in assignments
    assert 'Assignment:: a_lt = True' in assignments
    assert 'Assignment:: a_leq = True' in assignments
    assert 'Assignment:: a_eq = False' in assignments
    assert 'Assignment:: a_neq = True' in assignments
    assert 'Assignment:: a_geq = False' in assignments
    assert 'Assignment:: a_gt = False' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_ops_bool_short_circuiting_expected_future(tmp_path, frontend):
    fcode = """
subroutine test_transform_region_const_prop_ops_bool_short_circuiting(a_and, a_or)
  logical :: a = .true.
  logical :: b
  logical, intent(out) :: a_and, a_or

  integer :: n
  real :: r
  integer, allocatable :: seed(:)

  call random_seed(size = n)
  allocate(seed(n))
  seed(:) = 1
  call random_seed(put=seed)
  call random_number(r)

  b = floor(r) == 0

  a_and = .not. a .and. b
  a_or = a .or. b

end subroutine test_transform_region_const_prop_ops_bool_short_circuiting
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    outputs = function()

    assert (outputs[0] == 1) is False
    assert (outputs[1] == 1) is True

    transformed = ConstantPropagationTransformer().visit(routine)
    assignments = [str(a) for a in FindNodes(Assignment).visit(transformed.body)]

    assert 'Assignment:: a_and = False' in assignments
    assert 'Assignment:: a_or = True' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_conditional_basic(frontend):
    fcode = """
subroutine test_transform_region_const_prop_conditional_basic(c)
  integer :: a = 5
  integer :: b = 3
  logical :: cond = .true.
  integer, intent(out) :: c

  if (cond) then
    c = a
  else
    c = b
  endif

  c = c
end subroutine test_transform_region_const_prop_conditional_basic
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    transformed = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assert len(FindNodes(Conditional).visit(transformed.body)) == 1
    assignments = [str(a) for a in FindNodes(Assignment).visit(transformed.body)]
    assert 'Assignment:: c = 5' in assignments
    assert 'Assignment:: c = 3' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_basic(frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_basic(c)
  integer :: a = 5
  integer :: b = 3
  integer :: i
  integer, intent(out) :: c

  c = 0
  do i = 1, a
    c = c + b
  end do
end subroutine test_transform_region_const_prop_for_loop_basic
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    transformed = ConstantPropagationTransformer().visit(routine)

    assert len(FindNodes(Loop).visit(transformed.body)) == 0
    assignments = [str(a) for a in FindNodes(Assignment).visit(transformed.body)]
    for i in range(1, 6):
        assert f'Assignment:: c = {3*i}' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_basic_no_unroll(frontend):
    fcode = """
subroutine test_transform_region_const_prop_for_loop_basic_no_unroll(c)
  integer :: a = 5
  integer :: b = 3
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
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    transformed = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assignments = [str(a) for a in FindNodes(Assignment).visit(transformed.body)]
    assert 'Assignment:: c = 15' in assignments
    assert 'Assignment:: d = 5*i' in assignments
    assert 'Assignment:: c = 30' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_const_prop_for_loop_nested_siblings_no_unroll(frontend):
    fcode = """
subroutine test_transform_region_const_prop_loop_nested_siblings_no_unroll(c)
  integer :: a = 5
  integer :: b = 3
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
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    transformed = ConstantPropagationTransformer(unroll_loops=False).visit(routine)

    assignments = [str(a) for a in FindNodes(Assignment).visit(transformed.body)]
    assert 'Assignment:: c = 5' in assignments
    assert 'Assignment:: c = 3' in assignments
