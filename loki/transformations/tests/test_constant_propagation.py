# (C) Copyright 2024- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, available_frontends, jit_compile
from loki.expression import symbols as sym
from loki.ir import nodes as ir, FindNodes

from loki.transformations.constant_propagation import (
    ConstantPropagationTransformer, ConstantPropagationMapper, do_constant_propagation
)


def test_constant_propagation_analysis_declarations_map():
    fcode = """
subroutine const_prop_decls
  integer :: a = 1
  integer :: b(2) = (/2, 3/)
  logical :: l = .true.
end subroutine const_prop_decls
    """.strip()
    routine = Subroutine.from_source(fcode)

    declarations_map = ConstantPropagationTransformer().generate_declarations_map(routine)

    assert declarations_map[('a', ())] == sym.IntLiteral(1)
    assert declarations_map[('l', ())] == sym.LogicLiteral(True)
    assert declarations_map[('b', (sym.IntLiteral(1),))] == sym.IntLiteral(2)
    assert declarations_map[('b', (sym.IntLiteral(2),))] == sym.IntLiteral(3)


def test_constant_propagation_mapper_folds_expressions():
    mapper = ConstantPropagationMapper()

    assert mapper(sym.Sum((sym.IntLiteral(1), sym.IntLiteral(2)))) == sym.IntLiteral(3)
    assert mapper(sym.Quotient(sym.IntLiteral(7), sym.IntLiteral(2))) == sym.IntLiteral(3)
    assert mapper(sym.Quotient(sym.IntLiteral(-7), sym.IntLiteral(2))) == sym.IntLiteral(-3)
    assert mapper(sym.Power(sym.IntLiteral(2), sym.IntLiteral(3))) == sym.IntLiteral(8)
    assert mapper(sym.LogicalAnd((sym.LogicLiteral(True), sym.LogicLiteral(False)))) == sym.LogicLiteral(False)
    assert mapper(sym.StringConcat((sym.StringLiteral('foo'), sym.StringLiteral('bar')))) == sym.StringLiteral('foobar')
    assert mapper(sym.Sum((sym.FloatLiteral('1.5'), sym.FloatLiteral('2.5')))) == sym.FloatLiteral('4.0')


def test_constant_propagation_mapper_short_circuits_boolean_ops():
    mapper = ConstantPropagationMapper()
    dyn = sym.Variable(name='dyn')

    assert mapper(sym.LogicalOr((sym.LogicLiteral(True), dyn))) == sym.LogicLiteral(True)
    assert mapper(sym.LogicalAnd((sym.LogicLiteral(False), dyn))) == sym.LogicLiteral(False)


def test_constant_propagation_analysis_dynamic_array_invalidation():
    fcode = """
subroutine const_prop_dynamic_array(a, i)
  integer, intent(inout) :: a(3)
  integer, intent(in) :: i
  a(1) = 1
  a(2) = 2
  a(i) = 5
end subroutine const_prop_dynamic_array
    """.strip()
    routine = Subroutine.from_source(fcode)
    assignments = FindNodes(ir.Assignment).visit(routine.body)

    analysis = ConstantPropagationTransformer()
    constants_map = {
        ('a', (sym.IntLiteral(1),)): sym.IntLiteral(1),
        ('a', (sym.IntLiteral(2),)): sym.IntLiteral(2),
        ('a', (sym.IntLiteral(3),)): sym.IntLiteral(3),
    }

    analysis.visit(assignments[-1], constants_map=constants_map)

    assert ('a', (sym.IntLiteral(1),)) not in constants_map
    assert ('a', (sym.IntLiteral(2),)) not in constants_map
    assert ('a', (sym.IntLiteral(3),)) not in constants_map


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_literals_expected_future(frontend):
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

    transformed = do_constant_propagation(routine)
    assignments = [str(a) for a in FindNodes(ir.Assignment).visit(transformed.body)]

    assert 'Assignment:: a = 1' in assignments
    assert 'Assignment:: b = 1.5' in assignments
    assert 'Assignment:: c = \'foo\'' in assignments
    assert 'Assignment:: d = True' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_ops_int_expected_future(tmp_path, frontend):
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

    transformed = do_constant_propagation(routine)
    assignments = [str(a) for a in FindNodes(ir.Assignment).visit(transformed.body)]

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
def test_constant_propagation_ops_bool_short_circuiting_expected_future(tmp_path, frontend):
    fcode = """
subroutine test_constant_propagation_ops_bool_short_circuiting(a_and, a_or)
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

end subroutine test_constant_propagation_ops_bool_short_circuiting
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    outputs = function()

    assert (outputs[0] == 1) is False
    assert (outputs[1] == 1) is True

    transformed = do_constant_propagation(routine)
    assignments = [str(a) for a in FindNodes(ir.Assignment).visit(transformed.body)]

    assert 'Assignment:: a_and = False' in assignments
    assert 'Assignment:: a_or = True' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_conditional_basic(frontend):
    fcode = """
subroutine test_constant_propagation_conditional_basic(c, d)
  integer :: a = 5
  integer :: b = 3
  logical :: cond = .true.
  integer, intent(out) :: c, d

  if (cond) then
    c = a
    d = a
  else
    c = b
    d = a
  endif

  c = c
  d = d
end subroutine test_constant_propagation_conditional_basic
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    transformed = do_constant_propagation(routine)

    assert len(FindNodes(ir.Conditional).visit(transformed.body)) == 1
    assignments = [str(a) for a in FindNodes(ir.Assignment).visit(transformed.body)]
    assert 'Assignment:: c = 5' in assignments
    assert 'Assignment:: c = 3' in assignments
    assert 'Assignment:: c = c' in assignments
    assert 'Assignment:: c = 5' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_for_loop_basic(frontend):
    fcode = """
subroutine test_constant_propagation_for_loop_basic(c, d)
  integer :: a = 5
  integer :: b = 3
  integer :: i, j
  integer, intent(out) :: c
  integer, intent(inout) :: d

  c = 0
  do i = 1, a
    c = c + b
  end do

  do j = 1, d
    c = a + d
  end do

  d = c
end subroutine test_constant_propagation_for_loop_basic
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # First, propagate the initial constatns, then resolve the loop
    transformed = do_constant_propagation(routine, unroll_loops=True)

    loops = FindNodes(ir.Loop).visit(transformed.body)
    assert len(loops) == 1
    assert loops[0].variable == 'j'

    # Check that constant-bounds loop is unrolled
    assignments = [str(a) for a in FindNodes(ir.Assignment).visit(transformed.body)]
    for i in range(1, 6):
        assert f'Assignment:: c = {3*i}' in assignments

    # Check that non-constant-bounds loop is not unrolled and that
    # this loop invalidates the value of `c` before final assignment.
    assert len(assignments) == 8
    assert assignments[6] == 'Assignment:: c = 5 + d'
    assert assignments[7] == 'Assignment:: d = c'


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_for_loop_basic_no_unroll(frontend):
    fcode = """
subroutine test_constant_propagation_for_loop_basic_no_unroll(c)
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
end subroutine test_constant_propagation_for_loop_basic_no_unroll
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    transformed = do_constant_propagation(routine, unroll_loops=False)

    assignments = [str(a) for a in FindNodes(ir.Assignment).visit(transformed.body)]
    assert 'Assignment:: c = 15' in assignments
    assert 'Assignment:: d = 5*i' in assignments
    assert 'Assignment:: c = 30' in assignments


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_propagation_for_loop_nested_siblings_no_unroll(frontend):
    fcode = """
subroutine test_constant_propagation_loop_nested_siblings_no_unroll(c)
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
end subroutine test_constant_propagation_loop_nested_siblings_no_unroll
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    transformed = do_constant_propagation(routine, unroll_loops=False)

    assignments = [str(a) for a in FindNodes(ir.Assignment).visit(transformed.body)]
    assert 'Assignment:: c = 5' in assignments
    assert 'Assignment:: c = 3' in assignments
