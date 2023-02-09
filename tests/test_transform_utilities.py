# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends
from loki.transform import (
    single_variable_declaration, recursive_expression_map_update, convert_to_lower_case
)
from loki import (
    Module, Subroutine, OMNI, FindNodes, VariableDeclaration, FindVariables,
    SubstituteExpressions, fgen
)
from loki.expression import symbols as sym


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Makes variable declaration already unique')]))
def test_transform_utilities_single_variable_declaration(frontend):
    """
    Test correct inlining of elemental functions.
    """
    fcode = """
subroutine foo(a, x, y)
    integer, intent(in) :: a
    real, intent(inout):: x(a), y(a, a)
    integer :: i1, i2, i3, i4
    real :: r1, r2, r3, r4
    x = a
    y = a
end subroutine foo
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    single_variable_declaration(routine=routine, variables=('y', 'i1', 'i3', 'r1', 'r2', 'r3', 'r4'))

    declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    assert declarations[0].symbols == ('a',)
    assert [smbl.name for smbl in declarations[1].symbols] == ['x']
    assert [smbl.name for smbl in declarations[2].symbols] == ['y']
    assert declarations[3].symbols == ('i2', 'i4')
    assert declarations[4].symbols == ('i1',)
    assert declarations[5].symbols == ('i3',)
    assert declarations[6].symbols == ('r1',)
    assert declarations[7].symbols == ('r2',)
    assert declarations[8].symbols == ('r3',)
    assert declarations[9].symbols == ('r4',)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Makes variable declaration already unique')]))
def test_transform_utilities_single_variable_declarations(frontend):
    """
    Test correct inlining of elemental functions.
    """
    fcode = """
subroutine foo(a, x, y)
    integer, intent(in) :: a
    real, intent(inout):: x(a), y(a, a)
    integer :: i1, i2, i3, i4
    real :: r1, r2, r3, r4
    real :: x1, x2(a), x3(a), x4(a, a)
    x = a
    y = a
end subroutine foo
"""
    # variables=None and group_by_shape=False, meaning all variable declarations to be unique
    routine = Subroutine.from_source(fcode, frontend=frontend)
    single_variable_declaration(routine=routine)

    declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    assert len(declarations) == 15
    for decl in declarations:
        assert len(decl.symbols) == 1

    # group_by_shape = False and variables=None, meaning only non-similar variable declarations unique
    routine = Subroutine.from_source(fcode, frontend=frontend)
    single_variable_declaration(routine=routine, group_by_shape=True)

    declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    assert len(declarations) == 8
    for decl in declarations:
        types = [smbl.type for smbl in decl.symbols]
        _ = [type == types[0] for type in types]
        assert all(_)
        if isinstance(decl.symbols[0], sym.Array):
            shapes = [smbl.shape for smbl in decl.symbols]
            _ = [shape == shapes[0] for shape in shapes]
            assert all(_)

    # group_by_shape = False and variables=('x2', 'r3'), meaning only non-similar variable declarations unique
    routine = Subroutine.from_source(fcode, frontend=frontend)
    single_variable_declaration(routine=routine, variables=('x2', 'r3'), group_by_shape=True)

    declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    assert len(declarations) == 10
    assert declarations[5].symbols == ('r3',)
    assert [smbl.name for smbl in declarations[8].symbols] == ['x2']
    for decl in declarations:
        types = [smbl.type for smbl in decl.symbols]
        _ = [type == types[0] for type in types]
        assert all(_)
        if isinstance(decl.symbols[0], sym.Array):
            shapes = [smbl.shape for smbl in decl.symbols]
            _ = [shape == shapes[0] for shape in shapes]
            assert all(_)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_convert_to_lower_case(frontend):
    fcode = """
subroutine my_NOT_ALL_lowercase_ROUTINE(VAR1, another_VAR, lower_case, MiXeD_CasE)
    implicit none
    integer, intent(in) :: VAR1, another_VAR
    integer, intent(inout) :: lower_case(ANOTHER_VAR)
    integer, intent(inout) :: MiXeD_CasE(Var1)
    integer :: J, k

    do J=1,VAR1
        mixed_CASE(J) = J + j
    end do

    do K=1,ANOTHER_VAR
        LOWER_CASE(K) = K - 1
    end do
end subroutine my_NOT_ALL_lowercase_ROUTINE
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    convert_to_lower_case(routine)
    assert all(var.name.islower() and str(var).islower() for var in FindVariables(unique=False).visit(routine.ir))


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_utilities_recursive_expression_map_update(frontend):
    fcode = """
module some_mod
    implicit none

    type some_type
        integer :: m, n
        real, allocatable :: a(:, :)
    contains
        procedure, pass :: my_add
    end type some_type
contains
    function my_add(self, data, val)
        class(some_type), intent(inout) :: self
        real, intent(in) :: data(:,:)
        real, value :: val
        real :: my_add(:,:)
        my_add(:,:) = self%a(:,:) + data(:,:) + val
    end function my_add

    subroutine do(my_obj)
        type(some_type), intent(inout) :: my_obj
        my_obj%a = my_obj%my_add(MY_OBJ%a(1:my_obj%m, 1:MY_OBJ%n), 1.)
    end subroutine do
end module some_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
    routine = module['do']

    expr_map = {}
    expr_map[routine.variable_map['my_obj']] = routine.variable_map['my_obj'].clone(name='obj')
    for var in FindVariables().visit(routine.body):
        if var.parent == 'my_obj':
            expr_map[var] = var.clone(name=f'obj%{var.basename}', parent=var.parent.clone(name='obj'))

    # There are "my_obj" nodes still around...
    assert any(
        var == 'my_obj' or var.parent == 'my_obj' for var in FindVariables().visit(list(expr_map.values()))
    )

    # ...and application performs only a partial substitution
    cloned = routine.clone()
    cloned.body = SubstituteExpressions(expr_map).visit(cloned.body)
    assert fgen(cloned.body.body[0]).lower() == 'obj%a = obj%my_add(obj%a(1:my_obj%m, 1:my_obj%n), 1.)'

    # Apply recursive update
    expr_map = recursive_expression_map_update(expr_map)

    # No more "my_obj" nodes...
    assert all(
        var != 'my_obj' and var.parent != 'my_obj' for var in FindVariables().visit(list(expr_map.values()))
    )

    # ...and full substitution
    assert fgen(routine.body.body[0]).lower() == 'my_obj%a = my_obj%my_add(my_obj%a(1:my_obj%m, 1:my_obj%n), 1.)'
    routine.body = SubstituteExpressions(expr_map).visit(routine.body)
    assert fgen(routine.body.body[0]) == 'obj%a = obj%my_add(obj%a(1:obj%m, 1:obj%n), 1.)'
