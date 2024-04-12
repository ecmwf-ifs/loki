# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    Module, Subroutine, FindNodes, VariableDeclaration, FindVariables,
    SubstituteExpressions, fgen, FindInlineCalls
)
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.transform import (
    single_variable_declaration, recursive_expression_map_update,
    convert_to_lower_case, replace_intrinsics, rename_variables
)


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
    integer, intent(inout) :: MiXeD_CasE(Var1, ANOTHER_VAR)
    integer :: J, k

    do k=1,ANOTHER_VAR
        do J=1,VAR1
            mixed_CASE(J, K) = J + K
        end do
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

@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Argument mismatch for "min"')]))
def test_transform_utilites_replace_intrinsics(frontend):
    fcode = """
subroutine replace_intrinsics()
    implicit none
    real :: a, b, eps
    real, parameter :: param = min(0.1, epsilon(param)*1000.)

    eps = param * 10.
    eps = 0.1
    b = max(10., eps)
    a = min(1. + b, 1. - eps)

end subroutine replace_intrinsics
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    symbol_map = {'epsilon': 'DBL_EPSILON'}
    function_map = {'min': 'fmin', 'max': 'fmax'}
    replace_intrinsics(routine, symbol_map=symbol_map, function_map=function_map)
    inline_calls = FindInlineCalls(unique=False).visit(routine.ir)
    assert inline_calls[0].name == 'fmin'
    assert inline_calls[1].name == 'fmax'
    assert inline_calls[2].name == 'fmin'
    variables = FindVariables(unique=False).visit(routine.ir)
    assert 'DBL_EPSILON' in variables
    assert 'epsilon' not in variables
    # check wether it really worked for variable declarations or rather parameters
    assert 'DBL_EPSILON' in FindVariables().visit(routine.variable_map['param'].initial)

@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_utilites_rename_variables(frontend):
    fcode = """
subroutine rename_variables(some_arg, rename_arg)
    implicit none
    integer, intent(inout) :: some_arg, rename_arg
    integer :: some_var, rename_var
    integer :: i, j
    real :: some_array(10, 10), rename_array(10, 10)

    do i=1,10
        some_var = i
        rename_var = i + 1
        do J=1,10
            some_array(i, j) = 10. * some_arg * rename_arg
	        rename_array(i, j) = 5. * some_arg * rename_arg
        end do
    end do

end subroutine rename_variables
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    symbol_map = {'rename_var': 'renamed_var',
                  'rename_arg': 'renamed_arg',
                  'rename_array': 'renamed_array'}
    rename_variables(routine, symbol_map=symbol_map)
    variables = [var.name for var in FindVariables(unique=False).visit(routine.ir)]
    assert 'renamed_var' in variables
    assert 'rename_var'  not in variables
    assert 'renamed_arg' in variables
    assert 'rename_arg' not in variables
    assert 'renamed_array' in variables
    assert 'rename_array' not in variables
    # check routine arguments
    assert 'renamed_arg' in routine.arguments
    assert 'rename_arg' not in routine.arguments
    # check symbol table
    assert 'renamed_arg' in routine.symbol_attrs
    assert 'rename_arg' not in routine.symbol_attrs
    assert 'renamed_array' in routine.symbol_attrs
    assert 'rename_array' not in routine.symbol_attrs
    assert 'renamed_arg' in routine.symbol_attrs
    assert 'rename_arg' not in routine.symbol_attrs

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_utilites_rename_variables_extended(frontend):
    fcode = """
subroutine rename_variables_extended(KLON, ARR, TT)
    implicit none
    
    INTEGER, INTENT(IN) :: KLON
    REAL, INTENT(INOUT) :: ARR(KLON)
    REAL :: MY_TMP(KLON)
    TYPE(SOME_TYPE), INTENT(INOUT) :: TT
    TYPE(OTHER_TYPE) :: TMP_TT

    TMP_TT%SOME_MEMBER = TT%SOME_MEMBER + TT%PROC_FUNC(5.0)
    CALL TT%NESTED%PROC_SUB(TT%NESTED%VAR)
    TT%VAL = TMP_TT%VAL

end subroutine rename_variables_extended
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    symbol_map = {'klon': 'ncol', 'tt': 'arg_tt'}
    rename_variables(routine, symbol_map=symbol_map)
    # check arguments
    arguments = [arg.name.lower() for arg in routine.arguments]
    assert 'ncol' in arguments
    assert 'klon' not in arguments
    assert 'arg_tt' in arguments
    assert 'tt' not in arguments
    # check array shape
    assert routine.variable_map['arr'].shape == ('ncol',)
    assert routine.variable_map['my_tmp'].shape == ('ncol',)
    # check variables
    variables = [var.name.lower() for var in FindVariables(unique=False).visit(routine.ir)]
    assert 'ncol' in variables
    assert 'klon' not in variables
    assert 'arg_tt' in variables
    assert 'tt' not in variables
    assert 'arg_tt%some_member' in variables
    assert 'tt%some_member' not in variables
    assert 'arg_tt%proc_func' in variables
    assert 'tt%proc_func' not in variables
    assert 'arg_tt%nested' in variables
    assert 'tt%nested' not in variables
    assert 'arg_tt%nested%proc_sub' in variables
    assert 'tt%nested%proc_sub' not in variables
    assert 'arg_tt%nested%var' in variables
    assert 'tt%nested%var' not in variables
    # check symbol table
    routine_symbol_attrs_name = tuple(key.lower() for key in routine.symbol_attrs)+\
            tuple(key.split('%')[0].lower() for key in routine.symbol_attrs)
    assert 'ncol' in routine_symbol_attrs_name
    assert 'klon' not in routine_symbol_attrs_name
    assert 'arg_tt' in routine_symbol_attrs_name
    assert 'tt' not in routine_symbol_attrs_name
