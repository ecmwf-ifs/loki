# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

"""
Verify correct frontend behaviour and correct parsing of certain Fortran
language features.
"""

# pylint: disable=too-many-lines

import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import pytest

from loki import (
    Module, Subroutine, FindVariables, BasicType, config, Sourcefile,
    RawSource, RegexParserClass, ProcedureType, DerivedType,
    PreprocessorDirective, config_override
)
from loki.build import jit_compile, clean_test
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI, OFP, FP, REGEX
from loki.ir import nodes as ir, FindNodes


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


@pytest.fixture(name='reset_frontend_mode')
def fixture_reset_frontend_mode():
    original_frontend_mode = config['frontend-strict-mode']
    yield
    config['frontend-strict-mode'] = original_frontend_mode


@pytest.fixture(name='reset_regex_frontend_timeout')
def fixture_reset_regex_frontend_timeout():
    original_timeout = config['regex-frontend-timeout']
    yield
    config['regex-frontend-timeout'] = original_timeout


@pytest.mark.parametrize('frontend', available_frontends())
def test_check_alloc_opts(tmp_path, frontend):
    """
    Test the use of SOURCE and STAT in allocate
    """

    fcode = """
module alloc_mod
    integer, parameter :: jprb = selected_real_kind(13,300)

    type explicit
        real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
        real(kind=jprb) :: red_herring
    end type explicit

    type deferred
        real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
        real(kind=jprb), allocatable :: red_herring
    end type deferred

contains

    subroutine alloc_deferred(item)
        type(deferred), intent(inout) :: item
        integer :: stat
        allocate(item%vector(3), stat=stat)
        allocate(item%matrix(3, 3))
    end subroutine alloc_deferred

    subroutine free_deferred(item)
        type(deferred), intent(inout) :: item
        integer :: stat
        deallocate(item%vector, stat=stat)
        deallocate(item%matrix)
    end subroutine free_deferred

    subroutine check_alloc_source(item, item2)
        type(explicit), intent(inout) :: item
        type(deferred), intent(inout) :: item2
        real(kind=jprb), allocatable :: vector(:), vector2(:)

        allocate(vector, source=item%vector)
        vector(:) = vector(:) + item%scalar
        item%vector(:) = vector(:)

        allocate(vector2, source=item2%vector)  ! Try mold here when supported by fparser
        vector2(:) = item2%scalar
        item2%vector(:) = vector2(:)
    end subroutine check_alloc_source
end module alloc_mod
    """.strip()

    # Parse the source and validate the IR
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    allocations = FindNodes(ir.Allocation).visit(module['check_alloc_source'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is not None for alloc in allocations)
    assert all(alloc.status_var is None for alloc in allocations)

    allocations = FindNodes(ir.Allocation).visit(module['alloc_deferred'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is None for alloc in allocations)
    assert allocations[0].status_var is not None
    assert allocations[1].status_var is None

    deallocs = FindNodes(ir.Deallocation).visit(module['free_deferred'].body)
    assert len(deallocs) == 2
    assert deallocs[0].status_var is not None
    assert deallocs[1].status_var is None

    # Sanity check for the backend
    assert module.to_fortran().lower().count(', stat=stat') == 2

    # Generate Fortran and test it
    filepath = tmp_path/(f'frontends_check_alloc_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='alloc_mod')

    item = mod.explicit()
    item.scalar = 1.
    item.vector[:] = 1.

    item2 = mod.deferred()
    mod.alloc_deferred(item2)
    item2.scalar = 2.
    item2.vector[:] = -1.

    mod.check_alloc_source(item, item2)
    assert (item.vector == 2.).all()
    assert (item2.vector == 2.).all()
    mod.free_deferred(item2)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_associates(tmp_path, frontend):
    """Test the use of associate to access and modify other items"""

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine associates(item)
    type(explicit), intent(inout) :: item
    type(deferred) :: item2

    item%scalar = 17.0

    associate(vector2=>item%matrix(:,1))
        vector2(:) = 3.
        item%matrix(:,3) = vector2(:)
    end associate

    associate(vector=>item%vector)
        item%vector(2) = vector(1)
        vector(3) = item%vector(1) + vector(2)
        vector(1) = 1.
    end associate

    call alloc_deferred(item2)

    associate(vec=>item2%vector(2))
        vec = 1.
    end associate

    call free_deferred(item2)
  end subroutine associates
end module
"""
    # Test the internals
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['associates']
    variables = FindVariables().visit(routine.body)
    assert all(
        v.shape == ('3',) for v in variables if v.name in ['vector', 'vector2']
    )

    for assoc in FindNodes(ir.Associate).visit(routine.body):
        for var in FindVariables().visit(assoc.body):
            if var.name in assoc.variables:
                assert var.scope is assoc
                assert var.type.parent is None
            else:
                assert var.scope is routine

    # Test the generated module
    filepath = tmp_path/(f'derived_types_associates_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    mod.associates(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI fails to read without full module')]))
def test_associates_deferred(frontend):
    """
    Verify that reading in subroutines with deferred external type definitions
    and associates working on that are supported.
    """

    fcode = """
SUBROUTINE ASSOCIATES_DEFERRED(ITEM, IDX)
USE SOME_MOD, ONLY: SOME_TYPE
IMPLICIT NONE
TYPE(SOME_TYPE), INTENT(IN) :: ITEM
INTEGER, INTENT(IN) :: IDX
ASSOCIATE(SOME_VAR=>ITEM%SOME_VAR(IDX), SOME_OTHER_VAR=>ITEM%SOME_VAR(ITEM%OFFSET))
SOME_VAR = 5
END ASSOCIATE
END SUBROUTINE
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 6
    some_var = variables['SOME_VAR']
    assert isinstance(some_var, sym.DeferredTypeSymbol)
    assert some_var.name.upper() == 'SOME_VAR'
    assert some_var.type.dtype == BasicType.DEFERRED
    associate = FindNodes(ir.Associate).visit(routine.body)[0]
    assert some_var.scope is associate

    some_other_var = variables['SOME_OTHER_VAR']
    assert isinstance(some_var, sym.DeferredTypeSymbol)
    assert some_other_var.name.upper() == 'SOME_OTHER_VAR'
    assert some_other_var.type.dtype == BasicType.DEFERRED
    assert some_other_var.type.shape == ('ITEM%OFFSET',)
    assert some_other_var.scope is associate


@pytest.mark.parametrize('frontend', available_frontends())
def test_associates_expr(tmp_path, frontend):
    """Verify that associates with expressions are supported"""
    fcode = """
subroutine associates_expr(in, out)
  implicit none
  integer, intent(in) :: in(3)
  integer, intent(out) :: out(3)

  out(:) = 0

  associate(a=>1+3)
    out(:) = out(:) + a
  end associate

  associate(b=>2*in(:) + in(:))
    out(:) = out(:) + b(:)
  end associate
end subroutine associates_expr
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 4
    assert isinstance(variables['a'], sym.DeferredTypeSymbol)
    assert variables['a'].type.dtype is BasicType.DEFERRED  # TODO: support type derivation for expressions
    assert isinstance(variables['b'], sym.Array)  # Note: this is an array because we have a shape
    assert variables['b'].type.dtype is BasicType.DEFERRED  # TODO: support type derivation for expressions
    assert variables['b'].type.shape == ('3',)

    filepath = tmp_path/(f'associates_expr_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    a = np.array([1, 2, 3], dtype='i')
    b = np.zeros(3, dtype='i')
    function(a, b)
    assert np.all(b == [7, 10, 13])
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_enum(tmp_path, frontend):
    """Verify that enums are represented correctly"""
    # F2008, Note 4.67
    fcode = """
subroutine test_enum (out)
    implicit none

    ! Comment 1
    ENUM, BIND(C)
        ENUMERATOR :: RED = 4, BLUE = 9
        ! Comment 2
        ENUMERATOR YELLOW
    END ENUM
    ! Comment 3

    integer, intent(out) :: out

    out = RED + BLUE + YELLOW
end subroutine test_enum
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Check Enum exists
    enums = FindNodes(ir.Enumeration).visit(routine.spec)
    assert len(enums) == 1

    # Check symbols are available
    assert enums[0].symbols == ('red', 'blue', 'yellow')
    assert all(name in routine.symbols for name in ('red', 'blue', 'yellow'))
    assert all(s.scope is routine for s in enums[0].symbols)

    # Check assigned values
    assert routine.symbol_map['red'].type.initial == '4'
    assert routine.symbol_map['blue'].type.initial == '9'
    assert routine.symbol_map['yellow'].type.initial is None

    # Verify comments are preserved (don't care about the actual place)
    code = routine.to_fortran()
    for i in range(1, 4):
        assert f'! Comment {i}' in code

    # Check fgen produces valid code and runs
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    out = function()
    assert out == 23
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OFP, 'OFP fails to parse parameterized types')]
))
@pytest.mark.usefixtures('reset_frontend_mode')
def test_frontend_strict_mode(frontend, tmp_path):
    """
    Verify that frontends fail on unsupported features if strict mode is enabled
    """
    # Parameterized derived types currently not implemented
    fcode = """
module frontend_strict_mode
    implicit none
    TYPE matrix ( k, b )
      INTEGER,     KIND :: k = 4
      INTEGER (8), LEN  :: b
      REAL (k)          :: element (b,b)
    END TYPE matrix
end module frontend_strict_mode
    """
    config['frontend-strict-mode'] = True
    with pytest.raises(NotImplementedError):
        Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    config['frontend-strict-mode'] = False
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert 'matrix' in module.symbol_attrs
    assert 'matrix' in module.typedef_map


def test_regex_subroutine_from_source():
    """
    Verify that the regex frontend is able to parse subroutines
    """
    fcode = """
subroutine routine_b(
    ! arg 1
    i,
    ! arg2
    j
)
    use parkind1, only : jpim
    implicit none
    integer, intent(in) :: i, j
    integer b
    b = 4

    call contained_c(i)

    call routine_a()
contains
!abc ^$^**
    integer(kind=jpim) function contained_e(i)
        integer, intent(in) :: i
        contained_e = i
    end function

    subroutine contained_c(i)
        integer, intent(in) :: i
        integer c
        c = 5
    end subroutine contained_c
    ! cc£$^£$^

    subroutine contained_d(i)
        integer, intent(in) :: i
        integer c
        c = 8
    end subroutine !add"£^£$
end subroutine routine_b
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=REGEX)
    assert routine.name == 'routine_b'
    assert not routine.is_function
    assert routine.arguments == ()
    assert routine.argnames == []
    assert [r.name for r in routine.subroutines] == ['contained_e', 'contained_c', 'contained_d']

    contained_c = routine['contained_c']
    assert contained_c.name == 'contained_c'
    assert not contained_c.is_function
    assert contained_c.arguments == ()
    assert contained_c.argnames == []

    contained_e = routine['contained_e']
    assert contained_e.name == 'contained_e'
    assert contained_e.is_function
    assert contained_e.arguments == ()
    assert contained_e.argnames == []

    contained_d = routine['contained_d']
    assert contained_d.name == 'contained_d'
    assert not contained_d.is_function
    assert contained_d.arguments == ()
    assert contained_d.argnames == []

    code = routine.to_fortran()
    assert code.count('SUBROUTINE') == 6
    assert code.count('FUNCTION') == 2
    assert code.count('CONTAINS') == 1


def test_regex_module_from_source():
    """
    Verify that the regex frontend is able to parse modules
    """
    fcode = """
module some_module
    use foobar
    implicit none
    integer, parameter :: k = selected_int_kind(5)
contains
    subroutine module_routine
        integer m
        m = 2

        call routine_b(m, 6)
    end subroutine module_routine

    integer(kind=k) function module_function(n)
        integer n
        module_function = n + 2
    end function module_function
end module some_module
    """.strip()

    module = Module.from_source(fcode, frontend=REGEX)
    assert module.name == 'some_module'
    assert [r.name for r in module.subroutines] == ['module_routine', 'module_function']

    code = module.to_fortran()
    assert code.count('MODULE') == 2
    assert code.count('SUBROUTINE') == 2
    assert code.count('FUNCTION') == 2
    assert code.count('CONTAINS') == 1


def test_regex_sourcefile_from_source():
    """
    Verify that the regex frontend is able to parse source files containing
    multiple modules and subroutines
    """
    fcode = """
subroutine routine_a
    integer a, i
    a = 1
    i = a + 1

    call routine_b(a, i)
end subroutine routine_a

module some_module
contains
    subroutine module_routine
        integer m
        m = 2

        call routine_b(m, 6)
    end subroutine   module_routine

    function module_function(n)
        integer n
        integer module_function
        module_function = n + 3
    end function   module_function
end module some_module

module other_module
    integer :: n
end module

subroutine routine_b(
    ! arg 1
    i,
    ! arg2
    j,
    k!arg3
)
  integer, intent(in) :: i, j, k
  integer b
  b = 4

  call contained_c(i)

  call routine_a()
contains
!abc ^$^**
    subroutine contained_c(i)
        integer, intent(in) :: i
        integer c
        c = 5
    end subroutine contained_c
    ! cc£$^£$^
    integer function contained_e(i)
        integer, intent(in) :: i
        contained_e = i
    end function

    subroutine contained_d(i)
        integer, intent(in) :: i
        integer c
        c = 8
    end subroutine !add"£^£$
endsubroutine  routine_b

function function_d(d)
    integer d
    d = 6
end function function_d

module last_module
    implicit none
contains
    subroutine last_routine1
        call contained()
        contains
        subroutine contained
        integer n
        n = 1
        end subroutine contained
    end subroutine last_routine1
    subroutine last_routine2
        call contained2()
        contains
        subroutine contained2
        integer m
        m = 1
        end subroutine contained2
    end subroutine last_routine2
end module last_module
    """.strip()

    sourcefile = Sourcefile.from_source(fcode, frontend=REGEX)
    assert [m.name for m in sourcefile.modules] == ['some_module', 'other_module', 'last_module']
    assert [r.name for r in sourcefile.routines] == [
        'routine_a', 'routine_b', 'function_d'
    ]
    assert [r.name for r in sourcefile.all_subroutines] == [
        'routine_a', 'routine_b', 'function_d', 'module_routine', 'module_function',
        'last_routine1', 'last_routine2'
    ]

    assert len(r := sourcefile['last_module']['last_routine1'].routines) == 1 and r[0].name == 'contained'
    assert len(r := sourcefile['last_module']['last_routine2'].routines) == 1 and r[0].name == 'contained2'

    code = sourcefile.to_fortran()
    assert code.count('SUBROUTINE') == 18
    assert code.count('FUNCTION') == 6
    assert code.count('CONTAINS') == 5
    assert code.count('MODULE') == 6


def test_regex_sourcefile_from_file(testdir):
    """
    Verify that the regex frontend is able to parse source files containing
    multiple modules and subroutines
    """

    sourcefile = Sourcefile.from_file(testdir/'sources/sourcefile.f90', frontend=REGEX)
    assert [m.name for m in sourcefile.modules] == ['some_module']
    assert [r.name for r in sourcefile.routines] == [
        'routine_a', 'routine_b', 'function_d'
    ]
    assert [r.name for r in sourcefile.all_subroutines] == [
        'routine_a', 'routine_b', 'function_d', 'module_routine', 'module_function'
    ]

    routine_b = sourcefile['ROUTINE_B']
    assert routine_b.name == 'routine_b'
    assert not routine_b.is_function
    assert routine_b.arguments == ()
    assert routine_b.argnames == []
    assert [r.name for r in routine_b.subroutines] == ['contained_c']

    function_d = sourcefile['function_d']
    assert function_d.name == 'function_d'
    assert function_d.is_function
    assert function_d.arguments == ()
    assert function_d.argnames == []
    assert not function_d.contains

    code = sourcefile.to_fortran()
    assert code.count('SUBROUTINE') == 8
    assert code.count('FUNCTION') == 4
    assert code.count('CONTAINS') == 2
    assert code.count('MODULE') == 2


def test_regex_sourcefile_from_file_parser_classes(testdir):

    filepath = testdir/'sources/Fortran-extract-interface-source.f90'
    module_names = {'bar', 'foo'}
    routine_names = {
        'func_simple', 'func_simple_1', 'func_simple_2', 'func_simple_pure', 'func_simple_recursive_pure',
        'func_simple_elemental', 'func_with_use_and_args', 'func_with_parameters', 'func_with_parameters_1',
        'func_with_contains', 'func_mix_local_and_result', 'sub_simple', 'sub_simple_1', 'sub_simple_2',
        'sub_simple_3', 'sub_with_contains', 'sub_with_renamed_import', 'sub_with_external', 'sub_with_end'
    }
    module_routine_names = {'foo_sub', 'foo_func'}

    # Empty parse (since we don't match typedef without having the enclosing module first)
    sourcefile = Sourcefile.from_file(filepath, frontend=REGEX, parser_classes=RegexParserClass.TypeDefClass)
    assert not sourcefile.subroutines
    assert not sourcefile.modules
    assert FindNodes(RawSource).visit(sourcefile.ir)
    assert sourcefile._incomplete
    assert sourcefile._parser_classes == RegexParserClass.TypeDefClass

    # Incremental addition of program unit objects
    sourcefile.make_complete(frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)
    assert sourcefile._incomplete
    assert sourcefile._parser_classes == RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass
    # Note that the program unit objects don't include the TypeDefClass because it's lower in the hierarchy
    # and was not matched previously
    assert all(
        module._parser_classes == RegexParserClass.ProgramUnitClass
        for module in sourcefile.modules
    )
    assert all(
        routine._parser_classes == RegexParserClass.ProgramUnitClass
        for routine in sourcefile.routines
    )

    assert {module.name.lower() for module in sourcefile.modules} == module_names
    assert {routine.name.lower() for routine in sourcefile.routines} == routine_names
    assert {routine.name.lower() for routine in sourcefile.all_subroutines} == routine_names | module_routine_names

    assert {routine.name.lower() for routine in sourcefile['func_with_contains'].routines} == {'func_with_contains_1'}
    assert {routine.name.lower() for routine in sourcefile['sub_with_contains'].routines} == {
        'sub_with_contains_first', 'sub_with_contains_second', 'sub_with_contains_third'
    }

    for module in sourcefile.modules:
        assert not module.imports
    for routine in sourcefile.all_subroutines:
        assert not routine.imports
    assert not sourcefile['bar'].typedefs

    # Validate that a re-parse with same parser classes does not change anything
    sourcefile.make_complete(frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)
    assert sourcefile._incomplete
    assert sourcefile._parser_classes == RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass
    for module in sourcefile.modules:
        assert not module.imports
    for routine in sourcefile.all_subroutines:
        assert not routine.imports
    assert not sourcefile['bar'].typedefs

    # Incremental addition of imports
    sourcefile.make_complete(
        frontend=REGEX,
        parser_classes=RegexParserClass.ProgramUnitClass | RegexParserClass.ImportClass
    )
    assert sourcefile._parser_classes == (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.ImportClass
    )
    # Note that the program unit objects don't include the TypeDefClass because it's lower in the hierarchy
    # and was not matched previously
    assert all(
        module._parser_classes == (
            RegexParserClass.ProgramUnitClass | RegexParserClass.ImportClass
        ) for module in sourcefile.modules
    )
    assert all(
        routine._parser_classes == (
            RegexParserClass.ProgramUnitClass | RegexParserClass.ImportClass
        ) for routine in sourcefile.routines
    )

    assert {module.name.lower() for module in sourcefile.modules} == module_names
    assert {routine.name.lower() for routine in sourcefile.routines} == routine_names
    assert {routine.name.lower() for routine in sourcefile.all_subroutines} == routine_names | module_routine_names

    assert {routine.name.lower() for routine in sourcefile['func_with_contains'].routines} == {'func_with_contains_1'}
    assert {routine.name.lower() for routine in sourcefile['sub_with_contains'].routines} == {
        'sub_with_contains_first', 'sub_with_contains_second', 'sub_with_contains_third'
    }

    program_units_with_imports = {
        'foo': ['bar'], 'func_with_use_and_args': ['foo', 'bar'], 'sub_with_contains': ['bar'],
        'sub_with_renamed_import': ['bar']
    }

    for unit in module_names | routine_names | module_routine_names:
        if unit in program_units_with_imports:
            assert [import_.module.lower() for import_ in sourcefile[unit].imports] == program_units_with_imports[unit]
        else:
            assert not sourcefile[unit].imports
    assert not sourcefile['bar'].typedefs

    # Parse the rest
    sourcefile.make_complete(frontend=REGEX, parser_classes=RegexParserClass.AllClasses)
    assert sourcefile._parser_classes == RegexParserClass.AllClasses
    assert all(
        module._parser_classes == RegexParserClass.AllClasses
        for module in sourcefile.modules
    )
    assert all(
        routine._parser_classes == RegexParserClass.AllClasses
        for routine in sourcefile.routines
    )

    assert {module.name.lower() for module in sourcefile.modules} == module_names
    assert {routine.name.lower() for routine in sourcefile.routines} == routine_names
    assert {routine.name.lower() for routine in sourcefile.all_subroutines} == routine_names | module_routine_names

    assert {routine.name.lower() for routine in sourcefile['func_with_contains'].routines} == {'func_with_contains_1'}
    assert {routine.name.lower() for routine in sourcefile['sub_with_contains'].routines} == {
        'sub_with_contains_first', 'sub_with_contains_second', 'sub_with_contains_third'
    }

    program_units_with_imports = {
        'foo': ['bar'], 'func_with_use_and_args': ['foo', 'bar'], 'sub_with_contains': ['bar'],
        'sub_with_renamed_import': ['bar']
    }

    for unit in module_names | routine_names | module_routine_names:
        if unit in program_units_with_imports:
            assert [import_.module.lower() for import_ in sourcefile[unit].imports] == program_units_with_imports[unit]
        else:
            assert not sourcefile[unit].imports

    # Check access via properties
    assert 'bar' in sourcefile
    assert 'food' in sourcefile['bar']
    assert sorted(sourcefile['bar'].typedef_map) == ['food', 'organic']
    assert sourcefile['bar'].definitions == sourcefile['bar'].typedefs + ('i_am_dim',)
    assert 'cooking_method' in sourcefile['bar']['food']
    assert 'foobar' not in sourcefile['bar']['food']
    assert sourcefile['bar']['food'].interface_symbols == ()

    # Check that triggering a full parse works from nested scopes
    assert sourcefile['bar']._incomplete
    sourcefile['bar']['food'].make_complete()
    assert not sourcefile['bar']._incomplete


def test_regex_raw_source():
    """
    Verify that unparsed source appears in-between matched objects
    """
    fcode = """
! Some comment before the module
!
module some_mod
    ! Some docstring
    ! docstring
    ! docstring
    use some_mod
    ! Some comment
    ! comment
    ! comment
end module some_mod

! Other comment at the end
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)

    assert len(source.ir.body) == 3

    assert isinstance(source.ir.body[0], RawSource)
    assert source.ir.body[0].source.lines == (1, 2)
    assert source.ir.body[0].text == '! Some comment before the module\n!'
    assert source.ir.body[0].source.string == source.ir.body[0].text

    assert isinstance(source.ir.body[1], Module)
    assert source.ir.body[1].source.lines == (3, 11)
    assert source.ir.body[1].source.string.startswith('module')

    assert isinstance(source.ir.body[2], RawSource)
    assert source.ir.body[2].source.lines == (12, 13)
    assert source.ir.body[2].text == '\n! Other comment at the end'
    assert source.ir.body[2].source.string == source.ir.body[2].text

    module = source['some_mod']
    assert len(module.spec.body) == 3
    assert isinstance(module.spec.body[0], RawSource)
    assert isinstance(module.spec.body[1], ir.Import)
    assert isinstance(module.spec.body[2], RawSource)

    assert module.spec.body[0].text.count('docstring') == 3
    assert module.spec.body[2].text.count('comment') == 3


def test_regex_raw_source_with_cpp():
    """
    Verify that unparsed source appears in-between matched objects
    and preprocessor statements are preserved
    """
    fcode = """
! Some comment before the subroutine
#ifdef RS6K
@PROCESS HOT(NOVECTOR) NOSTRICT
#endif
SUBROUTINE SOME_ROUTINE (KLON, KLEV)
IMPLICIT NONE
INTEGER, INTENT(IN) :: KLON, KLEV
! Comment inside routine
END SUBROUTINE SOME_ROUTINE
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)

    assert len(source.ir.body) == 2

    assert isinstance(source.ir.body[0], RawSource)
    assert source.ir.body[0].source.lines == (1, 4)
    assert source.ir.body[0].text.startswith('! Some comment before the subroutine\n#')
    assert source.ir.body[0].text.endswith('#endif')
    assert source.ir.body[0].source.string == source.ir.body[0].text

    assert isinstance(source.ir.body[1], Subroutine)
    assert source.ir.body[1].source.lines == (5, 9)
    assert source.ir.body[1].source.string.startswith('SUBROUTINE')


def test_regex_raw_source_with_cpp_incomplete():
    """
    Verify that unparsed source appears inside matched objects if
    parser classes are used to restrict the matching
    """
    fcode = """
SUBROUTINE driver(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

#include "kernel.intfb.h"

  CALL kernel(a, b ,c)
END SUBROUTINE driver
    """.strip()
    parser_classes = RegexParserClass.ProgramUnitClass
    source = Sourcefile.from_source(fcode, frontend=REGEX, parser_classes=parser_classes)

    assert len(source.ir.body) == 1
    driver = source['driver']
    assert isinstance(driver, Subroutine)
    assert not driver.docstring
    assert not driver.body
    assert not driver.contains
    assert driver.spec and len(driver.spec.body) == 1
    assert isinstance(driver.spec.body[0], RawSource)
    assert 'INTEGER, INTENT' in driver.spec.body[0].text
    assert '#include' in driver.spec.body[0].text


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'Non-standard notation needs full preprocessing')]
))
def test_make_complete_sanitize(frontend):
    """
    Test that attempts to first REGEX-parse and then complete source code
    with unsupported features that require "frontend sanitization".
    """
    fcode = """
! Some comment before the subroutine
#ifdef RS6K
@PROCESS HOT(NOVECTOR) NOSTRICT
#endif
SUBROUTINE SOME_ROUTINE (KLON, KLEV)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: KLON, KLEV
  ! Comment inside routine
END SUBROUTINE SOME_ROUTINE
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)

    # Ensure completion handles the non-supported features (@PROCESS)
    source.make_complete(frontend=frontend)

    comments = FindNodes(ir.Comment).visit(source.ir)
    assert len(comments) == 2 if frontend == FP else 1
    assert comments[0].text == '! Some comment before the subroutine'
    if frontend == FP:
        assert comments[1].text == '@PROCESS HOT(NOVECTOR) NOSTRICT'

    directives = FindNodes(PreprocessorDirective).visit(source.ir)
    assert len(directives) == 2
    assert directives[0].text == '#ifdef RS6K'
    assert directives[1].text == '#endif'


@pytest.mark.skipif(platform.system() == 'Darwin',
    reason='Timeout utility test sporadically fails on MacOS CI runners.'
)
@pytest.mark.usefixtures('reset_regex_frontend_timeout')
def test_regex_timeout():
    """
    This source fails to parse because of missing SUBROUTINE in END
    statement, and the test verifies that a timeout is encountered
    """
    fcode = """
subroutine some_routine(a)
  real, intent(in) :: a
end
    """.strip()

    # Test timeout
    config['regex-frontend-timeout'] = 1
    start = perf_counter()
    with pytest.raises(RuntimeError) as exc:
        _ = Sourcefile.from_source(fcode, frontend=REGEX)
    stop = perf_counter()
    assert .9 < stop - start < 1.1
    assert 'REGEX frontend timeout of 1 s exceeded' in str(exc.value)

    # Test it works fine with proper Fortran:
    fcode += ' subroutine'
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    assert len(source.subroutines) == 1
    assert source.subroutines[0].name == 'some_routine'


def test_regex_module_imports():
    """
    Verify that the regex frontend is able to find and correctly parse
    Fortran imports
    """
    fcode = """
module some_mod
    use no_symbols_mod
    use only_mod, only: my_var
    use test_rename_mod, first_var1 => var1, first_var3 => var3
    use test_other_rename_mod, only: second_var1 => var1
    use test_other_rename_mod, only: other_var2 => var2, other_var3 => var3
    implicit none
end module some_mod
    """.strip()

    module = Module.from_source(fcode, frontend=REGEX)
    imports = FindNodes(ir.Import).visit(module.spec)
    assert len(imports) == 5
    assert [import_.module for import_ in imports] == [
        'no_symbols_mod', 'only_mod', 'test_rename_mod', 'test_other_rename_mod',
        'test_other_rename_mod'
    ]
    assert set(module.imported_symbols) == {
        'my_var', 'first_var1', 'first_var3', 'second_var1', 'other_var2', 'other_var3'
    }
    assert module.imported_symbol_map['first_var1'].type.use_name == 'var1'
    assert module.imported_symbol_map['first_var3'].type.use_name == 'var3'
    assert module.imported_symbol_map['second_var1'].type.use_name == 'var1'
    assert module.imported_symbol_map['other_var2'].type.use_name == 'var2'
    assert module.imported_symbol_map['other_var3'].type.use_name == 'var3'


def test_regex_subroutine_imports():
    """
    Verify that the regex frontend is able to find and correctly parse
    Fortran imports
    """
    fcode = """
subroutine some_routine
    use no_symbols_mod
    use only_mod, only: my_var
    use test_rename_mod, first_var1 => var1, first_var3 => var3
    use test_other_rename_mod, only: second_var1 => var1
    use test_other_rename_mod, only: other_var2 => var2, other_var3 => var3
    implicit none
end subroutine some_routine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=REGEX)
    imports = FindNodes(ir.Import).visit(routine.spec)
    assert len(imports) == 5
    assert [import_.module for import_ in imports] == [
        'no_symbols_mod', 'only_mod', 'test_rename_mod', 'test_other_rename_mod',
        'test_other_rename_mod'
    ]
    assert set(routine.imported_symbols) == {
        'my_var', 'first_var1', 'first_var3', 'second_var1', 'other_var2', 'other_var3'
    }
    assert routine.imported_symbol_map['first_var1'].type.use_name == 'var1'
    assert routine.imported_symbol_map['first_var3'].type.use_name == 'var3'
    assert routine.imported_symbol_map['second_var1'].type.use_name == 'var1'
    assert routine.imported_symbol_map['other_var2'].type.use_name == 'var2'
    assert routine.imported_symbol_map['other_var3'].type.use_name == 'var3'


def test_regex_import_linebreaks():
    """
    Verify correct handling of line breaks in import statements
    """
    fcode = """
module file_io_mod
    USE PARKIND1 , ONLY : JPIM, JPRB, JPRD

#ifdef HAVE_SERIALBOX
    USE m_serialize, ONLY: &
        fs_create_savepoint, &
        fs_add_serializer_metainfo, &
        fs_get_serializer_metainfo, &
        fs_read_field, &
        fs_write_field
    USE utils_ppser, ONLY:  &
        ppser_initialize, &
        ppser_finalize, &
        ppser_serializer, &
        ppser_serializer_ref, &
        ppser_set_mode, &
        ppser_savepoint
#endif

#ifdef HAVE_HDF5
    USE hdf5_file_mod, only: hdf5_file
#endif

    implicit none
end module file_io_mod
    """.strip()
    module = Module.from_source(fcode, frontend=REGEX)
    imports = FindNodes(ir.Import).visit(module.spec)
    assert len(imports) == 4
    assert [import_.module for import_ in imports] == ['PARKIND1', 'm_serialize', 'utils_ppser', 'hdf5_file_mod']
    assert all(
        s in module.imported_symbols for s in [
            'JPIM', 'JPRB', 'JPRD', 'fs_create_savepoint', 'fs_add_serializer_metainfo', 'fs_get_serializer_metainfo',
            'fs_read_field', 'fs_write_field', 'ppser_initialize', 'ppser_finalize', 'ppser_serializer',
            'ppser_serializer_ref', 'ppser_set_mode', 'ppser_savepoint', 'hdf5_file'
        ]
    )


def test_regex_typedef():
    """
    Verify that the regex frontend is able to parse type definitions and
    correctly parse procedure bindings.
    """
    fcode = """
module typebound_item
    implicit none
    type some_type
    contains
        procedure, nopass :: routine => module_routine
        procedure :: some_routine
        procedure, pass :: other_routine
        procedure :: routine1, &
            & routine2 => routine
        ! procedure :: routine1
        ! procedure :: routine2 => routine
    end type some_type
contains
    subroutine module_routine
        integer m
        m = 2
    end subroutine module_routine

    subroutine some_routine(self)
        class(some_type) :: self

        call self%routine
    end subroutine some_routine

    subroutine other_routine(self, m)
        class(some_type), intent(inout) :: self
        integer, intent(in) :: m
        integer :: j

        j = m
        call self%routine1
        call self%routine2
    end subroutine other_routine

    subroutine routine(self)
        class(some_type) :: self
        call self%some_routine
    end subroutine routine

    subroutine routine1(self)
        class(some_type) :: self
        call module_routine
    end subroutine routine1
end module typebound_item
    """.strip()

    module = Module.from_source(fcode, frontend=REGEX)

    assert 'some_type' in module.typedef_map
    some_type = module.typedef_map['some_type']

    proc_bindings = {
        'routine': ('module_routine',),
        'some_routine': None,
        'other_routine': None,
        'routine1': None,
        'routine2': ('routine',)
    }
    assert len(proc_bindings) == len(some_type.variables)
    assert all(proc in some_type.variables for proc in proc_bindings)
    assert all(
        some_type.variable_map[proc].type.bind_names == bind
        for proc, bind in proc_bindings.items()
    )


def test_regex_typedef_generic():
    fcode = """
module typebound_header
    implicit none

    type header_type
    contains
        procedure :: member_routine => header_member_routine
        procedure :: routine_real => header_routine_real
        procedure :: routine_integer
        generic :: routine => routine_real, routine_integer
    end type header_type

contains

    subroutine header_member_routine(self, val)
        class(header_type) :: self
        integer, intent(in) :: val
        integer :: j
        j = val
    end subroutine header_member_routine

    subroutine header_routine_real(self, val)
        class(header_type) :: self
        real, intent(out) :: val
        val = 1.0
    end subroutine header_routine_real

    subroutine routine_integer(self, val)
        class(header_type) :: self
        integer, intent(out) :: val
        val = 1
    end subroutine routine_integer
end module typebound_header
    """.strip()

    module = Module.from_source(fcode, frontend=REGEX)

    assert 'header_type' in module.typedef_map
    header_type = module.typedef_map['header_type']

    proc_bindings = {
        'member_routine': ('header_member_routine',),
        'routine_real': ('header_routine_real',),
        'routine_integer': None,
        'routine': ('routine_real', 'routine_integer')
    }
    assert len(proc_bindings) == len(header_type.variables)
    assert all(proc in header_type.variables for proc in proc_bindings)
    assert all(
        (
            header_type.variable_map[proc].type.bind_names == bind
            and header_type.variable_map[proc].type.initial is None
        )
        for proc, bind in proc_bindings.items()
    )


def test_regex_loki_69():
    """
    Test compliance of REGEX frontend with edge cases reported in LOKI-69.
    This should become a full-blown Scheduler test when REGEX frontend undeprins the scheduler.
    """
    fcode = """
subroutine random_call_0(v_out,v_in,v_inout)
implicit none

    real(kind=jprb),intent(in)  :: v_in
    real(kind=jprb),intent(out)  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout


end subroutine random_call_0

!subroutine random_call_1(v_out,v_in,v_inout)
!implicit none
!
!  real(kind=jprb),intent(in)  :: v_in
!  real(kind=jprb),intent(out)  :: v_out
!  real(kind=jprb),intent(inout)  :: v_inout
!
!
!end subroutine random_call_1

subroutine random_call_2(v_out,v_in,v_inout)
implicit none

    real(kind=jprb),intent(in)  :: v_in
    real(kind=jprb),intent(out)  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout


end subroutine random_call_2

subroutine test(v_out,v_in,v_inout,some_logical)
implicit none

    real(kind=jprb),intent(in   )  :: v_in
    real(kind=jprb),intent(out  )  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout

    logical,intent(in)             :: some_logical

    v_inout = 0._jprb
    if(some_logical)then
        call random_call_0(v_out,v_in,v_inout)
    endif

    if(some_logical) call random_call_2

end subroutine test
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=REGEX)
    assert [r.name for r in source.all_subroutines] == ['random_call_0', 'random_call_2', 'test']

    calls = FindNodes(ir.CallStatement).visit(source['test'].ir)
    assert [call.name for call in calls] == ['RANDOM_CALL_0', 'random_call_2']

    variable_map_test = source['test'].variable_map
    v_in_type = variable_map_test['v_in'].type
    assert v_in_type.dtype is BasicType.REAL
    assert v_in_type.kind == 'jprb'


def test_regex_variable_declaration(testdir):
    """
    Test correct parsing of derived type variable declarations
    """
    filepath = testdir/'sources/projTypeBound/typebound_item.F90'
    source = Sourcefile.from_file(filepath, frontend=REGEX)

    driver = source['driver']
    assert driver.variables == ('constant', 'obj', 'obj2', 'header', 'other_obj', 'derived', 'x', 'i')
    assert source['module_routine'].variables == ('m',)
    assert source['other_routine'].variables == ('self', 'm', 'j')
    assert source['routine'].variables == ('self',)
    assert source['routine1'].variables == ('self',)

    # Check this for REGEX and complete parse to make sure their behaviour is aligned
    for _ in range(2):
        var_map = driver.symbol_map
        assert isinstance(var_map['obj'].type.dtype, DerivedType)
        assert var_map['obj'].type.dtype.name == 'some_type'
        assert isinstance(var_map['obj2'].type.dtype, DerivedType)
        assert var_map['obj2'].type.dtype.name == 'some_type'
        assert isinstance(var_map['header'].type.dtype, DerivedType)
        assert var_map['header'].type.dtype.name == 'header_type'
        assert isinstance(var_map['other_obj'].type.dtype, DerivedType)
        assert var_map['other_obj'].type.dtype.name == 'other'
        assert isinstance(var_map['derived'].type.dtype, DerivedType)
        assert var_map['derived'].type.dtype.name == 'other'
        assert isinstance(var_map['x'].type.dtype, BasicType)
        assert var_map['x'].type.dtype is BasicType.REAL
        assert isinstance(var_map['i'].type.dtype, BasicType)
        assert var_map['i'].type.dtype is BasicType.INTEGER

        # While we're here: let's check the call statements, too
        calls = FindNodes(ir.CallStatement).visit(driver.ir)
        assert len(calls) == 7
        assert all(isinstance(call.name.type.dtype, ProcedureType) for call in calls)

        # Note: we're explicitly accessing the string name here (instead of relying
        # on the StrCompareMixin) as some have dimensions that only show up in the full
        # parse
        assert calls[0].name.name == 'obj%other_routine'
        assert calls[0].name.parent.name == 'obj'
        assert calls[1].name.name == 'obj2%some_routine'
        assert calls[1].name.parent.name == 'obj2'
        assert calls[2].name.name == 'header%member_routine'
        assert calls[2].name.parent.name == 'header'
        assert calls[3].name.name == 'header%routine'
        assert calls[3].name.parent.name == 'header'
        assert calls[4].name.name == 'header%routine'
        assert calls[4].name.parent.name == 'header'
        assert calls[5].name.name == 'other_obj%member'
        assert calls[5].name.parent.name == 'other_obj'
        assert calls[6].name.name == 'derived%var%member_routine'
        assert calls[6].name.parent.name == 'derived%var'
        assert calls[6].name.parent.parent.name == 'derived'

        # Hack: Split the procedure binding into one-per-line until Fparser
        # supports this...
        module = source['typebound_item']
        module.source.string = module.source.string.replace(
            'procedure :: routine1,', 'procedure :: routine1\nprocedure ::'
        )

        source.make_complete()


def test_regex_variable_declaration_parentheses():
    fcode = """
subroutine definitely_not_allfpos(ydfpdata)
implicit none
integer, parameter :: NMaxCloudTypes = 12
type(tfpdata), intent(in) :: ydfpdata
type(tfpofn) :: ylofn(size(ydfpdata%yfpos%yfpgeometry%yfpusergeo))
real, dimension(nproma, max(nang, 1), max(nfre, 1)) :: not_an_annoying_ecwam_var
character(len=511) :: cloud_type_name(NMaxCloudTypes) = ["","","","","","","","","","","",""], other_name = "", names(3) = (/ "", "", "" /)
character(len=511) :: more_names(2) = (/ "What", " is" /), naaaames(2) = [ " going ", "on?" ]
end subroutine definitely_not_allfpos
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=REGEX)
    routine = source['definitely_not_allfpos']
    assert routine.variables == (
        'nmaxcloudtypes', 'ydfpdata', 'ylofn', 'not_an_annoying_ecwam_var',
        'cloud_type_name', 'other_name', 'names', 'more_names', 'naaaames'
    )
    assert routine.symbol_map['not_an_annoying_ecwam_var'].type.dtype is BasicType.REAL
    assert routine.symbol_map['cloud_type_name'].type.dtype is BasicType.CHARACTER


def test_regex_preproc_in_contains():
    fcode = """
module preproc_in_contains
    implicit none
    public  :: routine1, routine2, func
contains
#include "some_include.h"
    subroutine routine1
    end subroutine routine1

    module subroutine mod_routine
        call other_routine
    contains
#define something
    subroutine other_routine
    end subroutine other_routine
    end subroutine mod_routine

    elemental function func
    real func
    end function func
end module preproc_in_contains
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)

    expected_names = {'preproc_in_contains', 'routine1', 'mod_routine', 'func'}
    actual_names = {r.name for r in source.all_subroutines} | {m.name for m in source.modules}
    assert expected_names == actual_names

    assert isinstance(source['mod_routine']['other_routine'], Subroutine)


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_pragma_vs_comment(frontend, tmp_path):
    """
    Make sure pragmas and comments are identified correctly
    """
    fcode = """
module frontend_pragma_vs_comment
    implicit none
!$some pragma
    integer :: var1
!!$some comment
    integer :: var2
!some comment
    integer :: var3
    !$some pragma
    integer :: var4
    ! !$some comment
    integer :: var5
end module frontend_pragma_vs_comment
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    pragmas = FindNodes(ir.Pragma).visit(module.ir)
    comments = FindNodes(ir.Comment).visit(module.ir)
    assert len(pragmas) == 2
    assert len(comments) == 3
    assert all(pragma.keyword == 'some' for pragma in pragmas)
    assert all(pragma.content == 'pragma' for pragma in pragmas)
    assert all('some comment' in comment.text for comment in comments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_main_program(frontend):
    """
    Loki can't handle PROGRAM blocks and the frontends should throw an exception
    """
    fcode = """
program hello
    print *, "Hello World!"
end program
    """.strip()

    with config_override({'frontend-strict-mode': True}):
        with pytest.raises(NotImplementedError):
            Sourcefile.from_source(fcode, frontend=frontend)

    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert source.ir.body == ()


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_source_lineno(frontend):
    """
    ...
    """
    fcode = """
    subroutine driver
        call kernel()
        call kernel()
        call kernel()
    end subroutine driver
    """

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['driver']
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert calls[0] != calls[1]
    assert calls[1] != calls[2]
    assert calls[0].source.lines[0] < calls[1].source.lines[0] < calls[2].source.lines[0]


def test_regex_interface_subroutine():
    fcode = """
subroutine test(callback)

implicit none
interface
    subroutine some_kernel(a, b, c)
    integer, intent(in) :: a, b
    integer, intent(out) :: c
    end subroutine some_kernel

    SUBROUTINE other_kernel(a)
    integer, intent(inout) :: a
    end subroutine
end interface

INTERFACE
    function other_func(a)
    integer, intent(in) :: a
    integer, other_func
    end function other_func
end interface

abstract interface
    function callback_func(a) result(b)
        integer, intent(in) :: a
        integer :: b
    end FUNCTION callback_func
end INTERFACE

procedure(callback_func), pointer, intent(in) :: callback
integer :: a, b, c

a = callback(1)
b = other_func(a)

call some_kernel(a, b, c)
call other_kernel(c)

end subroutine test
    """.strip()

    # Make sure only the host subroutine is captured
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    assert len(source.subroutines) == 1
    assert source.subroutines[0].name == 'test'
    assert source.subroutines[0].source.lines == (1, 38)

    # Make sure this also works for module procedures
    fcode = f"""
module my_mod
    implicit none
contains
{fcode}
end module my_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    assert not source.subroutines
    assert len(source.all_subroutines) == 1
    assert source.all_subroutines[0].name == 'test'
    assert source.all_subroutines[0].source.lines == (4, 41)


def test_regex_interface_module():
    fcode = """
module my_mod
    implicit none
    interface
        subroutine ext1 (x, y, z)
            real, dimension(100, 100), intent(inout) :: x, y, z
        end subroutine ext1
        subroutine ext2 (x, z)
            real, intent(in) :: x
            complex(kind = 4), intent(inout) :: z(2000)
        end subroutine ext2
        function ext3 (p, q)
            logical ext3
            integer, intent(in) :: p(1000)
            logical, intent(in) :: q(1000)
        end function ext3
    end interface
    interface sub
        subroutine sub_int (a)
            integer, intent(in) :: a(:)
        end subroutine sub_int
        subroutine sub_real (a)
            real, intent(in) :: a(:)
        end subroutine sub_real
    end interface sub
    interface func
        module procedure func_int
        module procedure func_real
    end interface func
contains
    subroutine sub_int (a)
        integer, intent(in) :: a(:)
    end subroutine sub_int
    subroutine sub_real (a)
        real, intent(in) :: a(:)
    end subroutine sub_real
    integer module function func_int (a)
        integer, intent(in) :: a(:)
    end function func_int
    real module function func_real (a)
        real, intent(in) :: a(:)
    end function func_real
end module my_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)

    assert len(source.modules) == 1
    assert source['my_mod'] is not None
    assert not source['my_mod'].interfaces

    source.make_complete(
        frontend=REGEX,
        parser_class=RegexParserClass.ProgramUnitClass | RegexParserClass.InterfaceClass
    )
    assert len(source['my_mod'].interfaces) == 3
    assert source['my_mod'].symbols == (
        'ext1', 'ext2', 'ext3',
        'sub', 'sub_int', 'sub_real',
        'func', 'func_int', 'func_real', 'func_int', 'func_real',
        'sub_int', 'sub_real',
        'func_int', 'func_real'
    )


def test_regex_function_inline_return_type():
    fcode = """
REAL(KIND=JPRB)  FUNCTION  DOT_PRODUCT_ECV()

END FUNCTION DOT_PRODUCT_ECV

SUBROUTINE DOT_PROD_SP_2D()

END SUBROUTINE DOT_PROD_SP_2D
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    assert {
        routine.name.lower() for routine in source.subroutines
    } == {'dot_product_ecv', 'dot_prod_sp_2d'}

    source.make_complete()
    routine = source['dot_product_ecv']
    assert 'dot_product_ecv' in routine.variables


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP, 'No support for prefix implemented')]))
def test_regex_prefix(frontend, tmp_path):
    fcode = """
module some_mod
    implicit none
contains
    pure elemental real function f_elem(a)
        real, intent(in) :: a
        f_elem = a
    end function f_elem

    pure recursive integer function fib(i) result(fib_i)
        integer, intent(in) :: i
        if (i <= 0) then
            fib_i = 0
        else if (i == 1) then
            fib_i = 1
        else
            fib_i = fib(i-1) + fib(i-2)
        end if
    end function fib
end module some_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    assert source['f_elem'].prefix == ('pure elemental real',)
    assert source['fib'].prefix == ('pure recursive integer',)
    source.make_complete(frontend=frontend, xmods=[tmp_path])
    assert tuple(p.lower() for p in source['f_elem'].prefix) == ('pure', 'elemental')
    assert tuple(p.lower() for p in source['fib'].prefix) == ('pure', 'recursive')


def test_regex_fypp():
    """
    Test that unexpanded fypp-annotations are handled gracefully in the REGEX frontend.
    """
    fcode = """
module fypp_mod
! A pre-set array of pre-prcessor variables
#:mute
#:set foo  = [2,3,4,5]
#:endmute

contains

! A non-templated routine
subroutine first_routine(i, x)
  integer, intent(in) :: i
  real, intent(inout) :: x(3)
end subroutine first_routine

! A fypp-loop with in-place directives for subroutine names
#:for bar in foo
#:set rname = 'routine_%s' % (bar,)
subroutine ${rname}$ (i, x)
  integer, intent(in) :: i
  real, intent(inout) :: x(3)
end subroutine ${rname}$
#:endfor

! Another non-templated routine
subroutine last_routine(i, x)
  integer, intent(in) :: i
  real, intent(inout) :: x(3)
end subroutine last_routine

end module fypp_mod
"""
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    module = source['fypp_mod']
    assert isinstance(module, Module)

    # Check that only non-templated routines are included
    assert len(module.routines) == 2
    assert module.routines[0].name == 'first_routine'
    assert module.routines[1].name == 'last_routine'


@pytest.mark.parametrize(
    'frontend',
    available_frontends(include_regex=True, xfail=[(OMNI, 'OMNI may segfault on empty files')])
)
@pytest.mark.parametrize('fcode', ['', '\n', '\n\n\n\n'])
def test_frontend_empty_file(frontend, fcode):
    """Ensure that all frontends can handle empty source files correctly (#186)"""
    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert isinstance(source.ir, ir.Section)
    assert not source.to_fortran().strip()


@pytest.mark.parametrize('frontend', available_frontends())
def test_pragma_line_continuation(frontend):
    """
    Test that multi-line pragmas are parsed and dealt with correctly.
    """
    fcode = """
SUBROUTINE TOTO(A,B)

IMPLICIT NONE
REAL, INTENT(IN) :: A
REAL, INTENT(INOUT) :: B

!$ACC PARALLEL LOOP GANG &
!$ACC& PRESENT(ZRDG_LCVQ,ZFLU_QSATS,ZRDG_CVGQ) &
!$ACC& PRIVATE (JBLK) &
!$ACC& VECTOR_LENGTH (YDCPG_OPTS%KLON)
!$ACC SEQUENTIAL

END SUBROUTINE TOTO
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    pragmas = FindNodes(ir.Pragma).visit(routine.body)

    assert len(pragmas) == 2
    assert pragmas[0].keyword == 'ACC'
    assert 'PARALLEL' in pragmas[0].content
    assert 'PRESENT' in pragmas[0].content
    assert 'PRIVATE' in pragmas[0].content
    assert 'VECTOR_LENGTH' in pragmas[0].content
    assert pragmas[1].content == 'SEQUENTIAL'

    # Check that source object was generated right
    assert pragmas[0].source
    assert pragmas[0].source.lines == (8, 8) if frontend == OMNI else (8, 11)
    assert pragmas[1].source
    assert pragmas[1].source.lines == (12, 12)


@pytest.mark.parametrize('frontend', available_frontends())
def test_comment_block_clustering(frontend):
    """
    Test that multiple :any:`Comment` nodes into a :any:`CommentBlock`.
    """
    fcode = """
subroutine test_comment_block(a, b)
  ! What is this?
  ! Ohhh, ... a docstring?
  real, intent(inout) :: a, b

  a = a + 1.0
  ! Never gonna
  b = b + 2
  ! give you
  ! up...

  a = a + b
  ! Shut up, ...
  ! Rick!
end subroutine test_comment_block
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    comments = FindNodes(ir.Comment).visit(routine.spec)
    assert len(comments) == 0
    blocks = FindNodes(ir.CommentBlock).visit(routine.spec)
    assert len(blocks) == 0

    assert isinstance(routine.docstring[0], ir.CommentBlock)
    assert len(routine.docstring[0].comments) == 2
    assert routine.docstring[0].comments[0].text == '! What is this?'
    assert routine.docstring[0].comments[1].text == '! Ohhh, ... a docstring?'

    comments = FindNodes(ir.Comment).visit(routine.body)
    assert len(comments) == 2 if frontend == FP else 1
    assert comments[-1].text == '! Never gonna'

    blocks = FindNodes(ir.CommentBlock).visit(routine.body)
    assert len(blocks) == 2
    assert len(blocks[0].comments) == 3 if frontend == FP else 2
    assert blocks[0].comments[0].text == '! give you'
    assert blocks[0].comments[1].text == '! up...'

    assert len(blocks[1].comments) == 2
    assert blocks[1].comments[0].text == '! Shut up, ...'
    assert blocks[1].comments[1].text == '! Rick!'


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI strips comments during parse')]
))
def test_inline_comments(frontend):
    """
    Test that multiple :any:`Comment` nodes into a :any:`CommentBlock`.
    """
    fcode = """
subroutine test_inline_comments(a, b)
  real, intent(inout) :: a, b  ! We don't need no education
  real, external :: alien_func ! We don't need no thought control
  integer :: i

  a = a + 1.0
  ! Who said that?
  b = b + 2             ! All in all it's just another

  do i=1, 10
    b = b + 2           ! Brick in the ...
  enddo

  a = a + alien_func()  ! wall !
end subroutine test_inline_comments
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    assert len(decls) == 2
    assert decls[0].comment.text == "! We don't need no education"
    assert decls[1].comment is None

    proc_decls = FindNodes(ir.ProcedureDeclaration).visit(routine.spec)
    assert len(proc_decls) == 1
    assert proc_decls[0].comment.text == "! We don't need no thought control"

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 4
    assert assigns[0].comment is None
    assert assigns[1].comment.text == "! All in all it's just another"
    assert assigns[2].comment.text == '! Brick in the ...'
    assert assigns[3].comment.text == '! wall !'

    comments = FindNodes(ir.Comment).visit(routine.body)
    assert len(comments) == 1 if frontend == OFP else 4
    if frontend == OFP:
        assert comments[0].text == '! Who said that?'
    else:
        assert comments[1].text == '! Who said that?'
        assert comments[0].text == comments[2].text == comments[3].text == ''


@pytest.mark.parametrize('from_file', (True, False))
@pytest.mark.parametrize('preprocess', (True, False))
def test_source_sanitize_fp_source(tmp_path, from_file, preprocess):
    """
    Test that source sanitizing works as expected and postprocessing
    rules are correctly applied
    """
    fcode = """
subroutine some_routine(input_path)
    implicit none
    character(len=255), intent(in) :: input_path
    integer :: ios, fu
    write(*,*) "we print CPP value ", MY_VAR
    ! In the following line the PP definition should be replace by '0'
    ! or the actual line number
    write(*,*) "We are in line ",__LINE__
    open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
end subroutine some_routine
""".strip()

    if from_file:
        filepath = tmp_path/'some_routine.F90'
        filepath.write_text(fcode)
        obj = Sourcefile.from_file(filepath, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))
    else:
        obj = Sourcefile.from_source(fcode, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))

    if preprocess:
        # CPP takes care of that
        assert '"We are in line ", 8' in obj.to_fortran()
        assert '"we print CPP value ", 5' in obj.to_fortran()
    else:
        # source sanitisation takes care of that
        assert '"We are in line ", 0' in obj.to_fortran()
        assert '"we print CPP value ", MY_VAR' in obj.to_fortran()

    assert 'newunit=fu' in obj.to_fortran()


@pytest.mark.parametrize('preprocess', (True, False))
def test_source_sanitize_fp_subroutine(preprocess):
    """
    Test that source sanitizing works as expected and postprocessing
    rules are correctly applied
    """
    fcode = """
subroutine some_routine(input_path)
    implicit none
    character(len=255), intent(in) :: input_path
    integer :: ios, fu
    write(*,*) "we print CPP value ", MY_VAR
    ! In the following line the PP definition should be replace by '0'
    ! or the actual line number
    write(*,*) "We are in line ",__LINE__
    open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
end subroutine some_routine
""".strip()

    obj = Subroutine.from_source(fcode, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))

    if preprocess:
        # CPP takes care of that
        assert '"We are in line ", 8' in obj.to_fortran()
        assert '"we print CPP value ", 5' in obj.to_fortran()
    else:
        # source sanitisation takes care of that
        assert '"We are in line ", 0' in obj.to_fortran()
        assert '"we print CPP value ", MY_VAR' in obj.to_fortran()

    assert 'newunit=fu' in obj.to_fortran()


@pytest.mark.parametrize('preprocess', (True, False))
def test_source_sanitize_fp_module(preprocess):
    """
    Test that source sanitizing works as expected and postprocessing
    rules are correctly applied
    """
    fcode = """
module some_mod
    implicit none
    integer line = __LINE__ + MY_VAR
contains
subroutine some_routine(input_path)
    implicit none
    character(len=255), intent(in) :: input_path
    integer :: ios, fu
    write(*,*) "we print CPP value ", MY_VAR
    ! In the following line the PP definition should be replace by '0'
    ! or the actual line number
    write(*,*) "We are in line ",__LINE__
    open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
end subroutine some_routine
end module some_mod
""".strip()

    obj = Module.from_source(fcode, frontend=FP, preprocess=preprocess, defines=('MY_VAR=5',))

    if preprocess:
        # CPP takes care of that
        assert 'line = 3 + 5' in obj.to_fortran()
        assert '"We are in line ", 12' in obj.to_fortran()
        assert '"we print CPP value ", 5' in obj.to_fortran()
    else:
        # source sanitisation takes care of that
        assert 'line = 0 + MY_VAR' in obj.to_fortran()
        assert '"We are in line ", 0' in obj.to_fortran()
        assert '"we print CPP value ", MY_VAR' in obj.to_fortran()

    assert 'newunit=fu' in obj.to_fortran()


# TODO: Add tests for source sanitizer with other frontends


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI does not like Loki pragmas, yet!')]))
def test_frontend_routine_variables_dimension_pragmas(frontend):
    """
    Test that `!$loki dimension` pragmas can be used to verride the
    conceptual `.shape` of local and argument variables.
    """
    fcode = """
subroutine routine_variables_dimensions(x, y, v0, v1, v2, v3, v4)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y

  !$loki dimension(10)
  real(kind=jprb), intent(inout) :: v0(:)
  !$loki dimension(x)
  real(kind=jprb), intent(inout) :: v1(:)
  !$loki dimension(x,y,:)
  real(kind=jprb), dimension(:,:,:), intent(inout) :: v2, v3
  !$loki dimension(x,y)
  real(kind=jprb), pointer, intent(inout) :: v4(:,:)
  !$loki dimension(x+y,2*x)
  real(kind=jprb), allocatable :: v5(:,:)
  !$loki dimension(x/2, x**2, (x+y)/x)
  real(kind=jprb), dimension(:, :, :), pointer :: v6

end subroutine routine_variables_dimensions
"""
    def to_str(expr):
        return str(expr).lower().replace(' ', '')

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert routine.variable_map['v0'].shape[0] == 10
    assert isinstance(routine.variable_map['v0'].shape[0], sym.IntLiteral)
    assert isinstance(routine.variable_map['v1'].shape[0], sym.Scalar)
    assert routine.variable_map['v2'].shape[0] == 'x'
    assert routine.variable_map['v2'].shape[1] == 'y'
    assert routine.variable_map['v2'].shape[2] == ':'
    assert isinstance(routine.variable_map['v2'].shape[0], sym.Scalar)
    assert isinstance(routine.variable_map['v2'].shape[1], sym.Scalar)
    assert isinstance(routine.variable_map['v2'].shape[2], sym.RangeIndex)
    assert routine.variable_map['v3'].shape[0] == 'x'
    assert routine.variable_map['v3'].shape[1] == 'y'
    assert routine.variable_map['v3'].shape[2] == ':'
    assert isinstance(routine.variable_map['v3'].shape[0], sym.Scalar)
    assert isinstance(routine.variable_map['v3'].shape[1], sym.Scalar)
    assert isinstance(routine.variable_map['v3'].shape[2], sym.RangeIndex)
    assert routine.variable_map['v4'].shape[0] == 'x'
    assert routine.variable_map['v4'].shape[1] == 'y'
    assert isinstance(routine.variable_map['v4'].shape[0], sym.Scalar)
    assert isinstance(routine.variable_map['v4'].shape[1], sym.Scalar)
    assert to_str(routine.variable_map['v5'].shape[0]) == 'x+y'
    assert to_str(routine.variable_map['v5'].shape[1]) == '2*x'
    assert isinstance(routine.variable_map['v5'].shape[0], sym.Sum)
    assert isinstance(routine.variable_map['v5'].shape[1], sym.Product)
    assert to_str(routine.variable_map['v6'].shape[0]) == 'x/2'
    assert to_str(routine.variable_map['v6'].shape[1]) == 'x**2'
    assert to_str(routine.variable_map['v6'].shape[2]) == '(x+y)/x'
    assert isinstance(routine.variable_map['v6'].shape[0], sym.Quotient)
    assert isinstance(routine.variable_map['v6'].shape[1], sym.Power)
    assert isinstance(routine.variable_map['v6'].shape[2], sym.Quotient)

@pytest.mark.parametrize('frontend', available_frontends())
def test_import_of_private_symbols(tmp_path, frontend):
    """
    Verify that only public symbols are imported from other modules.
    """
    code_mod_private = """
module mod_private
    private
    integer :: var
end module mod_private
    """
    code_mod_public = """
module mod_public
    public
    integer:: var
end module mod_public
    """
    code_mod_main = """
module mod_main
    use mod_public
    use mod_private
contains

    subroutine test_routine()
        integer :: result
        result = var
    end subroutine test_routine

end module mod_main
    """

    mod_private = Module.from_source(code_mod_private, frontend=frontend, xmods=[tmp_path])
    mod_public = Module.from_source(code_mod_public, frontend=frontend, xmods=[tmp_path])
    mod_main = Module.from_source(
        code_mod_main, frontend=frontend, definitions=[mod_private, mod_public], xmods=[tmp_path]
    )
    var = mod_main.subroutines[0].body.body[0].rhs
    # Check if this is really our symbol
    assert var.name == "var"
    assert var.scope is mod_main
    # Check if the symbol is imported
    assert var.type.imported is True
    # Check if the symbol comes from the mod_public module
    assert var.type.module is mod_public
