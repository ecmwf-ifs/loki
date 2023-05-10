"""
Verify correct frontend behaviour and correct parsing of certain Fortran
language features.
"""
from pathlib import Path
from time import perf_counter
import numpy as np
import pytest

from conftest import jit_compile, clean_test, available_frontends
from loki import (
    Module, Subroutine, FindNodes, FindVariables, Allocation,
    Deallocation, Associate, BasicType, OMNI, OFP, FP, Enumeration,
    config, REGEX, Sourcefile, Import, RawSource, CallStatement,
    RegexParserClass, ProcedureType, DerivedType, Comment, Pragma,
    PreprocessorDirective, config_override
)
from loki.expression import symbols as sym


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


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
def test_check_alloc_opts(here, frontend):
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
    module = Module.from_source(fcode, frontend=frontend)

    allocations = FindNodes(Allocation).visit(module['check_alloc_source'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is not None for alloc in allocations)
    assert all(alloc.status_var is None for alloc in allocations)

    allocations = FindNodes(Allocation).visit(module['alloc_deferred'].body)
    assert len(allocations) == 2
    assert all(alloc.data_source is None for alloc in allocations)
    assert allocations[0].status_var is not None
    assert allocations[1].status_var is None

    deallocs = FindNodes(Deallocation).visit(module['free_deferred'].body)
    assert len(deallocs) == 2
    assert deallocs[0].status_var is not None
    assert deallocs[1].status_var is None

    # Sanity check for the backend
    assert module.to_fortran().lower().count(', stat=stat') == 2

    # Generate Fortran and test it
    filepath = here/(f'frontends_check_alloc_{frontend}.f90')
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
def test_associates(here, frontend):
    """
    Test the use of associate to access and modify other items
    """

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
    module = Module.from_source(fcode, frontend=frontend)
    routine = module['associates']
    variables = FindVariables().visit(routine.body)
    if frontend == OMNI:
        assert all(v.shape == ('1:3',)
                   for v in variables if v.name in ['vector', 'vector2'])
    else:
        assert all(v.shape == ('3',)
                   for v in variables if v.name in ['vector', 'vector2'])

    for assoc in FindNodes(Associate).visit(routine.body):
        for var in FindVariables().visit(assoc.body):
            if var.name in assoc.variables:
                assert var.scope is assoc
                assert var.type.parent is None
            else:
                assert var.scope is routine

    # Test the generated module
    filepath = here/(f'derived_types_associates_{frontend}.f90')
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
ASSOCIATE(SOME_VAR=>ITEM%SOME_VAR(IDX))
SOME_VAR = 5
END ASSOCIATE
END SUBROUTINE
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(FindVariables(recurse_to_parent=False).visit(routine.body)) == 3
    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 4
    some_var = variables['SOME_VAR']
    assert isinstance(some_var, sym.DeferredTypeSymbol)
    assert some_var.name.upper() == 'SOME_VAR'
    assert some_var.type.dtype == BasicType.DEFERRED
    assert some_var.scope is FindNodes(Associate).visit(routine.body)[0]


@pytest.mark.parametrize('frontend', available_frontends())
def test_associates_expr(here, frontend):
    """
    Verify that associates with expressions are supported
    """
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

    filepath = here/(f'associates_expr_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    a = np.array([1, 2, 3], dtype='i')
    b = np.zeros(3, dtype='i')
    function(a, b)
    assert np.all(b == [7, 10, 13])
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_enum(here, frontend):
    """
    Verify that enums are represented correctly
    """
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
    enums = FindNodes(Enumeration).visit(routine.spec)
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
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    out = function()
    assert out == 23
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OFP, 'OFP fails to parse parameterized types')]
))
@pytest.mark.usefixtures('reset_frontend_mode')
def test_frontend_strict_mode(frontend):
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
        _ = Module.from_source(fcode, frontend=frontend)

    config['frontend-strict-mode'] = False
    module = Module.from_source(fcode, frontend=frontend)
    assert 'matrix' in module.symbol_attrs
    assert 'matrix' in module.typedefs


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
    implicit none
    integer, intent(in) :: i, j
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
end subroutine routine_b
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=REGEX)
    assert routine.name == 'routine_b'
    assert not routine.is_function
    assert routine.arguments == ()
    assert routine.argnames == []
    assert [r.name for r in routine.subroutines] == ['contained_c', 'contained_e', 'contained_d']

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
    implicit none
    use foobar
contains
    subroutine module_routine
        integer m
        m = 2

        call routine_b(m, 6)
    end subroutine module_routine

    function module_function(n)
        integer n
        n = 3
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
    end subroutine module_routine

    function module_function(n)
        integer n
        n = 3
    end function module_function
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
end subroutine routine_b

function function_d(d)
    integer d
    d = 6
end function function_d
    """.strip()

    sourcefile = Sourcefile.from_source(fcode, frontend=REGEX)
    assert [m.name for m in sourcefile.modules] == ['some_module', 'other_module']
    assert [r.name for r in sourcefile.routines] == [
        'routine_a', 'routine_b', 'function_d'
    ]
    assert [r.name for r in sourcefile.all_subroutines] == [
        'routine_a', 'routine_b', 'function_d', 'module_routine', 'module_function'
    ]

    code = sourcefile.to_fortran()
    assert code.count('SUBROUTINE') == 10
    assert code.count('FUNCTION') == 6
    assert code.count('CONTAINS') == 2
    assert code.count('MODULE') == 4


def test_regex_sourcefile_from_file(here):
    """
    Verify that the regex frontend is able to parse source files containing
    multiple modules and subroutines
    """

    sourcefile = Sourcefile.from_file(here/'sources/sourcefile.f90', frontend=REGEX)
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


def test_regex_sourcefile_from_file_parser_classes(here):

    filepath = here/'sources/Fortran-extract-interface-source.f90'
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

    # Incremental addition of program unit objects
    sourcefile.make_complete(frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)

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

    # Incremental addition of imports
    sourcefile.make_complete(
        frontend=REGEX,
        parser_classes=RegexParserClass.ProgramUnitClass | RegexParserClass.ImportClass
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

    assert sorted(sourcefile['bar'].typedefs) == ['food', 'organic']


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
    assert isinstance(module.spec.body[1], Import)
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

    comments = FindNodes(Comment).visit(source.ir)
    assert len(comments) == 2 if frontend == FP else 1
    assert comments[0].text == '! Some comment before the subroutine'
    if frontend == FP:
        assert comments[1].text == '@PROCESS HOT(NOVECTOR) NOSTRICT'

    directives = FindNodes(PreprocessorDirective).visit(source.ir)
    assert len(directives) == 2
    assert directives[0].text == '#ifdef RS6K'
    assert directives[1].text == '#endif'


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
    imports = FindNodes(Import).visit(module.spec)
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
    imports = FindNodes(Import).visit(routine.spec)
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
    imports = FindNodes(Import).visit(module.spec)
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

    assert 'some_type' in module.typedefs
    some_type = module.typedefs['some_type']

    proc_bindings = {
        'routine': 'module_routine',
        'some_routine': None,
        'other_routine': None,
        'routine1': None,
        'routine2': 'routine'
    }
    assert len(proc_bindings) == len(some_type.variables)
    assert all(proc in some_type.variables for proc in proc_bindings)
    assert all(some_type.variable_map[proc].type.initial == init for proc, init in proc_bindings.items())


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

    assert 'header_type' in module.typedefs
    header_type = module.typedefs['header_type']

    proc_bindings = {
        'member_routine': 'header_member_routine',
        'routine_real': 'header_routine_real',
        'routine_integer': None,
        'routine': ('routine_real', 'routine_integer')
    }
    assert len(proc_bindings) == len(header_type.variables)
    assert all(proc in header_type.variables for proc in proc_bindings)
    assert all(
        (
            header_type.variable_map[proc].type.bind_names == bind
            and header_type.variable_map[proc].type.initial is None
        ) if isinstance(bind, tuple) else (
            header_type.variable_map[proc].type.bind_names is None
            and header_type.variable_map[proc].type.initial == bind
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

    calls = FindNodes(CallStatement).visit(source['test'].ir)
    assert [call.name for call in calls] == ['RANDOM_CALL_0', 'random_call_2']


def test_regex_variable_declaration(here):
    """
    Test correct parsing of derived type variable declarations
    """
    filepath = here/'sources/projTypeBound/typebound_item.F90'
    source = Sourcefile.from_file(filepath, frontend=REGEX)

    driver = source['driver']
    assert driver.variables == ('obj', 'obj2', 'header', 'other_obj', 'derived', 'x', 'i')
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
        calls = FindNodes(CallStatement).visit(driver.ir)
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
type(tfpdata), intent(in) :: ydfpdata
type(tfpofn) :: ylofn(size(ydfpdata%yfpos%yfpgeometry%yfpusergeo))
real, dimension(nproma, max(nang, 1), max(nfre, 1)) :: not_an_annoying_ecwam_var
end subroutine definitely_not_allfpos
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=REGEX)
    routine = source['definitely_not_allfpos']
    assert routine.variables == ('ydfpdata', 'ylofn', 'not_an_annoying_ecwam_var')
    assert routine.symbol_map['not_an_annoying_ecwam_var'].type.dtype is BasicType.REAL


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
def test_frontend_pragma_vs_comment(frontend):
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

    module = Module.from_source(fcode, frontend=frontend)
    pragmas = FindNodes(Pragma).visit(module.ir)
    comments = FindNodes(Comment).visit(module.ir)
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
    calls = FindNodes(CallStatement).visit(routine.body)
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

    source.make_complete(frontend=REGEX, parser_class=RegexParserClass.ProgramUnitClass | RegexParserClass.InterfaceClass)
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


@pytest.mark.parametrize('frontend', available_frontends(xfail=(OFP, 'No support for prefix implemented')))
def test_regex_prefix(frontend):
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
    source.make_complete(frontend=frontend)
    assert tuple(p.lower() for p in source['f_elem'].prefix) == ('pure', 'elemental')
    assert tuple(p.lower() for p in source['fib'].prefix) == ('pure', 'recursive')
