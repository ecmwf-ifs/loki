"""
Verify correct frontend behaviour and correct parsing of certain Fortran
language features.
"""
from pathlib import Path
import numpy as np
import pytest

from conftest import jit_compile, clean_test, available_frontends
from loki import (
    Module, Subroutine, FindNodes, FindVariables, Allocation, Deallocation, Associate,
    BasicType, OMNI, OFP, Enumeration, config, REGEX, Sourcefile
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
def test_frontend_strict_mode(frontend, reset_frontend_mode):  # pylint: disable=unused-argument
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
    assert routine.arguments == ('i', 'j')
    assert routine.argnames == ['i', 'j']
    assert [r.name for r in routine.subroutines] == ['contained_c', 'contained_e', 'contained_d']

    contained_c = routine['contained_c']
    assert contained_c.name == 'contained_c'
    assert not contained_c.is_function
    assert contained_c.arguments == ('i',)
    assert contained_c.argnames == ['i']

    contained_e = routine['contained_e']
    assert contained_e.name == 'contained_e'
    assert contained_e.is_function
    assert contained_e.arguments == ('i',)
    assert contained_e.argnames == ['i']

    contained_d = routine['contained_d']
    assert contained_d.name == 'contained_d'
    assert not contained_d.is_function
    assert contained_d.arguments == ('i',)
    assert contained_d.argnames == ['i']

    code = routine.to_fortran()
    assert code.count('SUBROUTINE') == 6
    assert code.count('FUNCTION') == 2
    assert code.count('contains') == 1


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
    assert code.count('contains') == 1


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
    assert code.count('contains') == 2
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
    assert function_d.arguments == ('d',)
    assert function_d.argnames == ['d']
    assert not function_d.contains

    code = sourcefile.to_fortran()
    assert code.count('SUBROUTINE') == 8
    assert code.count('FUNCTION') == 4
    assert code.count('contains') == 2
    assert code.count('MODULE') == 2
