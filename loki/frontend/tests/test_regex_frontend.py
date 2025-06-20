# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct parsing behaviour of the REGEX frontend
"""

from pathlib import Path

import platform
from time import perf_counter
import pytest

from loki import Function, Module, Subroutine, Sourcefile, RawSource, config
from loki.frontend import (
    available_frontends, OMNI, FP, REGEX, RegexParserClass
)
from loki.ir import nodes as ir, FindNodes, PreprocessorDirective
from loki.types import BasicType, ProcedureType, DerivedType


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


@pytest.fixture(name='reset_regex_frontend_timeout')
def fixture_reset_regex_frontend_timeout():
    original_timeout = config['regex-frontend-timeout']
    yield
    config['regex-frontend-timeout'] = original_timeout


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
    assert isinstance(source['dot_product_ecv'], Function)
    assert isinstance(source['dot_prod_sp_2d'], Subroutine)

    source.make_complete()
    function = source['dot_product_ecv']
    assert function.return_type.dtype == BasicType.REAL
    assert function.return_type.kind == 'JPRB'


@pytest.mark.parametrize('frontend', available_frontends())
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


def test_declaration_whitespace_attributes():
    """
    Test correct behaviour with/without white space inside declaration attributes
    (reported in #318).
    """
    fcode = """
subroutine my_whitespace_declaration_routine(kdim, state_t0, paux)
    use type_header, only: dimension_type, STATE_TYPE, aux_type, jprb
    implicit none
    TYPE( DIMENSION_TYPE) , INTENT (IN) :: KDIM
    type (state_type  ) , intent ( in ) :: state_t0
    TYPE (AUX_TYPE) , InteNT( In) :: PAUX
    CHARACTER  ( LEN=10) :: STR
    REAL(  KIND = JPRB  ) :: VAR
end subroutine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=REGEX)

    # Verify that variables and dtype information has been extracted correctly
    assert routine.variables == ('kdim', 'state_t0', 'paux', 'str', 'var')
    assert isinstance(routine.variable_map['kdim'].type.dtype, DerivedType)
    assert routine.variable_map['kdim'].type.dtype.name.lower() == 'dimension_type'
    assert isinstance(routine.variable_map['state_t0'].type.dtype, DerivedType)
    assert routine.variable_map['state_t0'].type.dtype.name.lower() == 'state_type'
    assert isinstance(routine.variable_map['paux'].type.dtype, DerivedType)
    assert routine.variable_map['paux'].type.dtype.name.lower() == 'aux_type'
    assert routine.variable_map['str'].type.dtype == BasicType.CHARACTER
    assert routine.variable_map['var'].type.dtype == BasicType.REAL

    routine.make_complete()

    # Verify that additional type attributes are correct after full parse
    assert routine.variables == ('kdim', 'state_t0', 'paux', 'str', 'var')
    assert isinstance(routine.variable_map['kdim'].type.dtype, DerivedType)
    assert routine.variable_map['kdim'].type.dtype.name.lower() == 'dimension_type'
    assert routine.variable_map['kdim'].type.intent == 'in'
    assert isinstance(routine.variable_map['state_t0'].type.dtype, DerivedType)
    assert routine.variable_map['state_t0'].type.dtype.name.lower() == 'state_type'
    assert routine.variable_map['state_t0'].type.intent == 'in'
    assert isinstance(routine.variable_map['paux'].type.dtype, DerivedType)
    assert routine.variable_map['paux'].type.dtype.name.lower() == 'aux_type'
    assert routine.variable_map['paux'].type.intent == 'in'
    assert routine.variable_map['str'].type.dtype == BasicType.CHARACTER
    assert routine.variable_map['str'].type.length == 10
    assert routine.variable_map['var'].type.dtype == BasicType.REAL
    assert routine.variable_map['var'].type.kind == 'jprb'


def test_regex_sanitize_fypp_line_annotations():
    """
    Test that fypp line number annotations are sanitized correctly.
    """

    fcode = """
module some_templated_mod

# 1 "/path-to-hypp-macro/macro.hypp" 1
# 2 "/path-to-hypp-macro/macro.hypp"
# 3 "/path-to-hypp-macro/macro.hypp"
# 5 "/path-to-fypp-template/template.fypp" 2

integer :: a0
integer :: a1
integer :: a2
integer :: a3
integer :: a4

end module some_templated_mod
"""

    module = Module.from_source(fcode, frontend=REGEX)
    decls = FindNodes(ir.VariableDeclaration).visit(module.spec)

    assert len(decls) == 5

def test_regex_pragma():
    """
    Make sure the regex frontend can parse pragmas.
    """
    fcode = """
SUBROUTINE FOO(A)

INTEGER, INTENT(IN) :: A

! make sure this won't end up as VariableDeclaration
! INTEGER :: B
! make sure this won't end up as VariableDeclaration
!$loki INTEGER :: C
! this is just a comment
!$loki this-is-a-pragma
!$acc this is another openacc pragma

!$omp multiline &
!$omp & pragma to be tested

END SUBROUTINE FOO
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    routine = source['FOO']

    pragmas = FindNodes(ir.Pragma).visit(routine.ir)
    var_decls = FindNodes(ir.VariableDeclaration).visit(routine.ir)

    assert len(pragmas) == 4
    assert pragmas[0].keyword == 'loki'
    assert pragmas[0].content == 'INTEGER :: C'
    assert pragmas[1].keyword == 'loki'
    assert pragmas[1].content == 'this-is-a-pragma'
    assert pragmas[2].keyword == 'acc'
    assert pragmas[2].content == 'this is another openacc pragma'
    assert pragmas[3].keyword == 'omp'
    assert pragmas[3].content == 'multiline & pragma to be tested'

    assert len(var_decls) == 1
    assert var_decls[0].symbols == ('A',)

    # compare with fully parsed source
    source.make_complete()
    compl_pragmas = FindNodes(ir.Pragma).visit(routine.ir)
    for compl_pragma, pragma in zip(compl_pragmas, pragmas):
        assert compl_pragma.keyword == pragma.keyword
        assert compl_pragma.content == pragma.content

def test_regex_comments():
    """
    Make sure the REGEX frontend doesn't match any comments
    """
    fcode = """
SUBROUTINE my_routine
! use my_mod
use other_mod, only: foo
use third_mod ! use fourth_mod
use fifth_mod! , only: bar
implicit none

! type my_type
type other_type
end type

integer :: var !, val

! $acc not an acc pragma
!$ acc also not an acc pragma
var = 1 !$acc definitely not a pragma
!!$acc not a pragma either
!$$acc no pragma

var = 1 &
    &+1!$acc again no pragma

call some_routine(var)
var = var ! + function(val)
var = var + 1 ! call other_routine(val)
!call third routine(val)
END SUBROUTINE my_routine
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=REGEX)
    routine = source['my_routine']
    assert len(routine.imports) == 3
    assert [imprt.module for imprt in routine.imports] == ['other_mod', 'third_mod', 'fifth_mod']
    assert len(calls := FindNodes(ir.CallStatement).visit(routine.ir)) == 1 and calls[0].name == 'some_routine'
    assert not FindNodes(ir.Pragma).visit(routine.ir)
