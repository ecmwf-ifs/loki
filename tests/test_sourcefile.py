# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, clean_test, available_frontends
from loki import (
    Sourcefile, OFP, OMNI, REGEX, FindNodes, PreprocessorDirective,
    Intrinsic, Assignment, Import, fgen, ProcedureType, ProcedureSymbol,
    StatementFunction, Comment, CommentBlock, RawSource, Scalar
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_properties(here, frontend):
    """
    Test that all subroutines and functions are discovered
    and exposed via `subroutines` and `all_subroutines` properties.
    """
    # pylint: disable=no-member
    filepath = here/'sources/sourcefile.f90'
    source = Sourcefile.from_file(filepath, frontend=frontend)
    assert len(source.subroutines) == 3
    assert len(source.all_subroutines) == 5

    subroutines = ['routine_a', 'routine_b', 'function_d']
    all_subroutines = subroutines + ['module_routine', 'module_function']
    contained_routines = ['contained_c']

    assert sum(routine.name in subroutines for routine in source.subroutines) == 3
    assert sum(routine.name in all_subroutines for routine in source.subroutines) == 3
    assert sum(routine.name in contained_routines for routine in source.subroutines) == 0

    assert sum(routine.name in subroutines for routine in source.all_subroutines) == 3
    assert sum(routine.name in all_subroutines for routine in source.all_subroutines) == 5
    assert sum(routine.name in contained_routines for routine in source.all_subroutines) == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_from_source(frontend):
    """
    Test the `from_source` constructor for `Sourcefile` objects.
    """
    # pylint: disable=no-member

    fcode = """
! Some comment
subroutine routine_a
  integer a
  a = 1
end subroutine routine_a

! Some comment
module some_module
contains
  subroutine module_routine
    integer m
    m = 2
  end subroutine module_routine
  function module_function(n)
    integer n
    n = 3
  end function module_function
end module some_module
! Other comment

subroutine routine_b
  integer b
  b = 4
contains
  subroutine contained_c
    integer c
    c = 5
  end subroutine contained_c
end subroutine routine_b
! Other comment

function function_d(d)
  integer d
  d = 6
end function function_d
""".strip()
    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert len(source.subroutines) == 3
    assert len(source.all_subroutines) == 5

    subroutines = ['routine_a', 'routine_b', 'function_d']
    all_subroutines = subroutines + ['module_routine', 'module_function']

    assert [routine.name.lower() for routine in source.subroutines] == subroutines
    assert [routine.name.lower() for routine in source.all_subroutines] == all_subroutines
    assert 'contained_c' not in [routine.name.lower() for routine in source.subroutines]
    assert 'contained_c' not in [routine.name.lower() for routine in source.all_subroutines]

    comments = FindNodes((Comment, CommentBlock)).visit(source.ir)
    assert len(comments) == 4
    assert all(comment.text.strip() in ['! Some comment', '! Other comment'] for comment in comments)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Files are preprocessed')]))
def test_sourcefile_pp_macros(here, frontend):
    filepath = here/'sources/sourcefile_pp_macros.F90'
    routine = Sourcefile.from_file(filepath, frontend=frontend)['routine_pp_macros']
    directives = FindNodes(PreprocessorDirective).visit(routine.ir)
    assert len(directives) == 8
    assert all(node.text.startswith('#') for node in directives)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[
    (OFP, 'Cannot handle directives'), (OMNI, 'Files are preprocessed')
]))
def test_sourcefile_pp_directives(here, frontend):
    filepath = here/'sources/sourcefile_pp_directives.F90'
    routine = Sourcefile.from_file(filepath, frontend=frontend)['routine_pp_directives']

    # Note: these checks are rather loose as we currently do not restore the original version but
    # simply replace the PP constants by strings
    directives = FindNodes(PreprocessorDirective).visit(routine.body)
    assert len(directives) == 1
    assert directives[0].text == '#define __FILENAME__ __FILE__'
    intrinsics = FindNodes(Intrinsic).visit(routine.body)
    assert '__FILENAME__' in intrinsics[0].text and '__DATE__' in intrinsics[0].text
    assert '__FILE__' in intrinsics[1].text and '__VERSION__' in intrinsics[1].text

    statements = FindNodes(Assignment).visit(routine.body)
    assert len(statements) == 1
    assert fgen(statements[0]) == 'y = 0*5 + 0'


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_pp_include(here, frontend):
    filepath = here/'sources/sourcefile_pp_include.F90'
    sourcefile = Sourcefile.from_file(filepath, frontend=frontend, includes=[here/'include'])
    routine = sourcefile['routine_pp_include']

    statements = FindNodes(Assignment).visit(routine.body)
    assert len(statements) == 1
    if frontend == OMNI:
        # OMNI resolves that statement function!
        assert fgen(statements[0]) == 'c = real(a + b, kind=4)'
    else:
        assert fgen(statements[0]) == 'c = add(a, b)'

    if frontend is not OMNI:
        # OMNI resolves the import in the frontend
        imports = FindNodes(Import).visit([routine.spec, routine.body])
        assert len(imports) == 1
        assert imports[0].c_import
        assert imports[0].module == 'some_header.h'


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_cpp_preprocessing(here, frontend):
    """
    Test the use of the external CPP-preprocessor.
    """
    filepath = here/'sources/sourcefile_cpp_preprocessing.F90'

    source = Sourcefile.from_file(filepath, preprocess=True, frontend=frontend)
    routine = source['sourcefile_external_preprocessing']
    directives = FindNodes(PreprocessorDirective).visit(routine.ir)

    if frontend is not OMNI:
        # OMNI skips the import in the frontend
        imports = FindNodes(Import).visit([routine.spec, routine.body])
        assert len(imports) == 1
        assert imports[0].c_import
        assert imports[0].module == 'some_header.h'

    assert len(directives) == 0
    assert 'b = 123' in fgen(routine)

    # Check that the ``define`` gets propagated correctly
    source = Sourcefile.from_file(filepath, preprocess=True, defines='FLAG_SMALL',
                                  frontend=frontend)
    routine = source['sourcefile_external_preprocessing']
    directives = FindNodes(PreprocessorDirective).visit(routine.ir)

    assert len(directives) == 0
    assert 'b = 6' in fgen(routine)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP, 'No support for statement functions')]))
def test_sourcefile_cpp_stmt_func(here, frontend):
    """
    Test the correct identification of statement functions
    after inlining by preprocessor.
    """
    sourcepath = here/'sources'
    filepath = sourcepath/'sourcefile_cpp_stmt_func.F90'

    source = Sourcefile.from_file(filepath, includes=sourcepath, preprocess=True, frontend=frontend)
    module = source['sourcefile_cpp_stmt_func_mod']
    module.name += f'_{frontend!s}'

    # OMNI inlines statement functions, so we can't check the representation
    if frontend != OMNI:
        routine = source['sourcefile_cpp_stmt_func']
        stmt_func_decls = FindNodes(StatementFunction).visit(routine.spec)
        assert len(stmt_func_decls) == 4

        for decl in stmt_func_decls:
            var = routine.variable_map[str(decl.variable)]
            assert isinstance(var, ProcedureSymbol)
            assert isinstance(var.type.dtype, ProcedureType)
            assert var.type.dtype.procedure is decl

    # Generate code and compile
    filepath = here/f'{module.name}.f90'
    mod = jit_compile(source, filepath=filepath, objname=module.name)

    # Verify it produces correct results
    klon, klev = 10, 5
    kidia, kfdia = 1, klon
    zfoeew = np.zeros((klon, klev), order='F')
    mod.sourcefile_cpp_stmt_func(kidia, kfdia, klon, klev, zfoeew)
    assert (zfoeew == 0.25).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_lazy_construction(frontend):
    """
    Test delayed ("lazy") parsing of sourcefile content
    """
    fcode = """
! A comment to test
subroutine routine_a
integer a
a = 1
end subroutine routine_a

module some_module
contains
subroutine module_routine
integer m
m = 2
end subroutine module_routine
function module_function(n)
integer n
n = 3
end function module_function
end module some_module

#ifndef SOME_PREPROC_VAR
subroutine routine_b
integer b
b = 4
contains
subroutine contained_c
integer c
c = 5
end subroutine contained_c
end subroutine routine_b
#endif

function function_d(d)
integer d
d = 6
end function function_d
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)
    assert len(source.subroutines) == 3
    assert len(source.all_subroutines) == 5

    some_module = source['some_module']
    routine_b = source['routine_b']
    module_routine = some_module['module_routine']
    function_d = source['function_d']
    assert function_d.arguments == ()

    # Make sure we have an incomplete parse tree until now
    assert source._incomplete
    assert len(FindNodes(RawSource).visit(source.ir)) == 5
    assert len(FindNodes(RawSource).visit(source['routine_a'].ir)) == 1

    # Trigger the full parse
    source.make_complete(frontend=frontend)
    assert not source._incomplete

    # Make sure no RawSource nodes are left
    assert not FindNodes(RawSource).visit(source.ir)
    assert len(FindNodes(Comment).visit(source.ir)) in (1, 2) # Some newlines are also treated as comments
    if frontend == OMNI:
        assert not FindNodes(PreprocessorDirective).visit(source.ir)
    else:
        assert len(FindNodes(PreprocessorDirective).visit(source.ir)) == 2
    for routine in source.all_subroutines:
        assert not FindNodes(RawSource).visit(routine.ir)
        assert len(FindNodes(Assignment).visit(routine.ir)) == 1

    # The previously generated ProgramUnit objects should be the same as before
    assert routine_b is source['routine_b']
    assert some_module is source['some_module']
    assert module_routine is source['some_module']['module_routine']
    assert function_d.arguments == ('d',)
    assert isinstance(function_d.arguments[0], Scalar)


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_lazy_comments(frontend):
    """
    Make sure that lazy construction can handle comments on source file level
    (i.e. outside a program unit)
    """
    fcode = """
! Comment outside
subroutine myroutine
    ! Comment inside
end subroutine myroutine
! Other comment outside
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=REGEX)

    assert isinstance(source.ir.body[0], RawSource)
    assert isinstance(source.ir.body[2], RawSource)

    myroutine = source['myroutine']
    assert isinstance(myroutine.spec.body[0], RawSource)

    source.make_complete(frontend=frontend)

    assert isinstance(source.ir.body[0], Comment)
    assert isinstance(source.ir.body[2], Comment)
    assert isinstance(myroutine.body.body[0], Comment)

    code = source.to_fortran()
    assert '! Comment outside' in code
    assert '! Comment inside' in code
    assert '! Other comment outside' in code
