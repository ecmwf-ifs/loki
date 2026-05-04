# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for source-level model handling.
"""

from pathlib import Path
from subprocess import CalledProcessError

import pytest

from loki import (
    Module, Sourcefile, PreprocessorDirective, Assignment,
    Comment, CommentBlock, RawSource, Scalar, config, config_override
)
from loki.frontend import available_frontends, OMNI, FP, REGEX
from loki.ir import nodes as ir, FindNodes


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parents[2] / 'tests'


@pytest.mark.parametrize('frontend', available_frontends())
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


@pytest.mark.parametrize('frontend', available_frontends(
    include_regex=True, skip=[(OMNI, 'OMNI may segfault on empty files')]
))
@pytest.mark.parametrize('fcode', ['', '\n', '\n\n\n\n'])
def test_frontend_empty_file(frontend, fcode):
    """Ensure that all frontends can handle empty source files correctly (#186)"""
    source = Sourcefile.from_source(fcode, frontend=frontend)
    assert isinstance(source.ir, ir.Section)
    assert not source.to_fortran().strip()


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_properties(here, frontend, tmp_path):
    """
    Test that all subroutines and functions are discovered
    and exposed via `subroutines` and `all_subroutines` properties.
    """
    filepath = here/'sources/sourcefile.f90'
    source = Sourcefile.from_file(filepath, frontend=frontend, xmods=[tmp_path])
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
def test_sourcefile_from_source(frontend, tmp_path):
    """
    Test the `from_source` constructor for `Sourcefile` objects.
    """

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
    source = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])
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


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_lazy_construction(frontend, tmp_path):
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
    try:
        source.make_complete(frontend=frontend, xmods=[tmp_path])
    except CalledProcessError as ex:
        if frontend == OMNI and ex.returncode == -11:
            pytest.skip('F_Front segfault is a known issue on some platforms')
        raise
    assert not source._incomplete

    # Make sure no RawSource nodes are left
    assert not FindNodes(RawSource).visit(source.ir)
    if frontend == FP:
        # Some newlines are also treated as comments
        assert len(FindNodes(Comment).visit(source.ir)) == 2
    else:
        assert len(FindNodes(Comment).visit(source.ir)) == 1
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
    assert isinstance(myroutine.spec[0], RawSource)

    source.make_complete(frontend=frontend)

    assert isinstance(source.ir.body[0], Comment)
    assert isinstance(source.ir.body[2], Comment)
    if frontend == OMNI:
        assert isinstance(myroutine.body.body[0], Comment)
    else:
        assert isinstance(myroutine.docstring[0], Comment)

    code = source.to_fortran()
    assert '! Comment outside' in code
    assert '! Comment inside' in code
    assert '! Other comment outside' in code
