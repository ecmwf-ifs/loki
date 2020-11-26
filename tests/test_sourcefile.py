from pathlib import Path
import pytest

from loki import (
    Sourcefile, OFP, OMNI, FP, FindNodes, PreprocessorDirective,
    Intrinsic, Assignment, Import, fgen
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_sourcefile_properties(here, frontend):
    """
    Test that all subroutines and functions are discovered
    and exposed via `subroutines` and `all_subroutines` properties.
    """
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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_sourcefile_from_source(frontend):
    """
    Test the `from_source` constructor for `Sourcefile` objects.
    """

    fcode = """
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

subroutine routine_b
  integer b
  b = 4
contains

  subroutine contained_c
    integer c
    c = 5
  end subroutine contained_c
end subroutine routine_b

function function_d(d)
  integer d
  d = 6
end function function_d
"""
    source = Sourcefile.from_source(fcode, frontend=frontend)
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


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Files are preprocessed')),
    FP
])
def test_sourcefile_pp_macros(here, frontend):
    filepath = here/'sources/sourcefile_pp_macros.F90'
    routine = Sourcefile.from_file(filepath, frontend=frontend)['routine_pp_macros']
    directives = FindNodes(PreprocessorDirective).visit(routine.ir)
    assert len(directives) == 8
    assert all(node.text.startswith('#') for node in directives)


@pytest.mark.parametrize('frontend', [
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Cannot handle directives')),
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Files are preprocessed')),
    FP
])
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


@pytest.mark.parametrize('frontend', [OFP, FP, OMNI])
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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
