from pathlib import Path
import pytest

from loki import SourceFile, OFP, OMNI, FP


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent/'sources/sourcefile.f90'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_subroutine_properties(refpath, frontend):
    """Test that all subroutines and functions are discovered
    and exposed via `subroutines` and `all_subroutines` properties."""
    source = SourceFile.from_file(refpath, frontend=frontend)
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
def test_subroutine_from_source(frontend):
    """
    Test the `from_source` constructor for `SourceFile` objects.
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
    source = SourceFile.from_source(fcode, frontend=frontend)
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
