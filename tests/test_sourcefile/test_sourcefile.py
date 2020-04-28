from pathlib import Path
import pytest

from loki import SourceFile, OFP, OMNI, FP


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'sourcefile.f90'


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
