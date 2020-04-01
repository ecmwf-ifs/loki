import pytest
from pathlib import Path

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import ImplicitNoneRule
from loki.frontend import FP


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'implicit_none.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_implicit_none(refpath, frontend):
    handler = generate_report_handler()
    _ = generate_linter(refpath, [ImplicitNoneRule], frontend=frontend, handlers=[handler])
    print('\n'.join(handler.target.messages))
    assert len(handler.target.messages) == 5
    assert all('"IMPLICIT NONE"' in msg for msg in handler.target.messages)
    assert sum('"routine_not_okay"' in msg for msg in handler.target.messages) == 1
    assert sum('"routine_also_not_okay"' in msg for msg in handler.target.messages) == 1
    assert sum('"contained_routine_not_okay"' in msg for msg in handler.target.messages) == 1
    assert sum('"contained_mod_routine_not_okay"' in msg for msg in handler.target.messages) == 1
    assert sum('"contained_contained_routine_not_okay"' in msg
               for msg in handler.target.messages) == 1
