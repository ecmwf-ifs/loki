from pathlib import Path
import pytest

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import ExplicitKindRule
from loki.frontend import FP


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'explicit_kind.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_explicit_kind(refpath, frontend):
    handler = generate_report_handler()
    # Need to include INTEGER constants in config as (temporarily) removed from defaults
    config = {'ExplicitKindRule': {'constant_types': ['REAL', 'INTEGER']}}
    _ = generate_linter(refpath, [ExplicitKindRule], config=config,
                        frontend=frontend, handlers=[handler])
    assert len(handler.target.messages) == 11
    assert all('[4.7]' in msg for msg in handler.target.messages)
    assert all('ExplicitKindRule' in msg for msg in handler.target.messages)
    # declarations
    assert 'routine "routine_not_okay"' in handler.target.messages[0] and \
        '"i"' in handler.target.messages[0]
    assert 'routine "routine_not_okay"' in handler.target.messages[1] and \
        '"j"' in handler.target.messages[1] and '"1"' in handler.target.messages[1]
    assert 'routine "routine_not_okay"' in handler.target.messages[2] and \
        '"a(3)"' in handler.target.messages[2]
    assert 'routine "routine_not_okay"' in handler.target.messages[3] and \
        '"b"' in handler.target.messages[3]
    # literals
    assert 'l. 17' in handler.target.messages[4] and '"1"' in handler.target.messages[4]
    assert 'l. 17' in handler.target.messages[5] and '"7"' in handler.target.messages[5]
    assert 'l. 18' in handler.target.messages[6] and '"2"' in handler.target.messages[6]
    assert 'l. 19' in handler.target.messages[7] and '"3E0"' in handler.target.messages[7]
    assert 'l. 20' in handler.target.messages[8] and '"4.0"' in handler.target.messages[8]
    assert 'l. 20' in handler.target.messages[9] and '"5D0"' in handler.target.messages[9]
    assert 'l. 20' in handler.target.messages[10] and '"4"' in handler.target.messages[10]
