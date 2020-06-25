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

    # Keywords to search for in the messages as tuples:
    # ('var name' or 'literal', 'line number', 'invalid kind value' or None)
    keywords = (
        # Declarations
        ('i', '16', None), ('j', '17', '1'), ('a(3)', '18', None), ('b', '19', '8'),
        # Literals
        ('1', '21', None), ('7', '21', None), ('2', '22', None), ('3e0', '23', None),
        ('4.0', '24', None), ('5d0', '24', None), ('6._4', '24', '4')
    )
    for keys, msg in zip(keywords, handler.target.messages):
        assert all(kw in msg for kw in keys if kw is not None)
