from pathlib import Path
import pytest

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import ModuleNamingRule
from loki.frontend import FP


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'module_naming_mod.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_module_naming_ok(refpath, frontend):
    '''Test file and modules for checking that naming is correct and matches each other.'''
    handler = generate_report_handler()
    _ = generate_linter(refpath, [ModuleNamingRule], frontend=frontend, handlers=[handler])

    assert len(handler.target.messages) == 2
    assert all(all(keyword in msg for keyword in ('ModuleNamingRule', '[1.5]'))
               for msg in handler.target.messages)

    assert all('"module_naming"' in msg for msg in handler.target.messages)
    assert all(keyword in handler.target.messages[0] for keyword in ('"_mod"', 'Name of module'))
    assert all(keyword in handler.target.messages[1] for keyword in (refpath.name, 'filename'))
