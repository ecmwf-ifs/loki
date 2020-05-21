from pathlib import Path
import pytest

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import BannedStatementsRule
from loki.frontend import FP


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'banned_statements.f90'


@pytest.mark.parametrize('frontend, banned_statements, passes', [
    (FP, [], True),
    (FP, ['PRINT'], False),
    (FP, ['PRINT', 'RETURN'], False),
    (FP, ['RETURN'], True)])
def test_banned_statements(refpath, frontend, banned_statements, passes):
    '''Test for banned statements.'''
    handler = generate_report_handler()
    config = {'BannedStatementsRule': {'banned': banned_statements}}
    _ = generate_linter(refpath, [BannedStatementsRule], config=config,
                        frontend=frontend, handlers=[handler])

    assert len(handler.target.messages) == 0 if passes else 1
    assert all(all(keyword in msg for keyword in ('BannedStatementsRule', '[4.11]', 'PRINT'))
               for msg in handler.target.messages)
