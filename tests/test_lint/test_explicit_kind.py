import pytest
from pathlib import Path

from conftest import generate_report_handler, generate_linter
from loki.lint.linter import Linter
from loki.lint.rules import ExplicitKindRule
from loki.frontend import FP
from loki.sourcefile import SourceFile


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'explicit_kind.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_explicit_kind(refpath, frontend):
    handler = generate_report_handler()
    _ = generate_linter(refpath, [ExplicitKindRule], frontend=frontend, handlers=[handler])
    assert len(handler.target.messages) == 11
    # literals
    assert 'l. 17' in handler.target.messages[0] and '"1"' in handler.target.messages[0]
    assert 'l. 17' in handler.target.messages[1] and '"7"' in handler.target.messages[1]
    assert 'l. 18' in handler.target.messages[2] and '"2"' in handler.target.messages[2]
    assert 'l. 19' in handler.target.messages[3] and '"3E0"' in handler.target.messages[3]
    assert 'l. 20' in handler.target.messages[4] and '"4.0"' in handler.target.messages[4]
    assert 'l. 20' in handler.target.messages[5] and '"5D0"' in handler.target.messages[5]
    assert 'l. 20' in handler.target.messages[6] and '"4"' in handler.target.messages[6]
    # declarations
    assert 'routine "routine_not_okay"' in handler.target.messages[7] and '"i"' in handler.target.messages[7]
    assert 'routine "routine_not_okay"' in handler.target.messages[8] and '"j"' in handler.target.messages[8] \
        and '"1"' in handler.target.messages[8]
    assert 'routine "routine_not_okay"' in handler.target.messages[9] and '"a"' in handler.target.messages[9]
    assert 'routine "routine_not_okay"' in handler.target.messages[10] and '"b"' in handler.target.messages[10]
