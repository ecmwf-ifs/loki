from pathlib import Path
import pytest

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import MplCdstringRule
from loki.frontend import FP


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'mpl_cdstring.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_mpl_cdstring(refpath, frontend):
    handler = generate_report_handler()
    _ = generate_linter(refpath, [MplCdstringRule], frontend=frontend, handlers=[handler])
    assert len(handler.target.messages) == 2
    assert all('[3.12]' in msg for msg in handler.target.messages)
    assert all('MplCdstringRule' in msg for msg in handler.target.messages)
    assert all('"CDSTRING"' in msg for msg in handler.target.messages)
    assert all('MPL_INIT' in msg.upper() for msg in handler.target.messages)
    assert sum('(l. 13)' in msg for msg in handler.target.messages) == 1
    assert sum('(l. 18)' in msg for msg in handler.target.messages) == 1
