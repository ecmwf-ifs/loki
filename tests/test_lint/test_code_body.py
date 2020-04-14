import pytest
from pathlib import Path

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import CodeBodyRule
from loki.frontend import FP


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'code_body.f90'


@pytest.mark.parametrize('frontend, nesting_depth, lines', [
    (FP, 3, []),
    (FP, 2, [6, 12, 16, 22, 28, 35]),
    (FP, 1, [5, 6, 10, 12, 16, 22, 27, 28, 33, 34, 35])])
def test_code_body_messages(refpath, frontend, nesting_depth, lines):
    handler = generate_report_handler()
    config = {'CodeBodyRule': {'max_nesting_depth': nesting_depth}}
    _ = generate_linter(refpath, [CodeBodyRule], config=config,
                        frontend=frontend, handlers=[handler])
    assert len(handler.target.messages) == len(lines)
    print('\n'.join(handler.target.messages))

    for msg, ref_line in zip(handler.target.messages, lines):
        assert 'limit of {}'.format(nesting_depth) in msg
        assert 'l. {}'.format(ref_line) in msg
        assert 'CodeBodyRule' in msg
