import pytest
from pathlib import Path

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import Fortran90OperatorsRule
from loki.frontend import FP


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'fortran_90_operators.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_fortran_90_operators(refpath, frontend):
    '''Test for existence of non Fortran 90 comparison operators.'''
    handler = generate_report_handler()
    _ = generate_linter(refpath, [Fortran90OperatorsRule], frontend=frontend, handlers=[handler])

    assert len(handler.target.messages) == 9 
    assert all(all(keyword in msg for keyword in ('Fortran90OperatorsRule', '[4.15]'))
               for msg in handler.target.messages)

    for msg, ref_line in zip(handler.target.messages, lines):
        assert 'limit of {}'.format(nesting_depth) in msg
        assert 'l. {}'.format(ref_line) in msg
