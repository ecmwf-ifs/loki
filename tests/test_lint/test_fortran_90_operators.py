from pathlib import Path
import pytest

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import Fortran90OperatorsRule
from loki.frontend import FP


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'fortran_90_operators.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_fortran_90_operators(refpath, frontend):
    '''Test for existence of non Fortran 90 comparison operators.'''
    handler = generate_report_handler()
    _ = generate_linter(refpath, [Fortran90OperatorsRule], frontend=frontend, handlers=[handler])

    assert len(handler.target.messages) == 11
    assert all(all(keyword in msg for keyword in ('Fortran90OperatorsRule', '[4.15]',
                                                  'Use Fortran 90 comparison operator'))
               for msg in handler.target.messages)

    f77_f90_line = (('.ne.', '/=', '7'), ('.eq.', '==', '7'),
                    ('.lt.', '<', '6'), ('.gt.', '>', '6'),
                    ('.le.', '<=', '5'), ('.ge.', '>=', '5'),
                    ('.gt.', '>', '26'), ('.gt.', '>', '32'),
                    ('.gt.', '>', '25'), ('.eq.', '==', '29'),
                    ('.le.', '<=', '23-24'))

    for keywords, message in zip(f77_f90_line, handler.target.messages):
        assert all([str(keyword) in message for keyword in keywords])
