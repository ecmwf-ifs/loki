import pytest
from pathlib import Path

from conftest import generate_report_handler, generate_linter
from loki.lint.rules import DrHookRule
from loki.frontend import FP


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'dr_hook.f90'


@pytest.mark.parametrize('frontend', [FP])
def test_dr_hook(refpath, frontend):
    handler = generate_report_handler()
    _ = generate_linter(refpath, [DrHookRule], frontend=frontend, handlers=[handler])

    assert len(handler.target.messages) == 13
    assert all(all(keyword in msg for keyword in ('DrHookRule', 'DR_HOOK', '[1.9]'))
               for msg in handler.target.messages)

    assert all('First executable statement must be call to DR_HOOK.' in handler.target.messages[i]
               for i in [4, 8, 10])
    assert all('Last executable statement must be call to DR_HOOK.' in handler.target.messages[i]
               for i in [9, 11])
    assert all('String argument to DR_HOOK call should be "' in handler.target.messages[i]
               for i in [0, 1, 2, 5, 12])
    assert 'Second argument to DR_HOOK call should be "0".' in handler.target.messages[6]
    assert 'Second argument to DR_HOOK call should be "1".' in handler.target.messages[3]
    assert 'Third argument to DR_HOOK call should be "ZHOOK_HANDLE".' in handler.target.messages[7]

    # Later lines come first as modules are checked before subroutines
    assert '(l. 162)' in handler.target.messages[0]
    assert '(l. 173)' in handler.target.messages[1]
    assert '(l. 177)' in handler.target.messages[2]
    assert '(l. 177)' in handler.target.messages[3]
    assert '(l. 49)' in handler.target.messages[5]
    assert '(l. 58)' in handler.target.messages[6]
    assert '(l. 63)' in handler.target.messages[7]
    assert '(l. 128)' in handler.target.messages[12]

    assert all('routine_not_okay_{}'.format(letter) in handler.target.messages[i]
               for letter, i in (('a', 4), ('c', 8), ('c', 9), ('d', 10), ('e', 11)))
