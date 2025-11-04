# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from click.testing import CliRunner

from loki.cli.loki_lint import cli
from loki.logging import log_levels


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


def test_loki_lint_rules(caplog):
    """ Test the CLI invocation of the loki-lint "rules" and "default-config" mode """

    caplog.clear()
    with caplog.at_level(log_levels['DEBUG']):
        # Execute command in separate runner
        result = CliRunner().invoke(cli, ['--debug', 'rules'])

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert 'MissingImplicitNoneRule' in logout
        assert 'MissingIntfbRule' in logout
        assert 'OnlyParameterGlobalVarRule' in logout

    caplog.clear()
    with caplog.at_level(log_levels['DEBUG']):
        # Execute command in separate runner
        result = CliRunner().invoke(cli, ['--debug', 'default-config'])

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert 'MissingImplicitNoneRule' in logout
        assert 'MissingIntfbRule' in logout
        assert 'OnlyParameterGlobalVarRule' in logout


def test_loki_lint_check(testdir, caplog):
    """ Test the CLI invocation of the loki-lint "rules" mode """

    projA = testdir/'sources/projA'
    projInlineCalls = testdir/'sources/projInlineCalls'

    caplog.clear()
    with caplog.at_level(log_levels['WARNING']):
        # Execute command on a clean project
        result = CliRunner().invoke(
            cli, [
                'check', '--no-scheduler', f'--basedir={projA}', '--include=*.F90'
            ]
        )
        # Check that nothing triggered
        assert result.exit_code == 0
        assert not caplog.records

    caplog.clear()
    with caplog.at_level(log_levels['INFO']):
        # Execute check command in an unclean project
        result = CliRunner().invoke(
            cli, [
                '--debug', 'check', '--no-scheduler', f'--basedir={projInlineCalls}', '--include=*.F90'
            ]
        )

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert logout.count('[L3] OnlyParameterGlobalVarRule') == 4
