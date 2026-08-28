# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from click.testing import CliRunner

from loki.cli.loki_lint import cli
from loki.logging import log_levels


RULES_MODULE = 'loki.cli.tests.rules'

def test_loki_lint_rules(caplog):
    """ Test the CLI invocation of the loki-lint "rules" and "default-config" mode """

    caplog.clear()
    with caplog.at_level(log_levels['DEBUG']):
        # Execute command in separate runner
        result = CliRunner().invoke(cli, ['--debug', f'--rules-module={RULES_MODULE}', 'rules'])

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert 'CliDummyRule' in logout

    caplog.clear()
    with caplog.at_level(log_levels['DEBUG']):
        # Execute command in separate runner
        result = CliRunner().invoke(cli, ['--debug', f'--rules-module={RULES_MODULE}', 'default-config'])

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert 'CliDummyRule' in logout
        assert 'dummy_key' in logout


def test_loki_lint_check(tmp_path, caplog):
    """ Test the CLI invocation of the loki-lint "rules" mode """

    clean_dir = tmp_path/'clean'
    clean_dir.mkdir()
    clean_file = clean_dir/'clean.F90'
    clean_file.write_text('subroutine clean\nend subroutine clean\n')

    violating_dir = tmp_path/'violating'
    violating_dir.mkdir()
    violating_file = violating_dir/'violating.F90'
    violating_file.write_text('subroutine violating\nend subroutine violating\n')

    caplog.clear()
    with caplog.at_level(log_levels['WARNING']):
        # Execute command on a clean project
        result = CliRunner().invoke(
            cli, [
                f'--rules-module={RULES_MODULE}',
                'check', '--no-scheduler', f'--basedir={clean_dir}', '--include=*.F90'
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
                '--debug', f'--rules-module={RULES_MODULE}',
                'check', '--no-scheduler', f'--basedir={violating_dir}', '--include=*.F90'
            ]
        )

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert logout.count('[CLI.1] CliDummyRule') == 1
