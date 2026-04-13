# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import tomli_w

from click.testing import CliRunner

from loki import Sourcefile
from loki.cli.loki_transform import cli
from loki.ir import FindNodes, nodes as ir
from loki.logging import log_levels


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


@pytest.fixture(name='config')
def fixture_config():
    """
    Default configuration dict with basic options.
    """
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'disable': ['abort'],
            'enable_imports': True,
        },
        'routines': {},
        'transformations': {
            'Idem': {
                'classname': 'IdemTransformation',
                'module': 'loki.transformations',
            },
            'ModuleWrap': {
                'classname': 'ModuleWrapTransformation',
                'module': 'loki.transformations.build_system',
                'options': {'module_suffix': '_MOD'},
            },
            'Dependency': {
                'classname': 'DependencyTransformation',
                'module': 'loki.transformations.build_system',
                'options': {'suffix': '_LOKI', 'module_suffix': '_MOD'},
            },
        },
        'pipelines': {
            'idem': {
                'transformations': ['Idem', 'ModuleWrap', 'Dependency']
            }
        }
    }


def test_loki_transform_plan(testdir, config, caplog, tmp_path):
    """ Test the CLI invocation of the "plan" mode """

    projA = testdir/'sources/projA'
    projA_files = [
        'driverA_mod.f90', 'kernelA_mod.F90', 'compute_l1_mod.f90',
        'another_l1.F90', 'compute_l2_mod.f90', 'another_l2.F90'
    ]

    # Create final config
    config['routines'] = {
        'driverA': {'role': 'driver'},
        'another_l1': {'role': 'driver'},
    }
    (tmp_path/'my.config').write_text(tomli_w.dumps(config))
    plan_file = tmp_path/'plan.cmake'
    assert not plan_file.exists()

    caplog.clear()
    with caplog.at_level(log_levels['INFO']):
        # Execute command in separate runner
        result = CliRunner().invoke(
            cli, [
                '--debug', 'plan', '--mode=idem', f'--config={tmp_path}/my.config',
                '--frontend=fp', f'--source={projA}', f'--root={projA}',
                f'--build={tmp_path}/build', f'--header={projA}/module/header_mod.f90',
                f'--plan-file={tmp_path}/plan.cmake'
            ]
        )

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert '[Loki::Scheduler] Performed initial source scan' in logout
        assert '[Loki] Scheduler writing CMake plan' in logout
        assert '[Loki::Scheduler] Applied transformation <CMakePlanTransformation>' in logout

        # Check generated plan file
        assert plan_file.exists()
        plan_str = plan_file.read_text()
        for fname in projA_files:
            # Check that each file is named twice and the modified once
            assert plan_str.count(fname) == 2
            fname_mod = fname.replace('.f90', '.idem.f90').replace('.F90', '.idem.F90')
            assert plan_str.count(fname_mod) == 1

    assert not plan_file.unlink()


def test_loki_transform_convert(testdir, config, caplog, tmp_path):
    """ Test the CLI invocation of the "convert" mode """

    projA = testdir/'sources/projA'
    projA_files = [
        'driverA_mod.f90', 'kernelA_mod.F90', 'compute_l1_mod.f90',
        'another_l1.F90', 'compute_l2_mod.f90', 'another_l2.F90'
    ]

    # Create final config and build directory
    config['routines'] = {
        'driverA': {'role': 'driver'},
        'another_l1': {'role': 'driver'},
    }
    (tmp_path/'my.config').write_text(tomli_w.dumps(config))
    (tmp_path/'build').mkdir()

    caplog.clear()
    with caplog.at_level(log_levels['INFO']):
        # Execute command in separate runner
        result = CliRunner().invoke(
            cli, [
                '--debug', 'convert', '--mode=idem', f'--config={tmp_path}/my.config',
                '--frontend=fp', f'--source={projA}', f'--root={projA}',
                f'--build={tmp_path}/build', f'--header={projA}/module/header_mod.f90',
            ]
        )

        # Check execution and logs for certain messages
        assert result.exit_code == 0
        logout = ''.join(str(r) for r in caplog.records)
        assert '[Loki::Scheduler] Performed initial source scan' in logout
        assert '[Loki::Scheduler] Applied transformation <FileWriteTransformation>' in logout

        for fname in projA_files:
            # Ensure all files have been generated
            fname_mod = fname.replace('.f90', '.idem.f90').replace('.F90', '.idem.F90')
            assert (tmp_path/'build'/fname_mod).exists()


def test_loki_transform_ass_wipe(caplog, tmp_path):
    """Test the CLI invocation of "convert --ass-wipe" on a source file."""

    source_file = tmp_path/'ass_wipe_test.F90'
    source_file.write_text(
    """
subroutine ass_wipe_test(a, b, c)
  use iso_fortran_env, only: real64
  implicit none
  real(kind=real64), intent(in) :: a, b
  real(kind=real64), intent(out) :: c

  associate (x => a, y => b)
    c = x + y
  end associate
end subroutine ass_wipe_test
""".strip() + '\n'
    )

    caplog.clear()
    with caplog.at_level(log_levels['INFO']):
        result = CliRunner().invoke(
            cli, [
                '--debug', 'convert', '--frontend=fp',
                f'--source={source_file}', '--ass-wipe',
            ]
        )

    assert result.exit_code == 0
    updated = source_file.read_text().lower()
    assert 'associate' not in updated
    assert 'c = a + b' in updated

    logout = ''.join(str(r) for r in caplog.records)
    assert '[Loki] Removing ASSOCIATE blocks from source files' in logout
    assert '[Loki] ASS-WIPE request completed' in logout


def test_loki_transform_ass_wipe_nested(caplog, tmp_path):
    """Test "convert --ass-wipe" resolves nested ASSOCIATE blocks."""

    source_file = tmp_path/'ass_wipe_nested_test.F90'
    source_file.write_text(
                """
subroutine ass_wipe_nested_test(a, b, c)
    use iso_fortran_env, only: real64
    implicit none
    real(kind=real64), intent(in) :: a, b
    real(kind=real64), intent(out) :: c

    associate (x => a + b)
        associate (y => x * 2.0_real64)
            c = y
        end associate
    end associate
end subroutine ass_wipe_nested_test
""".strip() + '\n'
    )

    caplog.clear()
    with caplog.at_level(log_levels['INFO']):
        result = CliRunner().invoke(
                        cli, [
                                '--debug', 'convert', '--frontend=fp',
                                f'--source={source_file}', '--ass-wipe',
                        ]
                )

    assert result.exit_code == 0
    updated = source_file.read_text().lower()
    assert 'associate' not in updated
    assert 'c = (a + b)*2.0_real64' in updated

    logout = ''.join(str(r) for r in caplog.records)
    assert '[Loki] Removing ASSOCIATE blocks from source files' in logout
    assert '[Loki] ASS-WIPE request completed' in logout


def test_loki_transform_ass_wipe_nested_slices(caplog, tmp_path):
    """Test "convert --ass-wipe" resolves nested ASSOCIATE blocks with array slices."""

    source_file = tmp_path/'ass_wipe_nested_slices_test.F90'
    source_file.write_text(
                """
subroutine ass_wipe_nested_slices_test(n, arr, c)
    use iso_fortran_env, only: real64
    implicit none
    integer, intent(in) :: n
    real(kind=real64), intent(in) :: arr(n, 2)
    real(kind=real64), intent(out) :: c(n)

    associate (x => arr(:, 1), y => arr(:, 2))
        associate (tmp => x + y)
            c(:) = tmp(:)
        end associate
    end associate
end subroutine ass_wipe_nested_slices_test
""".strip() + '\n'
        )

    caplog.clear()
    with caplog.at_level(log_levels['INFO']):
        result = CliRunner().invoke(
                        cli, [
                                '--debug', 'convert', '--frontend=fp',
                                f'--source={source_file}', '--ass-wipe',
                        ]
                )

    assert result.exit_code == 0
    updated = source_file.read_text().lower()
    assert 'associate' not in updated

    sfile = Sourcefile.from_file(source_file, frontend='fp')
    associates = FindNodes(ir.Associate).visit(sfile.ir)
    assert len(associates) == 0

    logout = ''.join(str(r) for r in caplog.records)
    assert '[Loki] Removing ASSOCIATE blocks from source files' in logout
    assert '[Loki] ASS-WIPE request completed' in logout
