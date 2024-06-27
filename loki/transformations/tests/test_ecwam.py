# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import resource
from subprocess import CalledProcessError
from pathlib import Path
import pytest

from loki.tools import (
    execute, write_env_launch_script, local_loki_setup, local_loki_cleanup
)
from loki.frontend import HAVE_FP

pytestmark = pytest.mark.skipif('ECWAM_DIR' not in os.environ, reason='ECWAM_DIR not set')

@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(os.environ['ECWAM_DIR'])


@pytest.fixture(scope='module', name='local_loki_bundle')
def fixture_local_loki_bundle(here):
    """Call setup utilities for injecting ourselves into the ECWAM bundle"""
    lokidir, target, backup = local_loki_setup(here)
    yield lokidir
    local_loki_cleanup(target, backup)


@pytest.fixture(scope='module', name='bundle_create')
def fixture_bundle_create(here, local_loki_bundle):
    """Inject ourselves into the ECWAM bundle"""
    env = os.environ.copy()
    env['ECWAM_CONCEPT_BUNDLE_LOKI_DIR'] = local_loki_bundle

    # Run ecbundle to fetch dependencies
    execute(
        ['./package/bundle/ecwam-bundle', 'create', '--bundle', 'package/bundle/bundle.yml'],
        cwd=here,
        silent=False, env=env
    )


@pytest.mark.usefixtures('bundle_create')
@pytest.mark.skipif(not HAVE_FP, reason="FP needed for ECWAM parsing")
@pytest.mark.parametrize('mode', ['idem', 'idem-stack', 'scc', 'scc-stack'])
def test_ecwam(here, mode, tmp_path):
    build_dir = tmp_path/'build'
    build_cmd = [
        './package/bundle/ecwam-bundle', 'build', '--clean',
        '--with-loki', '--without-loki-install', '--loki-mode', mode,
        '--build-dir', str(build_dir)
    ]

    if 'ECWAM_ARCH' in os.environ:
        build_cmd += [f"--arch={os.environ['ECWAM_ARCH']}"]
    else:
        # Build without OpenACC support as this makes problems
        # with older versions of GNU
        build_cmd += ['--cmake=ENABLE_ACC=OFF']

    execute(build_cmd, cwd=here, silent=False)

    # Raise stack limit
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    env = os.environ.copy()
    env.update({'OMP_STACKSIZE': '2G', 'NVCOMPILER_ACC_CUDA_HEAPSIZE': '2G'})

    # create rundir
    rundir = build_dir/'wamrun_48'
    os.mkdir(rundir)

    # Run pre-processing steps
    preprocs = [
        ('ecwam-run-preproc', '--run-dir=wamrun_48', f'--config={here}/source/ecwam/tests/etopo1_oper_an_fc_O48.yml'),
        ('ecwam-run-preset', '--run-dir=wamrun_48')
    ]

    failures = {}
    for preproc, *args in preprocs:
        script = write_env_launch_script(tmp_path, preproc, args)

        # Run the script and verify error norms
        try:
            execute([str(script)], cwd=build_dir, silent=False, env=env)
        except CalledProcessError as err:
            failures[preproc] = err.returncode

    if failures:
        msg = '\n'.join([f'Non-zero return code {rcode} in {p}' for p, rcode in failures.items()])
        pytest.fail(msg)

    # Run the produced binary
    binary = 'ecwam-run-model'
    args = ('--run-dir=wamrun_48',)

    # Write a script to source env.sh and launch the binary
    script = write_env_launch_script(tmp_path, binary, args)

    def get_logs():
        logs = rundir.glob('**/*.log')
        return '\n'.join(
            (
                f'-------------------------------------------------------\n{log}:\n\n'
                + log.read_text()
            )
            for log in logs
        )

    # Run the script and verify error norms
    failure = None
    try:
        execute([str(script)], cwd=build_dir, silent=False, env=env)
    except CalledProcessError as err:
        pytest.fail(f'{binary}: Failed with error code: {err.returncode}\n{get_logs()}')

    with open(build_dir/"wamrun_48/logs/model/stdout.log") as reader:
        lines = list(reader)

    if 'Validation FAILED' in lines[-1]:
        failure = 'Validation failed'
    elif not 'Validation PASSED' in lines[-1]:
        failure = 'Validation check never run'

    if failure:
        pytest.fail(f'{binary}: {failure}\n{get_logs()}')
