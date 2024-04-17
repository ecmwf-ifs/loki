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
from loki.frontend import HAVE_FP, FP

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
        ['./ecwam-bundle', 'create'],
        cwd=here,
        silent=False, env=env
    )


@pytest.mark.usefixtures('bundle_create')
@pytest.mark.skipif(not HAVE_FP, reason="FP needed for ECWAM parsing")
def test_ecwam(here, frontend=FP):
    build_cmd = [
        './ecwam-bundle', 'build', '--clean',
        '--with-loki', '--loki-frontend=' + str(frontend), '--without-loki-install'
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
    os.mkdir(here/'build/wamrun_48')

    # Run pre-processing steps
    preprocs = [
        ('ecwam-run-preproc', '--run-dir=wamrun_48', '--config=../source/ecwam/tests/etopo1_oper_an_fc_O48.yml'),
        ('ecwam-run-preset', '--run-dir=wamrun_48')
    ]

    failures = {}
    for preproc, *args in preprocs:
        script = write_env_launch_script(here, preproc, args)

        # Run the script and verify error norms
        try:
            execute([str(script)], cwd=here/'build', silent=False, env=env)
        except CalledProcessError as err:
            failures[preproc] = err.returncode

    if failures:
        msg = '\n'.join([f'Non-zero return code {rcode} in {p}' for p, rcode in failures.items()])
        pytest.fail(msg)

    # Run the produced binaries
    binaries = [
        ('ecwam-run-model', '--run-dir=wamrun_48', '--variant=loki-idem'),
        ('ecwam-run-model', '--run-dir=wamrun_48', '--variant=loki-idem-stack'),
        ('ecwam-run-model', '--run-dir=wamrun_48', '--variant=loki-scc'),
        ('ecwam-run-model', '--run-dir=wamrun_48', '--variant=loki-scc-stack')
    ]

    failures = {}
    for binary, *args in binaries:
        # Write a script to source env.sh and launch the binary
        script = write_env_launch_script(here, binary, args)

        # Run the script and verify error norms
        try:
            execute([str(script)], cwd=here/'build', silent=False, env=env)
            with open(here/"build/wamrun_48/logs/model/stdout.log") as reader:
                lines = list(reader)

            if 'Validation FAILED' in lines[-1]:
                failures[binary] = 'Validation failed'
            elif not 'Validation PASSED' in lines[-1]:
                failures[binary] = 'Validation check never run'
        except CalledProcessError as err:
            failures[binary] = f'Failed with error code: {err.returncode}'

    if failures:
        msg = '\n'.join([f'{binary}: {stat}' for binary, stat in failures.items()])
        pytest.fail(msg)
