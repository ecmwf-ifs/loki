# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import io
import resource
from subprocess import CalledProcessError
from pathlib import Path
import pandas as pd
import pytest

from conftest import (
    available_frontends, write_env_launch_script, local_loki_setup, local_loki_cleanup
)
from loki import execute, OMNI, HAVE_FP, HAVE_OMNI, warning

pytestmark = pytest.mark.skipif('CLOUDSC_DIR' not in os.environ, reason='CLOUDSC_DIR not set')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(os.environ['CLOUDSC_DIR'])


@pytest.fixture(scope='module', name='local_loki_bundle')
def fixture_local_loki_bundle(here):
    """Call setup utilities for injecting ourselves into the CLOUDSC bundle"""
    lokidir, target, backup = local_loki_setup(here)
    yield lokidir
    local_loki_cleanup(target, backup)


@pytest.fixture(scope='module', name='bundle_create')
def fixture_bundle_create(here, local_loki_bundle):
    """Inject ourselves into the CLOUDSC bundle"""
    os.environ['CLOUDSC_BUNDLE_LOKI_DIR'] = local_loki_bundle
    env = os.environ.copy()

    # Run ecbundle to fetch dependencies
    execute(
        ['./cloudsc-bundle', 'create'], cwd=here, silent=False, env=env
    )


@pytest.mark.usefixtures('bundle_create')
@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs FParser for parsing dependencies')] if not HAVE_FP else None
))
def test_cloudsc(here, frontend):
    build_cmd = [
        './cloudsc-bundle', 'build', '--retry-verbose', '--clean',
        '--with-loki', '--loki-frontend=' + str(frontend), '--without-loki-install',
        '--cloudsc-prototype1=OFF', '--cloudsc-fortran=OFF', '--cloudsc-c=OFF',
    ]

    if HAVE_OMNI:
        build_cmd += ['--with-claw']

    if 'CLOUDSC_ARCH' in os.environ:
        build_cmd += [f"--arch={os.environ['CLOUDSC_ARCH']}"]
    else:
        # Build without OpenACC support as this makes problems
        # with older versions of GNU
        build_cmd += ['--cmake=ENABLE_ACC=OFF']

    execute(build_cmd, cwd=here, silent=False)

    # Raise stack limit
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    env = os.environ.copy()
    env.update({'OMP_STACKSIZE': '2G', 'NVCOMPILER_ACC_CUDA_HEAPSIZE': '2G'})

    # For some reason, the 'data' dir symlink is not created???
    os.symlink(here/'data', here/'build/data')

    # Run the produced binaries
    binaries = [
        ('dwarf-cloudsc-loki-idem', '2', '16000', '32'),
        ('dwarf-cloudsc-loki-sca', '2', '16000', '32'),
        ('dwarf-cloudsc-loki-scc', '1', '16000', '32'),
        ('dwarf-cloudsc-loki-scc-hoist', '1', '16000', '32'),
        ('dwarf-cloudsc-loki-c', '2', '16000', '32'),
        ('dwarf-cloudsc-loki-idem-stack', '2', '16000', '32'),
        ('dwarf-cloudsc-loki-scc-stack', '1', '16000', '32'),
    ]

    if HAVE_OMNI:
        # Skip CLAW binaries if we don't have OMNI installed
        binaries += [
            ('dwarf-cloudsc-loki-claw-cpu', '2', '16000', '64'),
            ('dwarf-cloudsc-loki-claw-gpu', '1', '16000', '64'),
        ]

    failures, warnings = {}, {}
    for binary, *args in binaries:
        # Write a script to source env.sh and launch the binary
        script = write_env_launch_script(here, binary, args)

        # Run the script and verify error norms
        try:
            output = execute([str(script)], cwd=here/'build', capture_output=True, silent=False, env=env)
            results = pd.read_fwf(io.StringIO(output.stdout.decode()), index_col='Variable')
            no_errors = results['AbsMaxErr'].astype('float') == 0
            if not no_errors.all(axis=None):
                only_small_errors = results['MaxRelErr-%'].astype('float') < 1e-12
                if not only_small_errors.all(axis=None):
                    failures[binary] = results
                else:
                    warnings[binary] = results
        except CalledProcessError as err:
            failures[binary] = err.stderr.decode()

    if warnings:
        msg = '\n'.join([f'{binary}:\n{results}' for binary, results in warnings.items()])
        warning(msg)

    if failures:
        msg = '\n'.join([f'{binary}:\n{results}' for binary, results in failures.items()])
        pytest.fail(msg)
