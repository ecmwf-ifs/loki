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

from loki.tools import (
    execute, write_env_launch_script, local_loki_setup, local_loki_cleanup
)
from loki.frontend import available_frontends, OMNI, HAVE_FP
from loki.logging import warning

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
    env = os.environ.copy()
    env['CLOUDSC_BUNDLE_LOKI_DIR'] = local_loki_bundle

    # Run ecbundle to fetch dependencies
    execute(
        ['./cloudsc-bundle', 'create'], cwd=here, silent=False, env=env
    )


@pytest.mark.usefixtures('bundle_create')
@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs FParser for parsing headers')] if not HAVE_FP else None
))
def test_cloudsc(here, frontend):
    build_cmd = [
        './cloudsc-bundle', 'build', '--retry-verbose', '--clean',
        '--with-loki=ON', '--loki-frontend=' + str(frontend), '--without-loki-install',
        '--with-double-precision=ON', '--with-single-precision=ON'
    ]

    if 'CLOUDSC_ARCH' in os.environ:
        build_cmd += [f"--arch={os.environ['CLOUDSC_ARCH']}"]

    execute(build_cmd, cwd=here, silent=False)

    # Raise stack limit
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    env = os.environ.copy()
    env.update({'OMP_STACKSIZE': '2G', 'NVCOMPILER_ACC_CUDA_HEAPSIZE': '2G'})

    # For some reason, the 'data' dir symlink is not created???
    os.symlink(here/'data', here/'build/data')

    # Run the produced binaries
    binaries = [('dwarf-cloudsc-loki-c-dp', '2', '16000', '32')]
    for prec in ('dp', 'sp'):
        binaries += [
            (f'dwarf-cloudsc-loki-idem-{prec}', '2', '16000', '32'),
            (f'dwarf-cloudsc-loki-idem-stack-{prec}', '2', '16000', '32'),
            (f'dwarf-cloudsc-loki-scc-{prec}', '1', '16000', '32'),
            (f'dwarf-cloudsc-loki-scc-hoist-{prec}', '1', '16000', '32'),
            (f'dwarf-cloudsc-loki-scc-stack-{prec}', '1', '16000', '32'),
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
                # We report only validation failures for double-precision as the single-precision
                # result validation is known to fail due to a lack of suitable reference data
                if binary.endswith('-dp') and not only_small_errors.all(axis=None):
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
