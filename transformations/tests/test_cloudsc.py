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
import shutil
import pandas as pd
import pytest
import yaml

from conftest import available_frontends
from loki import execute, OMNI, HAVE_FP, HAVE_OMNI, warning

pytestmark = pytest.mark.skipif('CLOUDSC_DIR' not in os.environ, reason='CLOUDSC_DIR not set')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(os.environ['CLOUDSC_DIR'])


@pytest.fixture(scope='module', name='local_loki_bundle')
def fixture_local_loki_bundle(here):
    """Inject ourselves into the CLOUDSC bundle"""
    lokidir = Path(__file__).parent.parent.parent
    target = here/'source/loki'
    backup = here/'source/loki.bak'
    bundlefile = here/'bundle.yml'
    local_loki_bundlefile = here/'__bundle_loki.yml'

    # Do not overwrite any existing Loki copy
    if target.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(target, backup)

    # Change bundle to symlink for Loki
    bundle = yaml.safe_load(bundlefile.read_text())
    loki_index = [i for i, p in enumerate(bundle['projects']) if 'loki' in p]
    assert len(loki_index) == 1
    if 'git' in bundle['projects'][loki_index[0]]['loki']:
        del bundle['projects'][loki_index[0]]['loki']['git']
    bundle['projects'][loki_index[0]]['loki']['dir'] = str(lokidir.resolve())
    local_loki_bundlefile.write_text(yaml.dump(bundle))

    yield local_loki_bundlefile

    if local_loki_bundlefile.exists():
        local_loki_bundlefile.unlink()
    if target.is_symlink():
        target.unlink()
    if not target.exists() and backup.exists():
        shutil.move(backup, target)


@pytest.fixture(scope='module', name='bundle_create')
def fixture_bundle_create(here, local_loki_bundle):
    # Run ecbundle to fetch dependencies
    execute(
        ['./cloudsc-bundle', 'create', '--bundle', str(local_loki_bundle)],
        cwd=here,
        silent=False
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
    env.update({'OMP_STACKSIZE': '2G'})

    # For some reason, the 'data' dir symlink is not created???
    os.symlink(here/'data', here/'build/data')

    # Run the produced binaries
    binaries = [
        ('dwarf-cloudsc-loki-idem', '2', '16000', '32'),
        ('dwarf-cloudsc-loki-sca', '2', '16000', '32'),
        ('dwarf-cloudsc-loki-scc', '1', '16000', '32'),
        ('dwarf-cloudsc-loki-scc-hoist', '1', '16000', '32'),
        ('dwarf-cloudsc-loki-c', '2', '16000', '32'),
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
        script = Path(here/f'build/run_{binary}.sh')
        script.write_text(f"""
#!/bin/bash

source env.sh >&2
bin/{binary} {' '.join(args)}
exit $?
        """.strip())
        script.chmod(0o750)

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
