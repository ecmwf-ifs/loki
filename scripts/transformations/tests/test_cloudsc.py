import os
import io
import resource
from subprocess import CalledProcessError
from pathlib import Path
import shutil
import pandas as pd
import pytest

from conftest import available_frontends
from loki import execute, OMNI, HAVE_FP, HAVE_OMNI, warning

pytestmark = pytest.mark.skipif('CLOUDSC_DIR' not in os.environ, reason='CLOUDSC_DIR not set')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(os.environ['CLOUDSC_DIR'])


@pytest.fixture(scope='module', name='link_local_loki')
def fixture_link_local_loki(here):
    """Inject ourselves into the CLOUDSC bundle"""
    # Note: CLOUDSC currently uses ecbundle v2.0.0 which can't handle symlinks
    # for non-symlinked bundle entries, yet. Therefore, we monkey-patch this link
    # after the bundle create for the moment, but ideally this fixture should
    # become a dependency of the bundle create step
    lokidir = Path(__file__).parent.parent.parent.parent
    target = here/'source/loki'
    backup = here/'source/loki.bak'

    if target.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(target, backup)

    target.symlink_to(lokidir)
    yield target

    target.unlink()
    if backup.exists():
        shutil.move(backup, target)


@pytest.fixture(scope='module', name='bundle_create')
def fixture_bundle_create(here):
    # Create the bundle
    execute('./cloudsc-bundle create', cwd=here, silent=False)


@pytest.mark.usefixtures('bundle_create', 'link_local_loki')
@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs FParser for parsing dependencies')] if not HAVE_FP else None
))
def test_cloudsc(here, frontend):
    build_cmd = [
        './cloudsc-bundle', 'build', '--retry-verbose', '--clean',
        '--with-loki', '--loki-frontend=' + str(frontend), '--without-loki-install',
        '--cloudsc-prototype1=OFF', '--cloudsc-fortran=OFF', '--cloudsc-c=OFF',
    ]

    if 'CLOUDSC_ARCH' in os.environ:
        build_cmd += [f"--arch={os.environ['CLOUDSC_ARCH']}"]

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
