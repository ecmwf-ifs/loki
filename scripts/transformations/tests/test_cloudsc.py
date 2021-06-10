import os
import io
from pathlib import Path
import pandas as pd
import pytest

from loki import execute, OFP, OMNI, FP

pytestmark = pytest.mark.skipif('CLOUDSC_DIR' not in os.environ, reason='CLOUDSC_DIR not set')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(os.environ['CLOUDSC_DIR'])


@pytest.fixture(scope='module', autouse=True)
def fixture_bundle_create(here):
    # Create the bundle
    execute('./cloudsc-bundle create', cwd=here, silent=False)

    # Reset cloudsc.F90
    execute('git checkout -- src/cloudsc_loki/cloudsc.F90', cwd=here, silent=False)

    # Apply precision bug for reproducible results
    patch = b"""
diff --git a/src/cloudsc_loki/cloudsc.F90 b/src/cloudsc_loki/cloudsc.F90
index 887a9e4..acefd85 100644
--- a/src/cloudsc_loki/cloudsc.F90
+++ b/src/cloudsc_loki/cloudsc.F90
@@ -1964,7 +1964,7 @@ DO JK=NCLDTOP,KLEV
     IF(ZTP1(JL,JK) <= RTT .AND. ZLIQCLD(JL)>ZEPSEC) THEN
 
       ! Fallspeed air density correction 
-      ZFALLCORR = (RDENSREF/ZRHO(JL))**0.4_JPRB
+      ZFALLCORR = (RDENSREF/ZRHO(JL))**0.4
 
       !------------------------------------------------------------------
       ! Riming of snow by cloud water - implicit in lwc
    """
    execute('patch -p1', cwd=here, input=patch)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_cloudsc(here, frontend):
    build_cmd = [
        './cloudsc-bundle', 'build', '--verbose', '--clean',
        '--with-loki', '--loki-frontend=' + str(frontend),
        '--cloudsc-prototype1=OFF', '--cloudsc-fortran=OFF', '--cloudsc-c=OFF',
        '--cmake="ENABLE_ACC=OFF"'
    ]

    if 'CLOUDSC_ARCH' in os.environ:
        build_cmd += ['--arch={}'.format(os.environ['CLOUDSC_ARCH'])]

    execute(build_cmd, cwd=here, silent=False)

    # For some reason, the 'data' dir symlink is not created???
    os.symlink(here/'data', here/'build/data')

    # Run the produced binaries
    binaries = [
        'dwarf-cloudsc-loki-claw-cpu', 'dwarf-cloudsc-loki-claw-gpu',
        'dwarf-cloudsc-loki-idem', 'dwarf-cloudsc-loki-sca'
    ]

    failures = {}
    for binary in binaries:
        # TODO: figure out how to source env.sh
        run_cmd = 'bin/{}'.format(binary)
        output = execute(run_cmd, cwd=here/'build', capture_output=True, silent=False)
        results = pd.read_fwf(io.StringIO(output.stdout.decode()), index_col='Variable')
        no_errors = results['AbsMaxErr'].astype('float') == 0
        if not no_errors.all(axis=None):
            failures[binary] = results

    # Handle C separately because of precision bug
    binary = 'dwarf-cloudsc-loki-c'
    output = execute(run_cmd, cwd=here/'build', capture_output=True, silent=False)
    results = pd.read_fwf(io.StringIO(output.stdout.decode()), index_col='Variable')
    no_errors = results['AbsMaxErr'].astype('float') < 1e-8
    if not no_errors.all(axis=None):
        failures[binary] = results

    if failures:
        pytest.fail(str(failures))
