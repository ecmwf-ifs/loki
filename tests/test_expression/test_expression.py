import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, fgen, OFP, OMNI


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'expression.f90'


@pytest.fixture(scope='module')
def reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    return compile_and_load(refpath)


def generate_identity(refpath, routinename, frontend=OFP):
    """
    Generate the "identity" of a single subroutine with a frontend-specific suffix.
    """
    testname = refpath.parent/('%s_%s_%s.f90' % (refpath.stem, routinename, frontend))
    source = SourceFile.from_file(refpath, frontend=frontend)
    routine = [r for r in source.subroutines if r.name == routinename][0]
    routine.name += '_%s' % frontend
    source.write(source=fgen(routine), filename=testname)
    return compile_and_load(testname)


@pytest.mark.parametrize('frontend', [OFP])
def test_simple_expr(refpath, reference, frontend):
    """
    v5 = (v1 + v2) * (v3 - v4)
    v6 = (v1 ** v2) - (v3 / v4)
    """
    # Test the reference solution
    v5, v6 = reference.simple_expr(2., 3., 10., 5.)
    assert v5 == 25. and v6 == 6.

    # Test the generated identity
    test = generate_identity(refpath, 'simple_expr', frontend=frontend)
    function = getattr(test, 'simple_expr_%s' % frontend)
    v5, v6 = function(2., 3., 10., 5.)
    assert v5 == 25. and v6 == 6.


@pytest.mark.parametrize('frontend', [OFP])
def test_intrinsic_functions(refpath, reference, frontend):
    """
    vmin = min(v1, v2)
    vmax = max(v1, v2)
    vabs = abs(v1 - v2)
    vexp = exp(v1 + v2)
    vsqrt = sqrt(v1 + v2)
    vlog = log(v1 + v2)
    """
    # Test the reference solution
    vmin, vmax, vabs, vexp, vsqrt, vlog = reference.intrinsic_functions(2., 4.)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vexp == np.exp(6.) and vsqrt == np.sqrt(6.) and vlog == np.log(6.)

    # Test the generated identity
    test = generate_identity(refpath, 'intrinsic_functions', frontend=frontend)
    function = getattr(test, 'intrinsic_functions_%s' % frontend)
    vmin, vmax, vabs, vexp, vsqrt, vlog = function(2., 4.)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vexp == np.exp(6.) and vsqrt == np.sqrt(6.) and vlog == np.log(6.)
