import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, fgen


@pytest.fixture
def reference():
    return Path(__file__).parent / 'expression.f90'


def generate_identity(reference, routinename, suffix):
    """
    Generate the "identity" of a single subroutine with a specific suffix.
    """
    testname = reference.parent/('%s_%s_%s.f90' % (reference.stem, routinename, suffix))
    source = SourceFile.from_file(reference)
    routine = [r for r in source.subroutines if r.name == routinename][0]
    routine.name += '_%s' % suffix
    source.write(source=fgen(routine), filename=testname)
    return compile_and_load(testname)


def test_expression(reference):
    """
    v5 = (v1 + v2) * (v3 - v4)
    v6 = (v1 ** v2) - (v3 / v4)
    """
    clean(filename=reference)  # Delete parser cache

    # Test the reference solution
    ref = compile_and_load(reference)
    v5, v6 = ref.simple_expr(2., 3., 10., 5.)
    assert v5 == 25. and v6 == 6.

    # Test the generated identity
    test = generate_identity(reference, 'simple_expr', suffix='test')
    v5, v6 = test.simple_expr_test(2., 3., 10., 5.)
    assert v5 == 25. and v6 == 6.


def test_intrinsic_functions(reference):
    """
    vmin = min(v1, v2)
    vmax = max(v1, v2)
    vabs = abs(v1 - v2)
    vexp = exp(v1 + v2)
    vsqrt = sqrt(v1 + v2)
    vlog = log(v1 + v2)
    """
    clean(filename=reference)  # Delete parser cache

    # Test the reference solution
    ref = compile_and_load(reference)
    vmin, vmax, vabs, vexp, vsqrt, vlog = ref.intrinsic_functions(2., 4.)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vexp == np.exp(6.) and vsqrt == np.sqrt(6.) and vlog == np.log(6.)

    # Test the generated identity
    test = generate_identity(reference, 'intrinsic_functions', suffix='test')
    vmin, vmax, vabs, vexp, vsqrt, vlog = test.intrinsic_functions_test(2., 4.)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vexp == np.exp(6.) and vsqrt == np.sqrt(6.) and vlog == np.log(6.)
