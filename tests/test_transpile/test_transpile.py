import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, Module, fgen, OFP, OMNI
from conftest import generate_identity


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'transpile.f90'


@pytest.fixture(scope='module')
def reference(refpath):
    """
    Compile and load the reference solution
    """
    clean(filename=refpath)  # Delete parser cache
    return compile_and_load(refpath, cwd=str(refpath.parent))


def c_transpile(routine, refpath):
    """
    Generate the C-wrapper with ISO-C bindings
    """
    wrapper = routine.generate_iso_c_wrapper(suffix='_iso_c')
    wrapperpath = (refpath.parent/wrapper.name).with_suffix('.f90')
    SourceFile.to_file(source=fgen(wrapper), path=wrapperpath)
    return compile_and_load(wrapperpath, cwd=str(refpath.parent))


def test_transpile_simple_loops(refpath, reference):
    """
    A simple standard looking routine to test C transpilation
    """
    frontend = OMNI

    # Test the reference solution
    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    reference.transpile_simple_loops(n, m, scalar, vector, tensor)
    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])

    # Generate the C kernel
    routine = SourceFile.from_file(refpath, frontend=frontend).routines[0]
    c_kernel = c_transpile(routine, refpath)

    # Test the trnapiled C kernel
    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    c_kernel.transpile_simple_loops_iso_c(n, m, scalar, vector, tensor)
    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])
