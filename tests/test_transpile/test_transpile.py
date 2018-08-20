import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, Module, OMNI, Builder, FortranCTransformation
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
    Generate the ISO-C bindings wrapper and C-transpiled source code
    """
    path = refpath.parent
    builder = Builder(source_dirs=path, build_dir=path/'build')
    builder.clean()

    # Create transformation object and apply
    f2c = FortranCTransformation()
    f2c.apply(routine=routine, path=refpath.parent)

    # Build and wrap the cross-compiled library
    builder = Builder(source_dirs=path, build_dir=path/'build')
    lib = builder.Lib(name='fclib', objects=[f2c.wrapperpath.name, f2c.c_path.name])
    lib.build()

    return lib.wrap(modname='fcmod', sources=[f2c.wrapperpath.name])


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
    # TODO: The test uses the iteration indices to compute the results,
    # which has not yet been adapted in the conversion engine.
    # As a result, we get the correct iteration order, but need to
    # count from 0 instead of one when writing out indices.
    assert np.all(tensor == [[0., 10., 20., 30.],
                             [1., 11., 21., 31.],
                             [2., 12., 22., 32.]])
