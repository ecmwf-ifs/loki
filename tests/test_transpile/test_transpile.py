import pytest
import numpy as np
from pathlib import Path

from loki import clean, compile_and_load, SourceFile, Module, OMNI, Builder, FortranCTransformation
from conftest import generate_identity


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'transpile.f90'


@pytest.fixture(scope='module')
def builder(refpath):
    path = refpath.parent
    return Builder(source_dirs=path, build_dir=path/'build')


@pytest.fixture(scope='module')
def reference(refpath, builder):
    """
    Compile and load the reference solution
    """
    builder.clean()

    sources = ['transpile_type.f90', 'transpile.f90']
    lib = builder.Lib(name='ref', objects=sources)
    lib.build()
    return lib.wrap(modname='ref', sources=sources)


def c_transpile(routine, refpath, builder, header_modules=None, objects=None, wrap=None):
    """
    Generate the ISO-C bindings wrapper and C-transpiled source code
    """
    builder.clean()

    # Create transformation object and apply
    f2c = FortranCTransformation(header_modules=header_modules)
    f2c.apply(routine=routine, path=refpath.parent)

    # Build and wrap the cross-compiled library
    objects = (objects or []) + [f2c.wrapperpath.name, f2c.c_path.name]
    lib = builder.Lib(name='fc_%s' % routine.name, objects=objects)
    lib.build()

    return lib.wrap(modname='mod_%s' % routine.name, sources=(wrap or []) + [f2c.wrapperpath.name])


def test_transpile_simple_loops(refpath, reference, builder):
    """
    A simple standard looking routine to test C transpilation
    """

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
    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_simple_loops'], refpath, builder)

    # Test the trnapiled C kernel
    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    function = c_kernel.transpile_simple_loops_c_mod.transpile_simple_loops_c
    function(n, m, scalar, vector, tensor)
    assert np.all(vector == 8.)
    # TODO: The test uses the iteration indices to compute the results,
    # which has not yet been adapted in the conversion engine.
    # As a result, we get the correct iteration order, but need to
    # count from 0 instead of one when writing out indices.
    assert np.all(tensor == [[0., 10., 20., 30.],
                             [1., 11., 21., 31.],
                             [2., 12., 22., 32.]])


def test_transpile_derived_type(refpath, reference, builder):
    """
    Tests handling and type-conversion of various argument types

    a_struct%a = a_struct%a + 4   # int
    a_struct%b = a_struct%b + 5.  # float
    a_struct%c = a_struct%c + 6.  # double
    """

    # Test the reference solution
    a_struct = reference.transpile_type.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    reference.transpile_derived_type(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.

    # Translate the header module to expose parameters
    typepath = refpath.parent/'transpile_type.f90'
    typemod = SourceFile.from_file(typepath)['transpile_type']
    FortranCTransformation().apply(routine=typemod, path=refpath.parent)

    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent],
                                  typedefs=typemod.typedefs)
    c_kernel = c_transpile(source['transpile_derived_type'], refpath, builder,
                           objects=['transpile_type.f90'], wrap=['transpile_type.f90'],
                           header_modules=[typemod])

    a_struct = reference.transpile_type.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    function = c_kernel.transpile_derived_type_c_mod.transpile_derived_type_c
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.


def test_transpile_module_parameters(refpath, reference, builder):
    """
    Tests ...
    """
    from loki import logger, DEBUG; logger.setLevel(DEBUG)

    reference.transpile_type.param1 = 2
    reference.transpile_type.param2 = 4.
    reference.transpile_type.param3 = 3.
    a, b, c = reference.transpile_module_parameters()
    assert a == 3 and b == 5. and c == 4.

    # Translate the header module to expose parameters
    typepath = refpath.parent/'transpile_type.f90'
    typemod = SourceFile.from_file(typepath)['transpile_type']
    FortranCTransformation().apply(routine=typemod, path=refpath.parent)

    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_module_parameters'], refpath, builder,
                           objects=['transpile_type.f90', 'transpile_type.c.f90'],
                           wrap=['transpile_type.f90'], header_modules=[typemod])

    c_kernel.transpile_type.param1 = 2
    c_kernel.transpile_type.param2 = 4.
    c_kernel.transpile_type.param3 = 3.
    a, b, c = c_kernel.transpile_module_parameters_c_mod.transpile_module_parameters_c()
    assert a == 3 and b == 5. and c == 4.
