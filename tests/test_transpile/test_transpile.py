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
    A simple test routine to test C transpilation of loops
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

    # Test the transpiled C kernel
    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    function = c_kernel.transpile_simple_loops_fc_mod.transpile_simple_loops_fc
    function(n, m, scalar, vector, tensor)
    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])


def test_transpile_arguments(refpath, reference, builder):
    """
    A test the correct exchange of arguments with varying intents
    """

    # Test the reference solution
    n = 3
    array = np.zeros(shape=(n,), order='F')
    array_io = np.zeros(shape=(n,), order='F') + 3.
    # To do scalar inout we allocate data in single-element arrays
    a_io = np.zeros(shape=(1,), order='F', dtype=np.int32) + 1
    b_io = np.zeros(shape=(1,), order='F', dtype=np.float32) + 2.
    c_io = np.zeros(shape=(1,), order='F', dtype=np.float64) + 3.

    a, b, c = reference.transpile_arguments(n, array, array_io, a_io, b_io, c_io)
    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 2 and np.isclose(b, 3.2) and np.isclose(c, 4.1)

    # Generate the C kernel
    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_arguments'], refpath, builder)

    array = np.zeros(shape=(n,), order='F')
    array_io = np.zeros(shape=(n,), order='F') + 3.
    a_io = np.zeros(shape=(1,), order='F', dtype=np.int32) + 1
    b_io = np.zeros(shape=(1,), order='F', dtype=np.float32) + 2.
    c_io = np.zeros(shape=(1,), order='F', dtype=np.float64) + 3.

    a, b, c = c_kernel.transpile_arguments_fc_mod.transpile_arguments_fc(n, array, array_io, a_io, b_io, c_io)
    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 2 and np.isclose(b, 3.2) and np.isclose(c, 4.1)


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
    function = c_kernel.transpile_derived_type_fc_mod.transpile_derived_type_fc
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.


def test_transpile_module_variables(refpath, reference, builder):
    """
    Tests the use of imported module variables (via getter routines in C)
    """
    reference.transpile_type.param1 = 2
    reference.transpile_type.param2 = 4.
    reference.transpile_type.param3 = 3.
    a, b, c = reference.transpile_module_variables()
    assert a == 3 and b == 5. and c == 4.

    # Translate the header module to expose parameters
    typepath = refpath.parent/'transpile_type.f90'
    typemod = SourceFile.from_file(typepath)['transpile_type']
    FortranCTransformation().apply(routine=typemod, path=refpath.parent)

    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_module_variables'], refpath, builder,
                           objects=['transpile_type.f90', 'transpile_type_fc.f90'],
                           wrap=['transpile_type.f90'], header_modules=[typemod])

    c_kernel.transpile_type.param1 = 2
    c_kernel.transpile_type.param2 = 4.
    c_kernel.transpile_type.param3 = 3.
    a, b, c = c_kernel.transpile_module_variables_fc_mod.transpile_module_variables_fc()
    assert a == 3 and b == 5. and c == 4.


def test_transpile_vectorization(refpath, reference, builder):
    """
    Tests vector-notation conversion and local multi-dimensional arrays.
    """

    # Test the reference solution
    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')

    reference.transpile_vectorization(n, m, scalar, v1, v2)
    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)

    # Generate the C kernel
    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_vectorization'], refpath, builder)

    # Test the trnapiled C kernel
    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')
    function = c_kernel.transpile_vectorization_fc_mod.transpile_vectorization_fc
    function(n, m, scalar, v1, v2)
    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)


def test_transpile_intrinsics(refpath, reference, builder):
    """
    A simple test routine to test supported intrinsic functions
    """

    # Test the reference solution
    v1, v2, v3, v4 = 2., 4., 1., 5.
    vmin, vmax, vabs, vmin_nested, vmax_nested = reference.transpile_intrinsics(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.

    # Generate the C kernel
    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_intrinsics'], refpath, builder)

    results = c_kernel.transpile_intrinsics_fc_mod.transpile_intrinsics_fc(v1, v2, v3, v4)
    vmin, vmax, vabs, vmin_nested, vmax_nested = results
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.


def test_transpile_loop_indices(refpath, reference, builder):
    """
    Test to ensure loop indexing translates correctly
    """

    # Test the reference solution
    n = 6
    cidx, fidx = 3, 4
    mask1 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask3 = np.zeros(shape=(n,), order='F', dtype=np.float64)

    reference.transpile_loop_indices(n=n, idx=fidx, mask1=mask1, mask2=mask2, mask3=mask3)
    assert np.all(mask1[:cidx-1] == 1)
    assert mask1[cidx] == 2
    assert np.all(mask1[cidx+1:] == 0)
    assert np.all(mask2 == np.arange(n, dtype=np.int32) + 1)
    assert np.all(mask3[:-1] == 0.)
    assert mask3[-1] == 3.

    # Generate the C kernel
    source = SourceFile.from_file(refpath, frontend=OMNI, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_loop_indices'], refpath, builder)

    mask1 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask3 = np.zeros(shape=(n,), order='F', dtype=np.float64)
    function = c_kernel.transpile_loop_indices_fc_mod.transpile_loop_indices_fc
    function(n=n, idx=fidx, mask1=mask1, mask2=mask2, mask3=mask3)
    assert np.all(mask1[:cidx-1] == 1)
    assert mask1[cidx] == 2
    assert np.all(mask1[cidx+1:] == 0)
    assert np.all(mask2 == np.arange(n, dtype=np.int32) + 1)
    assert np.all(mask3[:-1] == 0.)
    assert mask3[-1] == 3.
