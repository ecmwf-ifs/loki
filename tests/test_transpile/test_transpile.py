from pathlib import Path
import pytest
import numpy as np

from loki import SourceFile, OFP, OMNI, FP, FortranCTransformation
from loki.build import Builder, Obj, Lib


@pytest.fixture(scope='module', name='refpath')
def fixture_refpath():
    return Path(__file__).parent / 'transpile.f90'


@pytest.fixture(scope='module', name='builder')
def fixture_builder(refpath):
    path = refpath.parent
    return Builder(source_dirs=path, build_dir=path/'build')


@pytest.fixture(scope='module', name='reference')
def fixture_reference(builder):
    """
    Compile and load the reference solution
    """
    builder.clean()

    sources = ['transpile_type.f90', 'transpile.f90']
    objects = [Obj(source_path=s) for s in sources]
    lib = Lib(name='ref', objs=objects, shared=False)
    lib.build(builder=builder)
    return lib.wrap(modname='ref', sources=sources, builder=builder)


def c_transpile(routine, refpath, builder, header_modules=None, objects=None, wrap=None):
    """
    Generate the ISO-C bindings wrapper and C-transpiled source code
    """
    builder.clean()

    # Create transformation object and apply
    f2c = FortranCTransformation(header_modules=header_modules)
    f2c.apply(source=routine, path=refpath.parent)

    # Build and wrap the cross-compiled library
    objects = (objects or []) + [Obj(source_path=f2c.wrapperpath),
                                 Obj(source_path=f2c.c_path)]
    lib = Lib(name='fc_%s' % routine.name, objs=objects, shared=False)
    lib.build(builder=builder)

    return lib.wrap(modname='mod_%s' % routine.name, builder=builder,
                    sources=(wrap or []) + [f2c.wrapperpath.name])


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_simple_loops(refpath, reference, builder, frontend):
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
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
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


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_arguments(refpath, reference, builder, frontend):
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
    assert a == 8 and np.isclose(b, 3.2) and np.isclose(c, 4.1)

    # Generate the C kernel
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_arguments'], refpath, builder)

    array = np.zeros(shape=(n,), order='F')
    array_io = np.zeros(shape=(n,), order='F') + 3.
    a_io = np.zeros(shape=(1,), order='F', dtype=np.int32) + 1
    b_io = np.zeros(shape=(1,), order='F', dtype=np.float32) + 2.
    c_io = np.zeros(shape=(1,), order='F', dtype=np.float64) + 3.

    a, b, c = c_kernel.transpile_arguments_fc_mod.transpile_arguments_fc(n, array, array_io,
                                                                         a_io, b_io, c_io)
    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 8 and np.isclose(b, 3.2) and np.isclose(c, 4.1)


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_derived_type(refpath, reference, builder, frontend):
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
    FortranCTransformation().apply(source=typemod, path=refpath.parent)

    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent],
                                  typedefs=typemod.typedefs)
    c_kernel = c_transpile(source['transpile_derived_type'], refpath, builder,
                           objects=[Obj(source_path='transpile_type.f90')],
                           wrap=['transpile_type.f90'], header_modules=[typemod])

    a_struct = reference.transpile_type.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    function = c_kernel.transpile_derived_type_fc_mod.transpile_derived_type_fc
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_associates(refpath, reference, builder, frontend):
    """
    Tests associate statements

    associate(a_struct_a=>a_struct%a, a_struct_b=>a_struct%b,&
    & a_struct_c=>a_struct%c)
    a_struct%a = a_struct_a + 4.
    a_struct_b = a_struct%b + 5.
    a_struct_c = a_struct_a + a_struct%b + a_struct_c
    end associate
    """

    # Test the reference solution
    a_struct = reference.transpile_type.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    reference.transpile_associates(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 24.

    # Translate the header module to expose parameters
    typepath = refpath.parent/'transpile_type.f90'
    typemod = SourceFile.from_file(typepath)['transpile_type']
    FortranCTransformation().apply(source=typemod, path=refpath.parent)

    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent],
                                  typedefs=typemod.typedefs)
    c_kernel = c_transpile(source['transpile_associates'], refpath, builder,
                           objects=[Obj(source_path='transpile_type.f90')],
                           wrap=['transpile_type.f90'], header_modules=[typemod])

    a_struct = reference.transpile_type.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    function = c_kernel.transpile_associates_fc_mod.transpile_associates_fc
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 24.


@pytest.mark.skip(reason='More thought needed on how to test structs-of-arrays')
def test_transpile_derived_type_array(refpath, reference, builder, frontend):
    """
    Tests handling of multi-dimensional arrays and pointers.

    a_struct%scalar = 3.
    a_struct%vector(i) = a_struct%scalar + 2.
    a_struct%matrix(j,i) = a_struct%vector(i) + 1.
    """

    # Test the reference solution
    a_struct = reference.transpile_type.array_struct()
    reference.transpile_type.alloc_arrays(a_struct)
    reference.transpile_derived_type_array(a_struct)
    assert a_struct.scalar == 3.
    assert (a_struct.vector == 5.).all()
    assert (a_struct.matrix == 6.).all()
    reference.transpile_type.free_arrays(a_struct)

    # Translate the header module to expose parameters
    typepath = refpath.parent/'transpile_type.f90'
    typemod = SourceFile.from_file(typepath)['transpile_type']
    FortranCTransformation().apply(source=typemod, path=refpath.parent)

    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent],
                                  typedefs=typemod.typedefs)
    c_kernel = c_transpile(source['transpile_derived_type_array'], refpath, builder,
                           objects=[Obj(source_path='transpile_type.f90')],
                           wrap=['transpile_type.f90'], header_modules=[typemod])

    a_struct = reference.transpile_type.array_struct()
    reference.transpile_type.alloc_arrays(a_struct)
    function = c_kernel.transpile_derived_type_array_fc_mod.transpile_derived_type_fc
    function(a_struct)
    assert a_struct.scalar == 3.
    assert (a_struct.vector == 5.).all()
    assert (a_struct.matrix == 6.).all()
    reference.transpile_type.free_arrays(a_struct)


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_module_variables(refpath, reference, builder, frontend):
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
    FortranCTransformation().apply(source=typemod, path=refpath.parent)

    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_module_variables'], refpath, builder,
                           objects=[Obj(source_path=refpath.parent / 'transpile_type.f90'),
                                    Obj(source_path=refpath.parent / 'transpile_type_fc.f90')],
                           wrap=['transpile_type.f90'], header_modules=[typemod])

    c_kernel.transpile_type.param1 = 2
    c_kernel.transpile_type.param2 = 4.
    c_kernel.transpile_type.param3 = 3.
    a, b, c = c_kernel.transpile_module_variables_fc_mod.transpile_module_variables_fc()
    assert a == 3 and b == 5. and c == 4.


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_vectorization(refpath, reference, builder, frontend):
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
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
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


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_intrinsics(refpath, reference, builder, frontend):
    """
    A simple test routine to test supported intrinsic functions
    """

    # Test the reference solution
    v1, v2, v3, v4 = 2., 4., 1., 5.
    vmin, vmax, vabs, vmin_nested, vmax_nested = reference.transpile_intrinsics(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.

    # Generate the C kernel
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_intrinsics'], refpath, builder)

    results = c_kernel.transpile_intrinsics_fc_mod.transpile_intrinsics_fc(v1, v2, v3, v4)
    vmin, vmax, vabs, vmin_nested, vmax_nested = results
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_loop_indices(refpath, reference, builder, frontend):
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
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
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


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_logical_statements(refpath, reference, builder, frontend):
    """
    A simple test routine to test logical statements
    """

    # Test the reference solution
    for v1 in range(2):
        for v2 in range(2):
            v_val = np.zeros(shape=(2,), order='F', dtype=np.int32)
            v_xor, v_xnor, v_nand, v_neqv = reference.transpile_logical_statements(v1, v2, v_val)
            assert v_xor == (v1 and not v2) or (not v1 and v2)
            assert v_xnor == (v1 and v2) or not (v1 or v2)
            assert v_nand == (not (v1 and v2))
            assert v_neqv == ((not (v1 and v2)) and (v1 or v2))
            assert v_val[0] and not v_val[1]

    # Generate the C kernel
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    c_kernel = c_transpile(source['transpile_logical_statements'], refpath, builder)
    function = c_kernel.transpile_logical_statements_fc_mod.transpile_logical_statements_fc

    for v1 in range(2):
        for v2 in range(2):
            v_val = np.zeros(shape=(2,), order='F', dtype=np.int32)
            v_xor, v_xnor, v_nand, v_neqv = function(v1, v2, v_val)
            assert v_xor == (v1 and not v2) or (not v1 and v2)
            assert v_xnor == (v1 and v2) or not (v1 or v2)
            assert v_nand == (not (v1 and v2))
            assert v_neqv == ((not (v1 and v2)) and (v1 or v2))
            assert v_val[0] and not v_val[1]
