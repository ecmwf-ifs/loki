import pytest
import ctypes as ct
import numpy as np
import os
from pathlib import Path

from loki import SourceFile, OMNI, FP, FortranMaxTransformation
from loki.build import Builder, Obj, Lib, execute
from loki.build.max_compiler import (compile, compile_maxj, compile_max, generate_max,
                                     get_max_includes, get_max_libs, get_max_libdirs)


def check_maxeler():
    """
    Check if Maxeler environment variables are specified.
    """
    maxeler_vars = {'MAXCOMPILERDIR', 'MAXELEROSDIR'}
    return maxeler_vars <= os.environ.keys()


# Skip tests in this module if Maxeler environment not present
pytestmark = pytest.mark.skipif(not check_maxeler(),
                                reason='Maxeler compiler not installed')


@pytest.fixture(scope='module')
def simulator():

    class MaxCompilerSim(object):

        def __init__(self):
            name = '%s_pytest' % os.getlogin()
            self.base_cmd = ['maxcompilersim', '-n', name]
            os.environ['SLIC_CONF'] = 'use_simulation=%s' % name
            self.maxeleros = ct.CDLL(os.environ['MAXELEROSDIR'] + '/lib/libmaxeleros.so')

        def __del__(self):
            del self.maxeleros
            del os.environ['SLIC_CONF']

        def restart(self):
            cmd = self.base_cmd + ['-c', 'MAX5C', 'restart']
            execute(cmd)

        def stop(self):
            cmd = self.base_cmd + ['stop']
            execute(cmd)

        def run(self, target, *args):
            cmd = [str(target)]
            if args is not None:
                cmd += [str(a) for a in args]
            self.restart()
            env = os.environ.copy()
            env['LD_PRELOAD'] = '%s/lib/libmaxeleros.so:%s' % (os.environ['MAXELEROSDIR'],
                                                               os.environ.get('LD_PRELOAD', ''))
            execute(cmd, env=env)
            self.stop()

        def call(self, fn, *args, **kwargs):
            self.restart()
            ret = fn(*args, **kwargs)
            self.stop()
            return ret

    print('simulator()')
    return MaxCompilerSim()


@pytest.fixture(scope='module')
def build_dir():
    return Path(__file__).parent / 'build'


@pytest.fixture(scope='module')
def refpath():
    return Path(__file__).parent / 'maxeler.f90'


@pytest.fixture(scope='module')
def builder(refpath):
    path = refpath.parent
    return Builder(source_dirs=path, include_dirs=get_max_includes(), build_dir=path/'build')


@pytest.fixture(scope='module')
def reference(refpath, builder):
    """
    Compile and load the reference solution
    """
    builder.clean()

    sources = ['maxeler.f90']
    objects = [Obj(source_path=s) for s in sources]
    lib = Lib(name='max_ref', objs=objects, shared=False)
    lib.build(builder=builder)
    return lib.wrap(modname='max_ref', sources=sources, builder=builder)


def max_transpile(routine, refpath, builder, frontend, objects=None, wrap=None):
    builder.clean()

    # Create transformation object and apply
    f2max = FortranMaxTransformation()
    f2max.apply(routine=routine, path=refpath.parent)

    # Generate simulation object file from maxj kernel
    compile_maxj(src=f2max.maxj_kernel_path.parent, build_dir=builder.build_dir)
    max_path = generate_max(manager=f2max.maxj_manager_path.stem, maxj_src=f2max.maxj_src,
                            max_filename=routine.name, build_dir=builder.build_dir,
                            package=routine.name)
    max_obj = compile_max(max_path, '%s_max.o' % max_path.stem, build_dir=builder.build_dir)
    max_include = max_obj.parent / ('%s_MAX5C_DFE_SIM/results' % routine.name)

    # Build and wrap the cross-compiled library
    objects = (objects or []) + [Obj(source_path=f2max.c_path), Obj(source_path=f2max.wrapperpath)]
    lib = Lib(name='fmax_%s_%s' % (routine.name, frontend), objs=objects, shared=False)
    lib.build(builder=builder, include_dirs=[max_include], external_objs=[max_obj])

    return lib.wrap(modname='mod_%s_%s' % (routine.name, frontend), builder=builder,
                    sources=(wrap or []) + [f2max.wrapperpath.name],
                    libs=get_max_libs(), lib_dirs=get_max_libdirs())


def test_max_simulator(simulator):
    """
    Starts and stops the Maxeler Simulator.
    """
    simulator.restart()
    simulator.stop()
    assert True


def test_max_passthrough(simulator, build_dir, refpath):
    """
    A simple test streaming data to the DFE and back to CPU.
    """
    compile(c_src=refpath.parent / 'passthrough', maxj_src=refpath.parent / 'passthrough',
            build_dir=build_dir, target='PassThrough', manager='PassThroughMAX5CManager',
            package='passthrough')
    simulator.run(build_dir / 'PassThrough')


def test_max_passthrough_ctypes(simulator, build_dir, refpath):
    """
    A simple test streaming data to the DFE and back to CPU, called via ctypes
    """
    # First, build shared library
    compile(c_src=refpath.parent / 'passthrough', maxj_src=refpath.parent / 'passthrough',
            build_dir=build_dir, target='libPassThrough.so', manager='PassThroughMAX5CManager',
            package='passthrough')
    lib = ct.CDLL(build_dir / 'libPassThrough.so')

    # Extract function interfaces for CPU and DFE version
    func_cpu = lib.PassThroughCPU
    func_cpu.restype = None
    func_cpu.argtypes = [ct.c_int, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint32)]

    func_dfe = lib.passthrough
    func_dfe.restype = None
    func_dfe.argtypes = [ct.c_uint64, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]

    # Create input/output data structures
    size = 1024
    data_in = [i+1 for i in range(size)]
    data_out = size * [0]

    array_type = ct.c_uint32 * size
    size_bytes = ct.c_size_t(ct.sizeof(ct.c_uint32) * size)
    data_in = array_type(*data_in)
    expected_out = array_type(*data_out)
    data_out = array_type(*expected_out)

    # Run CPU function
    func_cpu(ct.c_int(size), data_in, expected_out)
    assert list(data_in) == list(expected_out)

    # Run DFE function
    simulator.call(func_dfe, ct.c_uint64(size), data_in, size_bytes, data_out, size_bytes)
    assert list(data_in) == list(data_out)


@pytest.mark.parametrize('frontend', [OMNI, FP])
def test_max_routine_axpy(refpath, reference, builder, simulator, frontend):

    # Test the reference solution
    a = -3.
    x = np.zeros(shape=(1,), order='F') + 2.
    y = np.zeros(shape=(1,), order='F') + 10.
    reference.routine_axpy(a=a, x=x, y=y)
    assert np.all(a * 2. + y == x)

    simulator.restart()
    # TODO: For some reason we have to generate and run the kernel twice in the same instance of
    # the simulator to actually get any results other than 0. Probably doing something wrong with
    # the Maxeler language...
    for _ in range(2):
        # Generate the transpiled kernel
        source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
        max_kernel = max_transpile(source['routine_axpy'], refpath, builder, frontend)

        # Test the transpiled kernel
        a = -3.
        x = np.zeros(shape=(1,), order='F') + 2.
        y = np.zeros(shape=(1,), order='F') + 10.
        max_kernel.routine_axpy_c_fmax_mod.routine_axpy_c_fmax(ticks=1, a=a, x=x, x_in=x, y=y)
        print(x)
    simulator.stop()
#    simulator.call(max_kernel.routine_axpy_fmax_mod.routine_axpy_fmax, a, x, y)
    assert np.all(a * 2. + y == x)


@pytest.mark.parametrize('frontend', [OMNI, FP])
def test_max_routine_copy(refpath, reference, builder, simulator, frontend):

    # Test the reference solution
    x = np.zeros(1) + 2.
    y = reference.routine_copy(x=x)
    assert np.all(y == x)

    simulator.restart()
    # TODO: For some reason we have to generate and run the kernel twice in the same instance of
    # the simulator to actually get any results other than 0. Probably doing something wrong with
    # the Maxeler language...
    for _ in range(2):
        # Generate the transpiled kernel
        source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
        max_kernel = max_transpile(source['routine_copy'], refpath, builder, frontend)

        # Test the transpiled kernel
        x = np.zeros(1) + 2.
        y = max_kernel.routine_copy_c_fmax_mod.routine_copy_c_fmax(ticks=1, x=x)
        print(y)
    simulator.stop()
    assert np.all(y == x)


@pytest.mark.parametrize('frontend', [OMNI, FP])
def test_max_routine_fixed_loop(refpath, reference, builder, simulator, frontend):

    # Test the reference solution
    n, m = 6, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    # tensor_out = np.zeros(shape=(n, m), order='F')
    reference.routine_fixed_loop(scalar, vector, vector, tensor)
    assert np.all(vector == 8.)
    # ref_tensor = (np.array([range(10, 10 * (m+1), 10)] * n)
    #               + np.transpose(np.array([range(1, n+1)] * m)))
    # assert np.all(tensor_out == tensor)

    # Generate the transpiled kernel
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    max_kernel = max_transpile(source['routine_fixed_loop'], refpath, builder, frontend)

    # Test the transpiled kernel
    n, m = 6, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    # tensor_out = np.zeros(shape=(n, m), order='F')
    function = max_kernel.routine_fixed_loop_c_fmax_mod.routine_fixed_loop_c_fmax
    simulator.call(function, ticks=1, scalar=scalar, vector=vector, vector_size=n * 8,
                   vector_out=vector, vector_out_size=n * 8, tensor=tensor, tensor_size=n * m * 8)
    assert np.all(vector == 8.)
    # assert np.all(tensor_out == tensor)
    # assert np.all(tensor == [[11., 21., 31., 41.],
    #                          [12., 22., 32., 42.],
    #                          [13., 23., 33., 43.]])


@pytest.mark.parametrize('frontend', [OMNI, FP])
def test_max_routine_shift(refpath, reference, builder, simulator, frontend):

    # Test the reference solution
    length = 32
    scalar = 7
    vector_in = np.array(range(length), order='F', dtype=np.intc)
    vector_out = np.zeros(length, order='F', dtype=np.intc)
    reference.routine_shift(length, scalar, vector_in, vector_out)
    assert np.all(vector_out == np.array(range(length)) + scalar)

    # Generate the transpiled kernel
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    max_kernel = max_transpile(source['routine_shift'], refpath, builder, frontend)

    vec_in = np.array(range(length), order='F', dtype=np.intc)
    vec_out = np.zeros(length, order='F', dtype=np.intc)
    function = max_kernel.routine_shift_c_fmax_mod.routine_shift_c_fmax
    simulator.call(function, ticks=length, length=length, scalar=scalar, vector_in=vec_in,
                   vector_in_size=length * 4, vector_out=vec_out, vector_out_size=length * 4)
    assert np.all(vec_out == np.array(range(length)) + scalar)


@pytest.mark.parametrize('frontend', [OMNI, FP])
def test_max_routine_moving_average(refpath, reference, builder, simulator, frontend):

    # Create random input data
    n = 32
    data_in = np.array(np.random.rand(n), order='F')

    # Compute reference solution
    expected = np.zeros(shape=(n,), order='F')
    expected[0] = (data_in[0] + data_in[1]) / 2.
    expected[1:-1] = (data_in[:-2] + data_in[1:-1] + data_in[2:]) / 3.
    expected[-1] = (data_in[-2] + data_in[-1]) / 2.

    # Test the Fortran kernel
    data_out = np.zeros(shape=(n,), order='F')
    reference.routine_moving_average(n, data_in, data_out)
    assert np.all(data_out == expected)

    # Generate the transpiled kernel
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    max_kernel = max_transpile(source['routine_moving_average'], refpath, builder, frontend)

    data_out = np.zeros(shape=(n,), order='F')
    function = max_kernel.routine_moving_average_c_fmax_mod.routine_moving_average_c_fmax
    simulator.call(function, ticks=n, length=n, data_in=data_in, data_in_size=n * 8,
                   data_out_size=n * 8, data_out=data_out)
    assert np.all(data_out == expected)


@pytest.mark.parametrize('frontend', [OMNI, FP])
def test_max_routine_laplace(refpath, reference, builder, simulator, frontend):

    # Create random input data
    m, n = 32, 32
    h, length = 1./m, m * n
    data_in = np.array(np.random.rand(length), order='F')

    # Compute reference solution
    expected = -4. * data_in

    expected[0:n] += data_in[n:2*n]
    expected[1:n] += data_in[0:n-1]
    expected[0:n-1] += data_in[1:n]

    for i in range(1, m-1):
        idx = i*n
        expected[idx+0:idx+n] += data_in[idx-n:idx] + data_in[idx+n:idx+2*n]
        expected[idx+1:idx+n] += data_in[idx+0:idx+n-1]
        expected[idx+0:idx+n-1] += data_in[idx+1:idx+n]

    idx = (m-1)*n
    expected[idx+0:idx+n] += data_in[idx-n:idx]
    expected[idx+1:idx+n] += data_in[idx+0:idx+n-1]
    expected[idx+0:idx+n-1] += data_in[idx+1:idx+n]
    expected /= h*h

    # Test the Fortran kernel
    data_out = np.zeros(shape=(length,), order='F')
    reference.routine_laplace(h, data_in, data_out)
    assert np.all(abs(data_out - expected) < 1e-12)

    # Generate the transpiled kernel
    source = SourceFile.from_file(refpath, frontend=frontend, xmods=[refpath.parent])
    max_kernel = max_transpile(source['routine_laplace'], refpath, builder, frontend)

    data_out = np.zeros(shape=(length,), order='F')
    function = max_kernel.routine_laplace_c_fmax_mod.routine_laplace_c_fmax
    simulator.call(function, ticks=length, h=h, data_in=data_in, data_in_size=length * 8,
                   data_out=data_out, data_out_size=length * 8)
    assert np.all(abs(data_out - expected) < 1e-12)
