import ctypes as ct
import os
from pathlib import Path
import numpy as np
import pytest

from conftest import jit_compile, clean_test, available_frontends
from loki import Subroutine, FortranMaxTransformation, execute, delete
from loki.build import Builder, Obj, Lib
from loki.build.max_compiler import (compile_all, compile_maxj, compile_max, generate_max,
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


@pytest.fixture(scope='module', name='simulator')
def fixture_simulator():

    class MaxCompilerSim:

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


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(here):
    return Builder(source_dirs=here, include_dirs=get_max_includes(), build_dir=here/'build')


def max_transpile(routine, path, builder, frontend, objects=None, wrap=None):
    builder.clean()

    # Create transformation object and apply
    f2max = FortranMaxTransformation()
    f2max.apply(routine, path=path)

    # Generate simulation object file from maxj kernel
    compile_maxj(src=f2max.maxj_kernel_path.parent, build_dir=builder.build_dir)
    max_path = generate_max(manager=f2max.maxj_manager_path.stem, maxj_src=f2max.maxj_src,
                            max_filename=routine.name, build_dir=builder.build_dir,
                            package=routine.name)
    max_obj = compile_max(max_path, '%s_max.o' % max_path.stem, build_dir=builder.build_dir)
    max_include = max_obj.parent/('%s_MAX5C_DFE_SIM/results' % routine.name)

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


def test_max_passthrough(simulator, here):
    """
    A simple test streaming data to the DFE and back to CPU.
    """
    build_dir = here/'build'
    compile_all(c_src=here/'passthrough', maxj_src=here/'passthrough', build_dir=build_dir,
                target='PassThrough', manager='PassThroughMAX5CManager', package='passthrough')
    simulator.run(build_dir/'PassThrough')


def test_max_passthrough_ctypes(simulator, here):
    """
    A simple test streaming data to the DFE and back to CPU, called via ctypes
    """
    # First, build shared library
    build_dir = here/'build'
    compile_all(c_src=here/'passthrough', maxj_src=here/'passthrough', build_dir=build_dir,
                target='libPassThrough.so', manager='PassThroughMAX5CManager', package='passthrough')
    lib = ct.CDLL(build_dir/'libPassThrough.so')

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


@pytest.mark.parametrize('frontend', available_frontends())
def test_max_routine_axpy_scalar(here, builder, simulator, frontend):

    fcode = """
subroutine routine_axpy_scalar(a, x, y)
  ! A simple standard routine that computes x = a * x + y for
  ! scalar arguments
  use iso_fortran_env, only: real64
  implicit none
  real(kind=real64), intent(in) :: a, y
  real(kind=real64), intent(inout) :: x

  x = a * x + y
end subroutine routine_axpy_scalar
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('routine_axpy_scalar_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_axpy_scalar')

    # Test the reference solution
    a = -3.
    x = np.zeros(shape=(1,), order='F') + 2.
    y = np.zeros(shape=(1,), order='F') + 10.
    function(a=a, x=x, y=y)
    assert np.all(a * 2. + y == x)

    simulator.restart()
    # TODO: For some reason we have to generate and run the kernel twice in the same instance of
    # the simulator to actually get any results other than 0. Probably doing something wrong with
    # the Maxeler language...
    for _ in range(2):
        # Generate the transpiled kernel
        max_kernel = max_transpile(routine, here, builder, frontend)

        # Test the transpiled kernel
        a = -3.
        x = np.zeros(shape=(1,), order='F') + 2.
        y = np.zeros(shape=(1,), order='F') + 10.
        max_kernel.routine_axpy_scalar_c_fc_mod.routine_axpy_scalar_c_fc(ticks=1, a=a, x=x, y=y)
        print(x)
    simulator.stop()
#    simulator.call(max_kernel.routine_axpy_scalar_fmax_mod.routine_axpy_scalar_fmax, a, x, y)
    assert np.all(a * 2. + y == x)

    clean_test(filepath)
    delete(here/routine.name, force=True)  # Delete MaxJ sources


@pytest.mark.parametrize('frontend', available_frontends())
def test_max_routine_copy_scalar(here, builder, simulator, frontend):

    fcode = """
subroutine routine_copy_scalar(x, y)
  ! A simple routine that copies the value of x to y
  use iso_fortran_env, only: real64
  implicit none
  ! integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=real64), intent(in) :: x
  real(kind=real64), intent(out) :: y

  y = x
end subroutine routine_copy_scalar
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('routine_copy_scalar_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_copy_scalar')

    # Test the reference solution
    x = np.zeros(1) + 2.
    y = function(x=x)
    assert np.all(y == x)

    simulator.restart()
    # TODO: For some reason we have to generate and run the kernel twice in the same instance of
    # the simulator to actually get any results other than 0. Probably doing something wrong with
    # the Maxeler language...
    for _ in range(2):
        # Generate the transpiled kernel
        max_kernel = max_transpile(routine, here, builder, frontend)

        # Test the transpiled kernel
        x = np.zeros(1) + 2.
        y = max_kernel.routine_copy_scalar_c_fc_mod.routine_copy_scalar_c_fc(ticks=1, x=x)
        print(y)
    simulator.stop()
    assert np.all(y == x)

    clean_test(filepath)
    delete(here/routine.name, force=True)  # Delete MaxJ sources


@pytest.mark.parametrize('frontend', available_frontends())
def test_max_routine_fixed_loop(here, builder, simulator, frontend):

    fcode = """
subroutine routine_fixed_loop(scalar, vector, vector_out, tensor, tensor_out)
  use iso_fortran_env, only: real64
  implicit none
  integer :: n=6, m=4
  real(kind=real64), intent(in) :: scalar
  real(kind=real64), intent(in) :: tensor(6, 4), vector(6)
  real(kind=real64), intent(out) :: tensor_out(4, 6), vector_out(6)
  integer :: i, j

  ! For testing, the operation is:
  do i=1, n
     vector_out(i) = vector(i) + tensor(i, 1) + 1.0
  end do

  do j=1, n
     do i=1, m
        tensor_out(i, j) = tensor(j, i)
     end do
  end do
end subroutine routine_fixed_loop
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('routine_fixed_loop_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_fixed_loop')

    # Test the reference solution
    n, m = 6, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.array([list(range(i, i+m)) for i in range(n)], order='F', dtype=np.float64)
    tensor_out = np.zeros(shape=(m, n), order='F')
    ref_vector = vector + np.array(list(range(n)), dtype=np.float64) + 1.
    ref_tensor = np.transpose(tensor)
    function(scalar=scalar, vector=vector, vector_out=vector, tensor=tensor, tensor_out=tensor_out)
    assert np.all(vector == ref_vector)
    assert np.all(tensor_out == ref_tensor)

    # Generate the transpiled kernel
    max_kernel = max_transpile(routine, here, builder, frontend)

    # Test the transpiled kernel
    n, m = 6, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    tensor = np.array([list(range(i, i+m)) for i in range(n)], order='F', dtype=np.float64)
    tensor_out = np.zeros(shape=(m, n), order='F')
    function = max_kernel.routine_fixed_loop_c_fc_mod.routine_fixed_loop_c_fc
    simulator.call(function, ticks=1, scalar=scalar, vector=vector, vector_size=n * 8,
                   vector_out=vector, vector_out_size=n * 8, tensor=tensor, tensor_size=n * m * 8,
                   tensor_out=tensor_out, tensor_out_size=n * m * 8)
    assert np.all(vector == ref_vector)
    assert np.all(tensor_out == ref_tensor)

    clean_test(filepath)
    delete(here/routine.name, force=True)  # Delete MaxJ sources


@pytest.mark.parametrize('frontend', available_frontends())
def test_max_routine_copy_stream(here, builder, simulator, frontend):

    fcode = """
subroutine routine_copy_stream(length, scalar, vector_in, vector_out)
  implicit none
  ! A simple standard looking routine to test argument declarations
  ! and generator toolchain
  integer, intent(in) :: length, scalar, vector_in(length)
  integer, intent(out) :: vector_out(length)
  integer :: i

  !$loki dataflow
  do i=1, length
    vector_out(i) = vector_in(i) + scalar
  end do
end subroutine routine_copy_stream
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('routine_copy_stream_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_copy_stream')

    # Test the reference solution
    length = 32
    scalar = 7
    vector_in = np.array(range(length), order='F', dtype=np.intc)
    vector_out = np.zeros(length, order='F', dtype=np.intc)
    function(length=length, scalar=scalar, vector_in=vector_in, vector_out=vector_out)
    assert np.all(vector_out == np.array(range(length)) + scalar)

    # Generate the transpiled kernel
    max_kernel = max_transpile(routine, here, builder, frontend)

    vec_in = np.array(range(length), order='F', dtype=np.intc)
    vec_out = np.zeros(length, order='F', dtype=np.intc)
    function = max_kernel.routine_copy_stream_c_fc_mod.routine_copy_stream_c_fc
    simulator.call(function, ticks=length, length=length, scalar=scalar, vector_in=vec_in,
                   vector_in_size=length * 4, vector_out=vec_out, vector_out_size=length * 4)
    assert np.all(vec_out == np.array(range(length)) + scalar)

    clean_test(filepath)
    delete(here/routine.name, force=True)  # Delete MaxJ sources


@pytest.mark.parametrize('frontend', available_frontends())
def test_max_routine_moving_average(here, builder, simulator, frontend):

    fcode = """
subroutine routine_moving_average(length, data_in, data_out)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: length
  real(kind=real64), intent(in) :: data_in(length)
  real(kind=real64), intent(out) :: data_out(length)
  integer :: i
  real(kind=real64) :: prev, next, divisor

  !$loki dataflow
  do i=1, length
    divisor = 1.0
    if (i > 1) then
      prev = data_in(i-1)
      divisor = divisor + 1.0
    else
      prev = 0
    end if
    if (i < length) then
      next = data_in(i+1)
      divisor = divisor + 1.0
    else
      next = 0
    end if
    data_out(i) = (prev + data_in(i) + next) / divisor
  end do
end subroutine routine_moving_average
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('routine_moving_average_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_moving_average')

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
    function(n, data_in, data_out)
    assert np.all(data_out == expected)

    # Generate and test the transpiled kernel
    max_kernel = max_transpile(routine, here, builder, frontend)

    data_out = np.zeros(shape=(n,), order='F')
    function = max_kernel.routine_moving_average_c_fc_mod.routine_moving_average_c_fc
    simulator.call(function, ticks=n, length=n, data_in=data_in, data_in_size=n * 8,
                   data_out_size=n * 8, data_out=data_out)
    assert np.all(data_out == expected)

    clean_test(filepath)
    delete(here/routine.name, force=True)  # Delete MaxJ sources


@pytest.mark.parametrize('frontend', available_frontends())
def test_max_routine_laplace(here, builder, simulator, frontend):
    fcode = """
subroutine routine_laplace(h, data_in, data_out)
  use iso_fortran_env, only: real64
  implicit none
  integer :: m = 32, n = 32
!  real(kind=real64), intent(in) :: h, rhs(m*n), data_in(m*n)
!  real(kind=real64), intent(out) :: data_out(m*n)
  real(kind=real64), intent(in) :: h, data_in(32*32)
  real(kind=real64), intent(out) :: data_out(32*32)
  integer :: i, i_mod_n
  real(kind=real64) :: north, south, east, west

  !$loki dataflow
  do i=1, m*n
    i_mod_n = mod(i, n)
    if (i_mod_n /= 0) then
        north = data_in(i+1)
    else
        north = 0
    endif
    if (i_mod_n /= 1) then
        south = data_in(i-1)
    else
        south = 0
    end if
    if (i > n) then
        west = data_in(i-n)
    else
        west = 0
    end if
    if (i <= (m-1)*n) then
        east = data_in(i+n)
    else
        east = 0
    end if
    data_out(i) = (north + south + east + west - 4 * data_in(i)) / (h * h)
  end do
end subroutine routine_laplace
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('routine_laplace_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_laplace')

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
    function(h, data_in, data_out)
    assert np.all(abs(data_out - expected) < 1e-12)

    # Generate the transpiled kernel
    max_kernel = max_transpile(routine, here, builder, frontend)

    data_out = np.zeros(shape=(length,), order='F')
    function = max_kernel.routine_laplace_c_fc_mod.routine_laplace_c_fc
    simulator.call(function, ticks=length, h=h, data_in=data_in, data_in_size=length * 8,
                   data_out=data_out, data_out_size=length * 8)
    assert np.all(abs(data_out - expected) < 1e-12)

    clean_test(filepath)
    delete(here/routine.name, force=True)  # Delete MaxJ sources
