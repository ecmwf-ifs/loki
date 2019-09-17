import pytest
import os
from pathlib import Path
import ctypes as ct

from loki.build.tools import execute
from loki.build.max_compiler import compile


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
            os.environ.update({'SLIC_CONF': 'use_simulation=%s' % name,
                               'LD_PRELOAD': ('%s/lib/libmaxeleros.so' %
                                              os.environ['MAXELEROSDIR'])})
            self.maxeleros = ct.CDLL(os.environ['MAXELEROSDIR'] + '/lib/libmaxeleros.so')

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
            execute(cmd)
            self.stop()

        def call(self, fn, *args):
            self.restart()
            fn(*args)
            self.stop()

    return MaxCompilerSim()


@pytest.fixture(scope='module')
def build_dir():
    return Path(__file__).parent / 'build'


@pytest.fixture(scope='module')
def src_root():
    return Path(__file__).parent


def test_detect_environment():
    """
    Skipped if the Maxeler environment was not detected, succeeds otherwise.
    """
    assert True


def test_simulator(simulator):
    """
    Starts and stops the Maxeler Simulator.
    """
    simulator.restart()
    simulator.stop()
    assert True


def test_passthrough(simulator, build_dir, src_root):
    """
    A simple test streaming data to the DFE and back to CPU.
    """
    compile(c_src=src_root / 'passthrough', maxj_src=src_root / 'passthrough',
            build_dir=build_dir, target='PassThrough', manager='PassThroughMAX5CManager',
            package='passthrough')
    simulator.run(build_dir / 'PassThrough')


def test_passthrough_ctypes(simulator, build_dir, src_root):
    """
    A simple test streaming data to the DFE and back to CPU, called via ctypes
    """
    # First, build shared library
    compile(c_src=src_root / 'passthrough', maxj_src=src_root / 'passthrough',
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
