import pytest
import os
from pathlib import Path

from loki.build.tools import execute
from loki.build.max_compiler import compile_and_load


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
            self.environ = os.environ.copy()
            self.environ.update({'SLIC_CONF': 'use_simulation=%s' % name,
                                 'LD_PRELOAD': ('%s/lib/libmaxeleros.so' %
                                                self.environ['MAXELEROSDIR'])})

        def restart(self):
            cmd = self.base_cmd + ['-c', 'MAX5C', 'restart']
            execute(cmd)

        def stop(self):
            cmd = self.base_cmd + ['stop']
            execute(cmd)

        def run(self, target, args=None):
            cmd = [str(target)]
            if args is not None:
                cmd += [str(a) for a in args]
            self.restart()
            execute(cmd, env=self.environ)
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
    compile_and_load(c_src=src_root / 'passthrough', maxj_src=src_root / 'passthrough',
                     build_dir=build_dir, target='PassThrough',
                     manager='PassThroughMAX5CManager', package='passthrough')
    simulator.run(build_dir / 'PassThrough')
