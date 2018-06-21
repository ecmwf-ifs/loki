from subprocess import run, PIPE, STDOUT, CalledProcessError
from pathlib import Path
from importlib import import_module
import os
import shutil

from loki.logging import info, error


__all__ = ['execute', 'compile_and_load']


def execute(args):
    info('Executing: %s' % ' '.join(args))
    try:
        run(args, check=True, stdout=PIPE, stderr=STDOUT)
    except CalledProcessError as e:
        error('Execution failed with:')
        info(e.output.decode("utf-8"))

def clean(filename):
    """
    Clean up compilation files of previous runs.

    :param filename: Filename that triggered the original compilation.
    """
    filepath = Path(filename)
    shutil.rmtree('%s' % filepath.with_suffix('.cpython*.so'), ignore_errors=True)
    # os.remove('%s' % filepath.with_suffix('.cpython*.so'))


def compile_and_load(filename, use_f90wrap=False):
    """
    Just-in-time compiles Fortran source code and loads the respective
    module or class. Both paths, classic subroutine-only and modern
    module-based are support ed via the ``f2py`` and ``f90wrap`` packages.

    :param filename: The source file to be compiled.
    :param use_f90wrap: Flag to trigger the ``f90wrap`` toolchain required
                        if the source code includes module or derived types.
    """

    if use_f90wrap:
        raise NotImplementedError('Not supporting f90wrap yet, patience...')
    else:
        # Basic subroutine compilation via f2py
        filepath = Path(filename)
        cmd = ['f2py']
        cmd += ['-c', '%s' % filepath.absolute()]
        cmd += ['-m', '%s' % filepath.stem]

        # Execute the f2py and load the resulting module
        clean(filename)
        execute(cmd)
        return import_module(filepath.stem)
