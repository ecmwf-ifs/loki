from subprocess import run, PIPE, STDOUT, CalledProcessError
from pathlib import Path
from importlib import import_module
import os
import shutil

from loki.logging import debug, info, error
from loki.tools import as_tuple


__all__ = ['execute', 'clean', 'compile_and_load']


def execute(args):
    debug('Executing: %s' % ' '.join(args))
    try:
        run(args, check=True, stdout=PIPE, stderr=STDOUT)
    except CalledProcessError as e:
        error('Execution failed with:')
        info(e.output.decode("utf-8"))


def delete(filename, force=False):
    filepath = Path(filename)
    debug('Deleting %s' % filepath)
    if force:
        shutil.rmtree('%s' % filepath, ignore_errors=True)
    else:
        if filepath.exists():
            os.remove('%s' % filepath)


def clean(filename, pattern=None):
    """
    Clean up compilation files of previous runs.

    :param filename: Filename that triggered the original compilation.
    :param suffixes: Optional list of filetype suffixes to delete.
    """
    filepath = Path(filename)
    pattern = pattern or ['*.f90.cache', '*.o', '*.mod']
    for p in as_tuple(pattern):
        for f in filepath.parent.glob(p):
            delete(f)


def compile_and_load(filename, use_f90wrap=False):
    """
    Just-in-time compiles Fortran source code and loads the respective
    module or class. Both paths, classic subroutine-only and modern
    module-based are support ed via the ``f2py`` and ``f90wrap`` packages.

    :param filename: The source file to be compiled.
    :param use_f90wrap: Flag to trigger the ``f90wrap`` toolchain required
                        if the source code includes module or derived types.
    """
    info('Compiling: %s' % filename)
    filepath = Path(filename)
    clean(filename)

    if use_f90wrap:
        pattern=['*.f90.cache', '*.o', '*.mod', 'f90wrap_*.f90',
                 '*.cpython*.so', '%s.py' % filepath.stem]
        clean(filename, pattern=pattern)

        # First, compile the module and object files
        build = ['gfortran', '-c', '%s' % filepath.absolute()]
        execute(build)

        # Generate the Python interfaces
        f90wrap = ['f90wrap']
        f90wrap += ['-m', '%s' % filepath.stem]
        f90wrap += ['-k', str(filepath.parent/'kind_map')]  # TODO: Generalize as option
        f90wrap += ['%s' % filepath.absolute()]
        execute(f90wrap)

        # Compile the dynamic library
        f2py = ['f2py-f90wrap', '-c']
        f2py += ['-m', '_%s' % filepath.stem]
        f2py += ['f90wrap_%s.f90' % filepath.stem, '%s.o' % filepath.stem]
        execute(f2py)

        modname = '_'.join(s.capitalize() for s in filepath.stem.split('_'))

        return getattr(import_module(filepath.stem), modname)

    else:
        # Basic subroutine compilation via f2py
        cmd = ['f2py']
        cmd += ['-c', '%s' % filepath.absolute()]
        cmd += ['-m', '%s' % filepath.stem]

        # Execute the f2py and load the resulting module
        execute(cmd)
        return import_module(filepath.stem)
