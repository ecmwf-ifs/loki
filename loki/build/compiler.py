# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from importlib import import_module, reload
import sys

from loki.logging import info
from loki.tools import execute, as_tuple, flatten, delete


__all__ = ['clean', 'compile', 'compile_and_load',
           '_default_compiler', 'Compiler', 'GNUCompiler', 'EscapeGNUCompiler']


def compile(filename, include_dirs=None, compiler=None, cwd=None):
    # Stop complaints about `compile` in this function
    # pylint: disable=redefined-builtin
    filepath = Path(filename)
    compiler = compiler or _default_compiler
    args = compiler.build_args(source=filepath.absolute(),
                               include_dirs=include_dirs)
    execute(args, cwd=cwd)


def clean(filename, pattern=None):
    """
    Clean up compilation files of previous runs.

    :param filename: Filename that triggered the original compilation.
    :param suffixes: Optional list of filetype suffixes to delete.
    """
    filepath = Path(filename)
    pattern = pattern or ['*.f90.cache', '*.o', '*.mod']
    for p in as_tuple(pattern):
        for f in filepath.glob(p):
            delete(f)


def compile_and_load(filename, cwd=None, use_f90wrap=True, f90wrap_kind_map=None):  # pylint: disable=unused-argument
    """
    Just-in-time compile Fortran source code and load the respective
    module or class.

    Both paths, classic subroutine-only and modern module-based are
    supported via the ``f2py`` and ``f90wrap`` packages.

    Parameters
    -----
    filename : str
        The source file to be compiled.
    cwd : str, optional
        Working directory to use for calls to compiler.
    use_f90wrap : bool, optional
        Flag to trigger the ``f90wrap`` compiler required
        if the source code includes module or derived types.
    f90wrap_kind_map : str, optional
        Path to ``f90wrap`` KIND_MAP file, containing a Python dictionary
        in f2py_f2cmap format.
    """
    info(f'Compiling: {filename}')
    filepath = Path(filename)
    clean(filename)

    pattern = ['*.f90.cache', '*.o', '*.mod', 'f90wrap_*.f90',
               f'{filepath.stem}.cpython*.so', f'{filepath.stem}.py']
    clean(filename, pattern=pattern)

    # First, compile the module and object files
    build = ['gfortran', '-c', '-fpic', str(filepath.absolute())]
    execute(build, cwd=cwd)

    # Generate the Python interfaces
    f90wrap = ['f90wrap']
    f90wrap += ['-m', str(filepath.stem)]
    if f90wrap_kind_map is not None:
        f90wrap += ['-k', str(f90wrap_kind_map)]
    f90wrap += [str(filepath.absolute())]
    execute(f90wrap, cwd=cwd)

    # Compile the dynamic library
    f2py = ['f2py-f90wrap', '-c']
    f2py += ['-m', f'_{filepath.stem}']
    f2py += [f'{filepath.stem}.o']
    for sourcefile in [f'f90wrap_{filepath.stem}.f90', 'f90wrap_toplevel.f90']:
        if (filepath.parent/sourcefile).exists():
            f2py += [sourcefile]
    execute(f2py, cwd=cwd)

    # Add directory to module search path
    moddir = str(filepath.parent)
    if moddir not in sys.path:
        sys.path.append(moddir)

    if filepath.stem in sys.modules:
        # Reload module if already imported
        reload(sys.modules[filepath.stem])
        return sys.modules[filepath.stem]

    # Import module
    return import_module(filepath.stem)


class Compiler:
    """
    Base class for specifying different compiler toolchains.
    """

    CC = None
    CFLAGS = None
    F90 = None
    F90FLAGS = None
    FC = None
    FCFLAGS = None
    LD = None
    LDFLAGS = None
    LD_STATIC = None
    LDFLAGS_STATIC = None

    def __init__(self):
        self.cc = self.CC or 'gcc'
        self.cflags = self.CFLAGS or ['-g', '-fPIC']
        self.f90 = self.F90 or 'gfortran'
        self.f90flags = self.F90FLAGS or ['-g', '-fPIC']
        self.fc = self.FC or 'gfortran'
        self.fcflags = self.FCFLAGS or ['-g', '-fPIC']
        self.ld = self.LD or 'gfortran'
        self.ldflags = self.LDFLAGS or ['-static']
        self.ld_static = self.LD_STATIC or 'ar'
        self.ldflags_static = self.LDFLAGS_STATIC or ['src']

    def compile_args(self, source, target=None, include_dirs=None, mod_dir=None, mode='F90'):
        """
        Generate arguments for the build line.

        :param mode: One of ``'f90'`` (free form), ``'f'`` (fixed form) or ``'c'``.
        """
        assert mode in ['f90', 'f', 'c']
        include_dirs = include_dirs or []
        cc = {'f90': self.f90, 'f': self.fc, 'c': self.cc}[mode]
        args = [cc, '-c']
        args += {'f90': self.f90flags, 'f': self.fcflags, 'c': self.cflags}[mode]
        args += flatten([('-I', str(incl)) for incl in include_dirs])
        args += [] if mod_dir is None else ['-J', str(mod_dir)]
        args += [] if target is None else ['-o', str(target)]
        args += [str(source)]
        return args

    def compile(self, source, target=None, include_dirs=None, use_c=False, cwd=None):
        """
        Execute a build command for a given source.
        """
        kwargs = {'target': target, 'include_dirs': include_dirs}
        if use_c:
            kwargs['mode'] = 'c'
        args = self.compile_args(source, **kwargs)
        execute(args, cwd=cwd)

    def linker_args(self, objs, target, shared=True):
        """
        Generate arguments for the linker line.
        """
        linker = self.ld if shared else self.ld_static
        args = [linker]
        args += self.ldflags if shared else self.ldflags_static
        if linker != "ar":
            args += ['-o', str(target)]
        else:
            args += [str(target)]
        args += [str(o) for o in objs]
        return args

    def link(self, objs, target, shared=True, cwd=None):
        """
        Execute a link command for a given source.
        """
        args = self.linker_args(objs=objs, target=target, shared=shared)
        execute(args, cwd=cwd)

    @staticmethod
    def f90wrap_args(modname, source, kind_map=None):
        """
        Generate arguments for the ``f90wrap`` utility invocation line.
        """
        args = ['f90wrap']
        args += ['-m', str(modname)]
        if kind_map is not None:
            args += ['-k', str(kind_map)]
        args += [str(s) for s in source]
        return args

    def f90wrap(self, modname, source, cwd=None, kind_map=None):
        """
        Invoke f90wrap command to create wrappers for a given module.
        """
        args = self.f90wrap_args(modname=modname, source=source, kind_map=kind_map)
        execute(args, cwd=cwd)

    @staticmethod
    def f2py_args(modname, source, libs=None, lib_dirs=None, incl_dirs=None):
        """
        Generate arguments for the ``f2py-f90wrap`` utility invocation line.
        """
        libs = libs or []
        lib_dirs = lib_dirs or []
        incl_dirs = incl_dirs or []

        args = ['f2py-f90wrap', '-c']
        args += ['-m', f'_{modname}']
        for incl_dir in incl_dirs:
            args += [f'-I{incl_dir}']
        for lib in libs:
            args += [f'-l{lib}']
        for lib_dir in lib_dirs:
            args += [f'-L{lib_dir}']
        args += [str(s) for s in source]
        return args

    def f2py(self, modname, source, libs=None, lib_dirs=None, incl_dirs=None, cwd=None):
        """
        Invoke f90wrap command to create wrappers for a given module.
        """
        args = self.f2py_args(modname=modname, source=source, libs=libs,
                              lib_dirs=lib_dirs, incl_dirs=incl_dirs)
        execute(args, cwd=cwd)


# TODO: Properly integrate with a config dict (with callbacks)
_default_compiler = Compiler()


class GNUCompiler(Compiler):

    CC = 'gcc'
    CFLAGS = ['-g', '-fPIC']
    F90 = 'gfortran'
    F90FLAGS = ['-g', '-fPIC']
    LD = 'gfortran'
    LDFLAGS = []


class EscapeGNUCompiler(GNUCompiler):

    F90FLAGS = ['-O3', '-g', '-fPIC',
                '-ffpe-trap=invalid,zero,overflow', '-fstack-arrays',
                '-fconvert=big-endian',
                '-fbacktrace',
                '-fno-second-underscore',
                '-ffree-form',
                '-ffast-math',
                '-fno-unsafe-math-optimizations']
