# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from importlib import import_module, reload
import os
from pathlib import Path
import re
import shutil
import sys

from loki.logging import info, debug
from loki.tools import execute, as_tuple, delete


__all__ = [
    'clean', 'compile', 'compile_and_load', '_default_compiler',
    'Compiler', 'get_compiler_from_env', 'GNUCompiler', 'NvidiaCompiler'
]


def _which(cmd):
    """
    Convenience wrapper around :any:`shutil.which` that adds the binary
    directory of the Python interpreter to the search path

    This is useful when called from a script that is installed
    in a virtual environment without having explicitly enabled that environment.
    In that case, utilities like f90wrap may be installed inside the virtual
    environment but the binary dir will not be part of the search path.
    """
    return shutil.which(cmd, path=f'{os.environ["PATH"]}:{Path(sys.executable).parent}')


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


def compile_and_load(filename, cwd=None, f90wrap_kind_map=None, compiler=None):
    """
    Just-in-time compile Fortran source code and load the respective
    module or class.

    Both paths, classic subroutine-only and modern module-based are
    supported via the ``f2py`` and ``f90wrap`` packages.

    Parameters
    ----------
    filename : str
        The source file to be compiled.
    cwd : str, optional
        Working directory to use for calls to compiler.
    f90wrap_kind_map : str, optional
        Path to ``f90wrap`` KIND_MAP file, containing a Python dictionary
        in f2py_f2cmap format.
    compiler : :any:`Compiler`, optional
        Use the specified compiler to compile the Fortran source code. Defaults
        to :any:`_default_compiler`
    """
    info(f'Compiling: {filename}')
    filepath = Path(filename)
    clean(filename)

    pattern = ['*.f90.cache', '*.o', '*.mod', 'f90wrap_*.f90',
               f'{filepath.stem}.cpython*.so', f'{filepath.stem}.py']
    clean(filename, pattern=pattern)

    # First, compile the module and object files
    if not compiler:
        compiler = _default_compiler
    compiler.compile(filepath.absolute(), cwd=cwd)

    # Generate the Python interfaces
    compiler.f90wrap(modname=filepath.stem, source=[filepath.absolute()], kind_map=f90wrap_kind_map, cwd=cwd)

    # Compile the dynamic library
    f2py_source = [f'{filepath.stem}.o']
    for sourcefile in [f'f90wrap_{filepath.stem}.f90', 'f90wrap_toplevel.f90']:
        if (filepath.parent/sourcefile).exists():
            f2py_source += [sourcefile]
    compiler.f2py(modname=filepath.stem, source=f2py_source, cwd=cwd)

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
    F2PY_FCOMPILER_TYPE = None

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
        self.f2py_fcompiler_type = self.F2PY_FCOMPILER_TYPE or 'gnu95'

    def compile_args(self, source, target=None, include_dirs=None, mod_dir=None, mode='f90'):
        """
        Generate arguments for the build line.

        Parameters:
        -----------
        source : str or pathlib.Path
            Path to the source file to compile
        target : str or pathlib.Path, optional
            Path to the output binary to generate
        include_dirs : list of str or pathlib.Path, optional
            Path of include directories to specify during compile
        mod_dir : str or pathlib.Path, optional
            Path to directory containing Fortran .mod files
        mode : str, optional
            One of ``'f90'`` (free form), ``'f'`` (fixed form) or ``'c'``
        """
        assert mode in ['f90', 'f', 'c']
        include_dirs = include_dirs or []
        cc = {'f90': self.f90, 'f': self.fc, 'c': self.cc}[mode]
        args = [cc, '-c']
        args += {'f90': self.f90flags, 'f': self.fcflags, 'c': self.cflags}[mode]
        args += self._include_dir_args(include_dirs)
        if mode != 'c':
            args += self._mod_dir_args(mod_dir)
        args += [] if target is None else ['-o', str(target)]
        args += [str(source)]
        return args

    def _include_dir_args(self, include_dirs):
        """
        Return a list of compile command arguments for adding
        all paths in :data:`include_dirs` as include directories
        """
        return [
            f'-I{incl!s}' for incl in as_tuple(include_dirs)
        ]

    def _mod_dir_args(self, mod_dir):
        """
        Return a list of compile command arguments for setting
        :data:`mod_dir` as search and output directory for module files
        """
        if mod_dir is None:
            return []
        return [f'-J{mod_dir!s}']

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
        args = [_which('f90wrap')]
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

    def f2py_args(self, modname, source, libs=None, lib_dirs=None, incl_dirs=None):
        """
        Generate arguments for the ``f2py-f90wrap`` utility invocation line.
        """
        libs = libs or []
        lib_dirs = lib_dirs or []
        incl_dirs = incl_dirs or []

        args = [_which('f2py-f90wrap'), '-c']
        args += [f'--fcompiler={self.f2py_fcompiler_type}']
        args += [f'--f77exec={self.fc}']
        args += [f'--f90exec={self.f90}']
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


class GNUCompiler(Compiler):
    """
    GNU compiler configuration for gcc and gfortran
    """

    CC = 'gcc'
    CFLAGS = ['-g', '-fPIC']
    F90 = 'gfortran'
    F90FLAGS = ['-g', '-fPIC']
    FC = 'gfortran'
    FCFLAGS = ['-g', '-fPIC']
    LD = 'gfortran'
    LDFLAGS = ['-static']
    LD_STATIC = 'ar'
    LDFLAGS_STATIC = ['src']
    F2PY_FCOMPILER_TYPE = 'gnu95'

    CC_PATTERN = re.compile(r'(^|/|\\)gcc\b')
    FC_PATTERN = re.compile(r'(^|/|\\)gfortran\b')


class NvidiaCompiler(Compiler):
    """
    NVHPC compiler configuration for nvc and nvfortran
    """

    CC = 'nvc'
    CFLAGS = ['-g', '-fPIC']
    F90 = 'nvfortran'
    F90FLAGS = ['-g', '-fPIC']
    FC = 'nvfortran'
    FCFLAGS = ['-g', '-fPIC']
    LD = 'nvfortran'
    LDFLAGS = ['-static']
    LD_STATIC = 'ar'
    LDFLAGS_STATIC = ['src']
    F2PY_FCOMPILER_TYPE = 'nv'

    CC_PATTERN = re.compile(r'(^|/|\\)nvc\b')
    FC_PATTERN = re.compile(r'(^|/|\\)(pgf9[05]|pgfortran|nvfortran)\b')

    def _mod_dir_args(self, mod_dir):
        if mod_dir is None:
            return []
        return ['-module', str(mod_dir)]


def get_compiler_from_env(env=None):
    """
    Utility function to determine what compiler to use

    This takes the following environment variables in the given order
    into account to determine the most likely compiler family:
    ``F90``, ``FC``, ``CC``.

    Currently, :any:`GNUCompiler` and :any:`NvidiaCompiler` are available.

    The compiler binary and flags can be further overwritten by setting
    the corresponding environment variables:

    - ``CC``, ``FC``, ``F90``, ``LD`` for compiler/linker binary name or path
    - ``CFLAGS``, ``FCFLAGS``, ``LDFLAGS`` for compiler/linker flags to use

    Parameters
    ----------
    env : dict, optional
        Use the specified environment (default: :any:`os.environ`)

    Returns
    -------
    :any:`Compiler`
        A compiler object
    """
    if env is None:
        env = os.environ

    candidates = (GNUCompiler, NvidiaCompiler)
    compiler = None

    # "guess" the most likely compiler choice
    var_pattern_map = {
        'F90': 'FC_PATTERN',
        'FC': 'FC_PATTERN',
        'CC': 'CC_PATTERN'
    }
    for var, pattern in var_pattern_map.items():
        if env.get(var):
            for candidate in candidates:
                if getattr(candidate, pattern).search(env[var]):
                    compiler = candidate()
                    debug(f'Environment variable {var}={env[var]} set, using {candidate}')
                    break
            else:
                continue
            break

    if compiler is None:
        compiler = Compiler()

    # overwrite compiler executable and compiler flags with environment values
    var_compiler_map = {
        'CC': 'cc',
        'FC': 'fc',
        'F90': 'f90',
        'LD': 'ld',
    }
    for var, attr in var_compiler_map.items():
        if var in env:
            setattr(compiler, attr, env[var].strip())
            debug(f'Environment variable {var} set, using custom compiler executable {env[var]}')

    var_flag_map = {
        'CFLAGS': 'cflags',
        'FCFLAGS': 'fcflags',
        'LDFLAGS': 'ldflags',
    }
    for var, attr in var_flag_map.items():
        if var in env:
            setattr(compiler, attr, env[var].strip().split())
            debug(f'Environment variable {var} set, overwriting compiler flags as {env[var]}')

    return compiler


# TODO: Properly integrate with a config dict (with callbacks)
_default_compiler = get_compiler_from_env()
