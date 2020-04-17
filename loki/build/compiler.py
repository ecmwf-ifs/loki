from pathlib import Path
from importlib import import_module

from loki.logging import info
from loki.build.tools import as_tuple, execute, flatten, delete


__all__ = ['clean', 'compile', 'compile_and_load',
           '_default_compiler', 'Compiler', 'GNUCompiler', 'EscapeGNUCompiler']


_test_base_dir = Path(__file__).parent.parent.parent/'tests'


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


def compile_and_load(filename, cwd=None, use_f90wrap=True):
    """
    Just-in-time compiles Fortran source code and loads the respective
    module or class. Both paths, classic subroutine-only and modern
    module-based are support ed via the ``f2py`` and ``f90wrap`` packages.

    :param filename: The source file to be compiled.
    :param use_f90wrap: Flag to trigger the ``f90wrap`` compiler required
                        if the source code includes module or derived types.
    """
    info('Compiling: %s' % filename)
    filepath = Path(filename)
    clean(filename)

    pattern = ['*.f90.cache', '*.o', '*.mod', 'f90wrap_*.f90',
               '%s.cpython*.so' % filepath.stem, '%s.py' % filepath.stem]
    clean(filename, pattern=pattern)

    # First, compile the module and object files
    build = ['gfortran', '-c', '-fpic', '%s' % filepath.absolute()]
    execute(build, cwd=cwd)

    # Generate the Python interfaces
    f90wrap = ['f90wrap']
    f90wrap += ['-m', '%s' % filepath.stem]
    f90wrap += ['-k', str(_test_base_dir/'kind_map')]  # TODO: Generalize as option
    f90wrap += ['%s' % filepath.absolute()]
    execute(f90wrap, cwd=cwd)

    # Compile the dynamic library
    f2py = ['f2py-f90wrap', '-c']
    f2py += ['-m', '_%s' % filepath.stem]
    f2py += ['%s.o' % filepath.stem]
    for sourcefile in ['f90wrap_%s.f90' % filepath.stem, 'f90wrap_toplevel.f90']:
        if (filepath.parent/sourcefile).exists():
            f2py += [sourcefile]
    execute(f2py, cwd=cwd)

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
        args += flatten([('-I', '%s' % incl) for incl in include_dirs])
        args += [] if mod_dir is None else ['-J', '%s' % mod_dir]
        args += [] if target is None else ['-o', '%s' % target]
        args += [str(source)]
        return args

    def compile(self, source, target=None, include_dirs=None, use_c=False, cwd=None, logger=None):
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
        args = [self.ld if shared else self.ld_static]
        args += self.ldflags if shared else self.ldflags_static
        args += ['-o', '%s' % target]
        args += [str(o) for o in objs]
        return args

    def link(self, objs, target, shared=True, cwd=None):
        """
        Execute a link command for a given source.
        """
        args = self.linker_args(objs=objs, target=target, shared=shared)
        execute(args, cwd=cwd)

    def f90wrap_args(self, modname, source):
        """
        Generate arguments for the ``f90wrap`` utility invocation line.
        """
        args = ['f90wrap']
        args += ['-m', '%s' % modname]
        args += ['-k', str(_test_base_dir/'kind_map')]  # TODO: Generalize as option
        args += ['%s' % s for s in source]
        return args

    def f90wrap(self, modname, source, cwd=None):
        """
        Invoke f90wrap command to create wrappers for a given module.
        """
        args = self.f90wrap_args(modname=modname, source=source)
        execute(args, cwd=cwd)

    def f2py_args(self, modname, source, libs=None, lib_dirs=None, incl_dirs=None):
        """
        Generate arguments for the ``f2py-f90wrap`` utility invocation line.
        """
        libs = libs or []
        lib_dirs = lib_dirs or []
        incl_dirs = incl_dirs or []

        args = ['f2py-f90wrap', '-c']
        args += ['-m', '_%s' % modname]
        for incl_dir in incl_dirs:
            args += ['-I%s' % incl_dir]
        for lib in libs:
            args += ['-l%s' % lib]
        for lib_dir in lib_dirs:
            args += ['-L%s' % lib_dir]
        args += ['%s' % s for s in source]
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
