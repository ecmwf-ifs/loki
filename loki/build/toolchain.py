from pathlib import Path
from loki.build.tools import flatten, execute


__all__ = ['Toolchain', 'GNUToolchain', 'EscapeGNUToolchain']


"""
Hard-coded relative location of a default ``kindmap`` file
for f90wrap.
"""
_test_base_dir = Path(__file__).parent.parent.parent/'tests'


class Toolchain(object):
    """
    Base class for specifying different compiler toolchains.

    Note, we are currently using GCC settings as default.
    """

    CC = None
    CFLAGS = None
    F90 = None
    F90FLAGS = None
    LD = None
    LDFLAGS = None

    def __init__(self):
        self.cc = self.CC or 'gcc'
        self.cflags = self.CFLAGS or ['-g', '-fPIC']
        self.f90 = self.F90 or 'gfortran'
        self.f90flags = self.F90FLAGS or ['-g', '-fPIC']
        self.ld = self.LD or 'gfortran'
        self.ldflags = self.LDFLAGS or []

    def build_args(self, source, target=None, include_dirs=None):
        """
        Generate arguments for the build line.
        """
        include_dirs = include_dirs or []
        args = [self.f90, '-c']
        args += self.f90flags
        args += flatten([('-I', '%s' % incl) for incl in include_dirs])
        args += [] if target is None else ['-o', '%s' % target]
        args += [str(source)]
        return args

    def build(self, source, target=None, include_dirs=None, cwd=None):
        """
        Execute a build command for a given source.
        """
        args = self.build_args(source, target=target, include_dirs=include_dirs)
        execute(args, cwd=cwd)

    def linker_args(self, objs, target, shared=True):
        """
        Generate arguments for the linker line.
        """
        args = [self.ld] if shared else ['ar', 'src']
        args += self.ldflags
        args += ['-shared'] if shared else []
        args += ['-o', '%s' % target] if shared else [target]
        args += objs
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
_default_toolchain = Toolchain()


class GNUToolchain(Toolchain):

    CC = 'gcc'
    CFLAGS = ['-g', '-fPIC']
    F90 = 'gfortran'
    F90FLAGS = ['-g', '-fPIC']
    LD = 'gfortran'
    LDFLAGS = []


class EscapeGNUToolchain(GNUToolchain):

    F90FLAGS = ['-O3', '-g', '-fPIC',
                '-ffpe-trap=invalid,zero,overflow', '-fstack-arrays',
                '-fconvert=big-endian',
                '-fbacktrace',
                '-fno-second-underscore',
                '-ffree-form',
                '-ffast-math',
                '-fno-unsafe-math-optimizations']
