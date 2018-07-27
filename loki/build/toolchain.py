from pathlib import Path
from loki.build.tools import flatten


__all__ = ['Toolchain']


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

    def __init__(self, cc=None, cflags=None, f90=None, f90flags=None, ld=None, ldflags=None):
        self.cc = cc or 'gcc'
        self.cflag = cflags or ['-g', '-fPIC']
        self.f90 = f90 or 'gfortran'
        self.f90flags = f90flags or ['-g', '-fPIC']
        self.ld = ld or 'gfortran'
        self.ldflags = ['-shared']

    def build_args(self, source, target=None, include_dirs=[]):
        """
        Generate arguments for the build line.
        """
        args = [self.f90, '-c']
        args += self.f90flags
        args += flatten([('-I', '%s' % incl) for incl in include_dirs])
        args += [] if target is None else ['-o', '%s' % target]
        args += [str(source)]
        return args

    def linker_args(self, objs, target):
        """
        Generate arguments for the linker line.
        """
        args = [self.ld]
        args += self.ldflags
        args += ['-o', '%s' % target]
        args += objs
        return args

    def f90wrap_args(self, modname, source):
        """
        Generate arguments for the ``f90wrap`` utility invocation line.
        """
        args = ['f90wrap']
        args += ['-m', '%s' % modname]
        args += ['-k', str(_test_base_dir/'kind_map')]  # TODO: Generalize as option
        args += ['%s' % source]
        return args

    def f2py_args(self, modname, source, libs=[], lib_dirs=[]):
        """
        Generate arguments for the ``f2py-f90wrap`` utility invocation line.
        """
        args = ['f2py-f90wrap', '-c']
        args += ['-m', '_%s' % modname]
        args += ['%s.o' % modname]
        for lib in libs:
            args += ['-l%s' % lib]
        for lib_dir in lib_dirs:
            args += ['-L%s' %lib_dir]
        args += ['%s' % source]
        return args


# TODO: Properly integrate with a config dict (with callbacks)
_default_toolchain = Toolchain()


# TODO: Build more common and custom presets
