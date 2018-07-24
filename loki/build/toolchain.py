from loki.build.tools import flatten

__all__ = ['Toolchain']


class Toolchain(object):
    """
    Base class for specifying different compiler toolchains.

    Note, we are currently using GCC settings as default.
    """

    def __init__(self, cc=None, cflags=None, f90=None, f90flags=None, ld=None, ldflags=None):
        self.cc = cc or 'gcc'
        self.cflag = cflags or ['-g']
        self.f90 = f90 or 'gfortran'
        self.f90flags = f90flags or ['-g']
        self.ld = ld or 'gfortran'
        self.ldflags = '-fpic'

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
        args += ['-o %s' % target]
        args += objs
        return args


# TODO: Properly integrate with a config dict (with callbacks)
_default_toolchain = Toolchain()

# TODO: Build more common and custom presets
