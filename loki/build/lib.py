from pathlib import Path

from loki.build.tools import as_tuple
from loki.build.logging import _default_logger
from loki.build.toolchain import _default_toolchain


__all__ = ['Lib']


class Lib(object):
    """
    A library linked from multiple objects.
    """

    def __init__(self, name, objects=None, builder=None, logger=None):
        self.name = name
        self.path = Path('lib%s.so' % name)
        self.builder = builder
        self.objects = objects or []
        self.logger = logger or _default_logger

    def __repr__(self):
        return 'Lib<%s>' % self.path.name

    def build(self):
        """
        Build the source objects and create target library.

        TODO: This does not yet(!) auto-build dependencies.
        """
        build_dir = str(self.builder.build_dir)
        # TODO: Support static libs
        target = '%s.a' % self.path.stem
        toolchain = self.builder.toolchain or _default_toolchain

        self.logger.info('Building lib %s' % self)
        for obj in self.objects:
            obj.build()

        # Important: Since we cannot set LD_LIBRARY_PATH from within the
        # Python interpreter (not easily anyway), we ned to compile the
        # library statically, so that it can be baked into the wrapper.
        objs = ['%s.o' % o.path.stem for o in self.objects]
        toolchain.link(target=target, objs=objs, shared=False, cwd=build_dir)

    def wrap(self, modname, sources=None):
        """
        Wrap the compiled library using ``f90wrap`` and return the loaded module.

        :param sources: List of source files to wrap for Python access.
        """
        items = as_tuple(self.builder.Obj(s) for s in as_tuple(sources))
        build_dir = self.builder.build_dir
        toolchain = self.builder.toolchain or _default_toolchain

        sourcepaths = [str(i.path) for i in items]
        toolchain.f90wrap(modname=modname, source=sourcepaths, cwd=str(build_dir))

        # Execute the second-level wrapper (f2py-f90wrap)
        wrappers = ['f90wrap_%s.f90' % item.path.stem for item in items]
        wrappers += ['f90wrap_toplevel.f90']  # Include the generic wrapper
        wrappers = [w for w in wrappers if (build_dir/w).exists()]

        libs = [self.name]
        lib_dirs = [str(build_dir.absolute())]
        toolchain.f2py(modname=modname, source=wrappers,
                       libs=libs, lib_dirs=lib_dirs, cwd=str(build_dir))

        return self.builder.load_module(modname)
