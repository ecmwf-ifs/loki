from pathlib import Path

from loki.build.tools import as_tuple
from loki.build.logging import _default_logger, warning
from loki.build.compiler import _default_compiler
from loki.build.obj import Obj


__all__ = ['Lib']


class Lib(object):
    """
    A library object linked from multiple compiled objects.

    :param name: Name of the resulting library (without leading ``lib``).
    :param shared: Flag indicating a shared library build.
    :param objs: Either a :class:`Obj` or a string to be used as a
                 glob pattern to search for in :param:`source_dir`.
    :param source_dir: A file path to generate dependency objects from
                       if :param obj: is given as a glob pattern.
    """

    def __init__(self, name, objs=None, shared=True, source_dir=None):
        self.name = name
        self.shared = shared

        self.path = Path('lib%s' % name)
        self.path = self.path.with_suffix('.so' if shared else '.a')

        if isinstance(objs, str):
            # Generate object list by globbing the source_dir
            assert source_dir is not None
            source_dir = Path(source_dir)
            self.objs = [Obj(filename=f) for f in source_dir.glob(objs)]
        else:
            self.objs = objs

        if len(objs) == 0:
            warning('%s:: Empty dependency list: %s' % (self, self.objs))

    def __repr__(self):
        return 'Lib<%s>' % self.name

    def build(self, builder=None, logger=None, compiler=None, shared=None):
        """
        Build the source objects and create target library.

        TODO: This does not yet(!) auto-build dependencies.
        """
        compiler = compiler or builder.compiler
        logger = logger or builder.logger
        shared = shared or self.shared
        build_dir = builder.build_dir

        logger.info('Building %s' % self)
        for obj in self.objs:
            obj.build(builder=builder, logger=logger)

        objs = [obj.path.with_suffix('.o') for obj in self.objs]
        target = self.path if shared else self.path.with_suffix('.a')
        compiler.link(target=target, objs=objs, shared=shared, cwd=build_dir)

    def wrap(self, modname, sources=None):
        """
        Wrap the compiled library using ``f90wrap`` and return the loaded module.

        :param sources: List of source files to wrap for Python access.
        """
        items = as_tuple(self.builder.Obj(s) for s in as_tuple(sources))
        build_dir = self.builder.build_dir
        compiler = self.builder.compiler or _default_compiler

        sourcepaths = [str(i.path) for i in items]
        compiler.f90wrap(modname=modname, source=sourcepaths, cwd=str(build_dir))

        # Execute the second-level wrapper (f2py-f90wrap)
        wrappers = ['f90wrap_%s.f90' % item.path.stem for item in items]
        wrappers += ['f90wrap_toplevel.f90']  # Include the generic wrapper
        wrappers = [w for w in wrappers if (build_dir/w).exists()]

        libs = [self.name]
        lib_dirs = [str(build_dir.absolute())]
        compiler.f2py(modname=modname, source=wrappers,
                      libs=libs, lib_dirs=lib_dirs, cwd=str(build_dir))

        return self.builder.load_module(modname)
