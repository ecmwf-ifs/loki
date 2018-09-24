from pathlib import Path
import networkx as nx
from operator import attrgetter
from tqdm import tqdm

from loki.build.tools import as_tuple, find_paths, execute
from loki.build.logging import _default_logger, warning, error
from loki.build.compiler import _default_compiler
from loki.build.obj import Obj
from loki.build.workqueue import workqueue, DEFAULT_TIMEOUT


__all__ = ['Lib']


class Lib(object):
    """
    A library object linked from multiple compiled objects (:class:`Obj`).

    Note, eitehr :param objs: or the arguments :param pattern: and
    :param source_dir: are required to generated the necessary dependencies.

    :param name: Name of the resulting library (without leading ``lib``).
    :param shared: Flag indicating a shared library build.
    :param objs: List of :class:`Obj` objects that define the objects to link.
    :param pattern: A glob pattern that determines the objects to link.
    :param source_dir: A file path to find objects on when resolving glob patterns.
    :param ignore: A (list of) glob patterns definig file to ignore when
                   generating dependencies from a glob pattern.
    """

    def __init__(self, name, shared=True, objs=None, pattern=None, source_dir=None, ignore=None):
        self.name = name
        self.shared = shared

        if objs is not None:
            self.objs = objs

        else:
            # Generate object list by globbing the source_dir according to pattern
            if source_dir is None:
                raise RuntimeError('No source directory found for pattern expansion in %s' % self)

            obj_paths = find_paths(directory=source_dir, pattern=pattern, ignore=ignore)
            self.objs = [Obj(source_path=p) for p in obj_paths]

        if len(self.objs) == 0:
            warning('%s:: Empty dependency list: %s' % (self, self.objs))

    def __repr__(self):
        return 'Lib<%s>' % self.name

    def build(self, builder=None, logger=None, compiler=None, shared=None):
        """
        Build the source objects and create target library.
        """
        compiler = compiler or builder.compiler
        logger = logger or builder.logger
        shared = shared or self.shared
        build_dir = builder.build_dir
        workers = builder.workers

        logger.info('Building %s (workers=%s)' % (self, workers))

        def wait_and_check(obj):
            if obj.q_task is not None:
                obj.q_task.wait(DEFAULT_TIMEOUT)

                if obj.q_task.status =='failed':
                    error('Failed task: %s' % obj.q_task)
                    raise RuntimeError('Error compiling object: %s' % obj)

        # Generate the dependncy graph implied by .mod files
        dep_graph = builder.get_dependency_graph(self.objs, depgen=attrgetter('dependencies'))

        with workqueue(workers=workers) as q:

            topo_nodes = list(reversed(list(nx.topological_sort(dep_graph))))
            for obj in tqdm(topo_nodes):
                if obj.source_path and obj.q_task is None:

                    # Wait dependencies to complete before scheduling item
                    for dep in obj.obj_dependencies:
                        wait_and_check(dep)

                    # Schedule object compilation on the workqueue
                    obj.build(builder=builder, logger=logger, workqueue=q)

            # Ensure all build tasks have finished
            for obj in dep_graph.nodes:
                if obj.q_task is not None:
                    wait_and_check(obj)

        # Link the final library
        objs = [(build_dir/obj.name).with_suffix('.o') for obj in self.objs]
        target = (build_dir/('lib%s' % self.name)).with_suffix('.so' if shared else '.a')
        args = compiler.linker_args(target=target, objs=objs, shared=shared)
        execute(args)

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
