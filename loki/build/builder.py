import sys
from pathlib import Path
from functools import lru_cache
import networkx as nx
from collections import deque, OrderedDict
from importlib import import_module

from loki.logging import info, debug  # The only upwards dependency!

from loki.build.tools import as_tuple, delete
from loki.build.compiler import _default_compiler
from loki.build.logging import _default_logger


__all__ = ['Builder']


class Builder(object):
    """
    A :class:`Builder` that compiles binaries or libraries, while performing
    automated dependency discovery from one or more source paths.

    :param sources: One or more paths to search for source files
    :param includes: One or more paths to that include header files
    """

    def __init__(self, source_dirs=None, include_dirs=None, root_dir=None,
                 build_dir=None, compiler=None, logger=None):
        self.compiler = compiler or _default_compiler
        self.logger = logger or _default_logger

        # Source dirs for auto-detection and include dis for preprocessing
        self.source_dirs = [Path(p).resolve() for p in as_tuple(source_dirs)]
        self.include_dirs = [Path(p).resolve() for p in as_tuple(include_dirs)]

        # Root and source directories for out-of source builds
        self.root_dir = None if root_dir is None else Path(root_dir)
        self.build_dir = Path.cwd() if build_dir is None else Path(build_dir)
        self.build_dir.mkdir(exist_ok=True)

        # Create the dependency graph and it's utilities
        self.dependency_graph = nx.DiGraph()
        self._cache = OrderedDict()

    def find_path(self, filename):
        """
        Scan all source paths for source files and create build item.

        :param filename: Name of the source file we are looking for.
        """
        for s in self.source_dirs:
            filepaths = list(s.glob('**/%s' % filename))
            if len(filepaths) == 0:
                return None
            elif len(filepaths) == 1:
                return filepaths[0]
            else:
                return filepaths

    def __getitem__(self, *args, **kwargs):
        return self.Obj(*args, **kwargs)

    def get_dependency_graph(self, builditem):
        """
        Construct a :class:`networkxDiGraph` that represents the dependency graph.
        """
        q = deque([builditem])
        g = nx.DiGraph()
        seen = []

        while len(q) > 0:
            item = q.popleft()
            seen.append(item)

            for dep in item.dependencies:
                node = self.get_item(dep)

                if node is None:
                    # TODO: Warn for missing dependency
                    continue

                if node not in seen:
                    g.add_node(node)
                    q.append(node)

                g.add_edge(item, node)

        return g

    def clean(self, rules=None, path=None):
        """
        Clean up a build directory according, either according to
        globbing rules or via explicit file paths.

        :param rules: String or list of strings with either explicit
                      filepaths or globbing rules; default is
                      ``'*.o *.mod *.so f90wrap*.f90'``.
        :param path: Optional directory path to clean; defaults
                     first to ``self.build_dir``, then simply ``./``.
        """
        # Derive defaults, split string rules and ensure iterability
        rules = rules or '*.o *.mod *.so f90wrap*.f90'
        if isinstance(rules, str):
            rules = rules.split(' ')
        rules = as_tuple(rules)

        path = path or self.build_dir or Path('.')

        for r in rules:
            for f in path.glob(r):
                delete(f)

    def build(self, filename, target=None, shared=True):
        item = self.get_item(filename)
        self.logger.info("Building %s" % item)

        build_dir = str(self.build_dir) if self.build_dir else None

        # Build the entire dependency graph, including the source object
        objs = []
        dependencies = self.get_dependency_graph(item)
        for dep in reversed(list(nx.topological_sort(dependencies))):
            dep.build(compiler=self.compiler, build_dir=build_dir,
                      include_dirs=self.include_dirs)
            objs += ['%s.o' % dep.path.stem]

        if target is not None:
            self.logger.info('Linking target: %s' % target)
            self.compiler.link(objs=objs, target=target, cwd=build_dir)

    def load_module(self, module):
        """
        Handle import paths and load the compiled module
        """
        if str(self.build_dir.absolute()) not in sys.path:
            sys.path.insert(0, str(self.build_dir.absolute()))
        return import_module(module)

    def wrap_and_load(self, sources, modname=None, build=True,
                      libs=None, lib_dirs=None, incl_dirs=None):
        """
        Performs the necessary build steps to compile and wrap a set
        of sources using ``f90wrap``. It returns return dynamically
        loaded Python module containinig wrappers for each Fortran
        procedure and module specified in :param sources:.

        :param source: Name(s) of source files to wrap
        :param modname: Optional module name for f90wrap to use
        :param build: Flag to force building the source first; default: True.
        :param libs: Optional override for library names to link
        :param lib_dirs: Optional override for library paths to link from
        :param incl_dirs: Optional override for include directories
        """
        items = as_tuple(self.get_item(s) for s in as_tuple(sources))
        build_dir = str(self.build_dir) if self.build_dir else None
        modname = modname or items[0].path.stem

        # Invoke build to ensure all base objects are built
        # TODO: Could automate this via timestamps/hashes, etc.
        if build:
            for item in items:
                target = 'lib%s.a' % item.path.stem
                self.build(item.path.name, target=target, shared=False)

        # Execute the first-level wrapper (f90wrap)
        self.logger.info('Python-wrapping %s' % items[0])
        sourcepaths = [str(i.path) for i in items]
        self.compiler.f90wrap(modname=modname, source=sourcepaths, cwd=build_dir)

        # Execute the second-level wrapper (f2py-f90wrap)
        wrappers = ['f90wrap_%s.f90' % item.path.stem for item in items]
        wrappers += ['f90wrap_toplevel.f90']  # Include the generic wrapper
        wrappers = [w for w in wrappers if (self.build_dir/w).exists()]

        # Resolve final compilation libraries and include dirs
        libs = libs or [modname]
        lib_dirs = lib_dirs or ['%s' % self.build_dir.absolute()]
        incl_dirs = incl_dirs or []

        self.compiler.f2py(modname=modname, source=wrappers,
                           libs=libs, lib_dirs=lib_dirs,
                           incl_dirs=incl_dirs, cwd=build_dir)

        self.load_module(module=modname)
