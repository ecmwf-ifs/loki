# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys
from pathlib import Path
from collections import deque
from importlib import import_module, reload, invalidate_caches
from operator import attrgetter
import networkx as nx

from loki.logging import default_logger
from loki.tools import as_tuple, delete
from loki.build.compiler import _default_compiler
from loki.build.obj import Obj
from loki.build.header import Header


__all__ = ['Builder']


class Builder:
    """
    A :class:`Builder` that compiles binaries or libraries, while performing
    automated dependency discovery from one or more source paths.

    :param sources: One or more paths to search for source files
    :param includes: One or more paths to that include header files
    """

    def __init__(self, source_dirs=None, include_dirs=None, root_dir=None,
                 build_dir=None, compiler=None, logger=None, workers=3):
        self.compiler = compiler or _default_compiler
        self.logger = logger or default_logger
        self.workers = workers

        # Source dirs for auto-detection and include dis for preprocessing
        self.source_dirs = [Path(p).resolve() for p in as_tuple(source_dirs)]
        self.include_dirs = [Path(p).resolve() for p in as_tuple(include_dirs)]

        # Root and source directories for out-of source builds
        self.root_dir = None if root_dir is None else Path(root_dir)
        self.build_dir = Path.cwd() if build_dir is None else Path(build_dir)
        self.build_dir.mkdir(exist_ok=True)

        # Populate _object_cache for everything in source_dirs
        for source_dir in self.source_dirs:
            for ext in Obj._ext:
                _ = [Obj(source_path=f) for f in source_dir.glob(f'**/*{ext}')]

        for include_dir in self.include_dirs:
            for ext in Header._ext:
                _ = [Header(source_path=f) for f in include_dir.glob(f'**/*{ext}')]

    def __getitem__(self, *args, **kwargs):
        return Obj(*args, **kwargs)

    def get_item(self, key):
        return self[key]

    @staticmethod
    def get_dependency_graph(objs, depgen=None):
        """
        Construct a :class:`networkx.DiGraph` that represents the dependency graph.

        :param objs: List of :class:`Obj` to use as the root of the graph.
        :param depgen: Generator object to generate the next level of dependencies
                       from an item. Defaults to ``operator.attrgetter('dependencies')``.
        """
        depgen = depgen or attrgetter('dependencies')

        q = deque(as_tuple(objs))
        nodes = []
        edges = []

        while len(q) > 0:
            item = q.popleft()
            nodes.append(item)

            # Record the actual :class:`Obj` dependency objects
            item.obj_dependencies = []

            for dep in depgen(item):
                # Note, we always create an `Obj` node, even
                # if it has no source attached.
                node = Obj(name=dep)

                item.obj_dependencies.append(node)

                if node not in nodes:
                    nodes.append(node)
                    q.append(node)

                edges.append((item, node))

        # Create a nw.DiGraph from nodes/edges
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

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
        rules = rules or '*.o *.mod *.so *.a f90wrap*.f90'
        if isinstance(rules, str):
            rules = rules.split(' ')
        rules = as_tuple(rules)

        path = path or self.build_dir or Path('.')

        for r in rules:
            for f in path.glob(r):
                delete(f)

    def build(self, filename, target=None, shared=True, include_dirs=None, external_objs=None):  # pylint: disable=unused-argument
        item = self.get_item(filename)
        self.logger.info("Building %s", item)

        build_dir = str(self.build_dir) if self.build_dir else None

        # Include optional external objects in the build
        objs = [Path(o).resolve() for o in external_objs or []]

        # Build the entire dependency graph, including the source object
        dependencies = self.get_dependency_graph(item)
        for dep in reversed(list(nx.topological_sort(dependencies))):
            dep.build(compiler=self.compiler, build_dir=build_dir,
                      include_dirs=self.include_dirs)
            objs += [f'{dep.path.stem}.o']

        if target is not None:
            self.logger.info('Linking target: %s', target)
            self.compiler.link(objs=objs, target=target, cwd=build_dir)

    def load_module(self, module):
        """
        Handle import paths and load the compiled module
        """
        if str(self.build_dir.absolute()) not in sys.path:
            sys.path.insert(0, str(self.build_dir.absolute()))
        if module in sys.modules:
            reload(sys.modules[module])
            return sys.modules[module]

        try:
            # Attempt to load module directly
            return import_module(module)
        except ModuleNotFoundError:
            # If module caching interferes, try again with clean caches
            invalidate_caches()
            return import_module(module)


    def wrap_and_load(self, sources, modname=None, build=True,
                      libs=None, lib_dirs=None, incl_dirs=None,
                      kind_map=None):
        """
        Performs the necessary build steps to compile and wrap a set
        of sources using ``f90wrap``

        This method returns a dynamically loaded Python module
        containinig wrappers for each Fortran
        procedure and module specified in :data:`sources`.

        Parameters
        ==========
        source : str or list of str
            Name(s) of source files to wrap
        modname : str, optional
            Optional module name for f90wrap to use
        build : bool, optional
            Flag to force building the source first; default: True.
        libs : list of str, optional
            Override for library names to link
        lib_dirs : list of str, optional
            Override for library paths to link from
        incl_dirs : list of str, optional
            Override for include directories
        kind_map : str, optional
            Path to ``f90wrap`` KIND_MAP file, containing a Python dictionary
            in f2py_f2cmap format.
        """
        items = as_tuple(self.get_item(s) for s in as_tuple(sources))
        build_dir = str(self.build_dir) if self.build_dir else None
        modname = modname or items[0].path.stem

        # Invoke build to ensure all base objects are built
        # TODO: Could automate this via timestamps/hashes, etc.
        if build:
            for item in items:
                target = f'lib{item.path.stem}.a'
                self.build(item.path.name, target=target, shared=False)

        # Execute the first-level wrapper (f90wrap)
        self.logger.info('Python-wrapping %s', items[0])
        sourcepaths = [str(i.path) for i in items]
        self.compiler.f90wrap(modname=modname, source=sourcepaths, cwd=build_dir, kind_map=kind_map)

        # Execute the second-level wrapper (f2py-f90wrap)
        wrappers = [f'f90wrap_{item.path.stem}.f90' for item in items]
        wrappers += ['f90wrap_toplevel.f90']  # Include the generic wrapper
        wrappers = [w for w in wrappers if (self.build_dir/w).exists()]

        # Resolve final compilation libraries and include dirs
        libs = libs or [modname]
        lib_dirs = lib_dirs or [str(self.build_dir.absolute())]
        incl_dirs = incl_dirs or []

        self.compiler.f2py(modname=modname, source=wrappers,
                           libs=libs, lib_dirs=lib_dirs,
                           incl_dirs=incl_dirs, cwd=build_dir)

        self.load_module(module=modname)
