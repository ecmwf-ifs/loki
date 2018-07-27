import sys
import re
from pathlib import Path
from functools import lru_cache
import networkx as nx
from collections import deque, OrderedDict
from importlib import import_module

from loki.logging import info, debug  # The only upwards dependency!

from loki.build.tools import as_tuple
from loki.build.compiler import execute
from loki.build.toolchain import _default_toolchain


__all__ = ['Builder', 'BuildItem']


_re_use = re.compile('use\s+(?P<use>\w+)', re.IGNORECASE)
_re_include = re.compile('\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile('module\s+(\w+).*end module', re.IGNORECASE | re.DOTALL)
_re_subroutine = re.compile('subroutine\s+(\w+).*end subroutine', re.IGNORECASE | re.DOTALL)


class BuildItem(object):
    """
    A simple representation of a compilable item (source file) that
    provides quick access to dependencies and provided definitions.
    """

    def __init__(self, filename):
        self.path = Path(filename)

        with self.path.open() as f:
            source = f.read()

        self.modules = [m.lower() for m in _re_module.findall(source)]
        self.subroutines = [m.lower() for m in _re_subroutine.findall(source)]

        self.uses = [m.lower() for m in _re_use.findall(source)]
        self.includes = [m.lower() for m in _re_include.findall(source)]

    def __repr__(self):
        return '<%s>' % self.path.name

    @property
    def dependencies(self):
        """
        Names of build items that this item depends on.
        """
        return as_tuple('%s.F90' % u for u in self.uses)

    def build(self, toolchain, include_dirs=None, build_dir=None):
        """
        Execute the respective build command according to the given
        :param toochain:.

        Please note that this does not build any dependencies. For this
        a :class:`Builder` is requried.
        """
        debug('Building item %s' % self)
        args = toolchain.build_args(source=self.path.absolute(),
                                    include_dirs=include_dirs)
        execute(args, cwd=build_dir)


class Builder(object):
    """
    A :class:`Builder` that compiles binaries or libraries, while performing
    automated dependency discovery from one or more source paths.

    :param sources: One or more paths to search for source files
    :param includes: One or more paths to that include header files
    """

    def __init__(self, source_dirs, include_dirs=None, root_dir=None, build_dir=None):
        # TODO: Make configurable and supply more presets
        self.toolchain = _default_toolchain

        # Source dirs for auto-detection and include dis for preprocessing
        self.source_dirs = [Path(p).resolve() for p in as_tuple(source_dirs)]
        self.include_dirs = [Path(p).resolve() for p in as_tuple(include_dirs)]

        # Root and source directories for out-of source builds
        self.root_dir = None if root_dir is None else Path(root_dir)
        self.build_dir = None if build_dir is None else Path(build_dir)
        self.build_dir.mkdir(exist_ok=True)

        # Create the dependency graph and it's utilities
        self.dependency_graph = nx.DiGraph()
        self._cache = OrderedDict()

    @lru_cache(maxsize=None)
    def get_item(self, filename):
        """
        Scan all source paths for source files and create a :class:`BuildItem`.

        :param filename: Name of the source file we are looking for.
        """
        for s in self.source_dirs:
            filepaths = [BuildItem(fp) for fp in s.glob('**/%s' % filename)]
            if len(filepaths) == 0:
                return None
            elif len(filepaths) == 1:
                return filepaths[0]
            else:
                return filepaths

    def __getitem__(self, *args, **kwargs):
        return self.get_item(*args, **kwargs)

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

    def build(self, filename, target=None):
        item = self.get_item(filename)
        info("Building %s" % item)

        build_dir = str(self.build_dir) if self.build_dir else None

        # Build the entire dependency graph, including the source object
        objs = []
        dependencies = self.get_dependency_graph(item)
        for dep in reversed(list(nx.topological_sort(dependencies))):
            dep.build(toolchain=self.toolchain, build_dir=build_dir,
                      include_dirs=self.include_dirs)
            objs += ['%s.o' % dep.path.stem]

        if target is not None:
            debug('Linking target: %s' % target)
            args = self.toolchain.linker_args(objs=objs, target=target)
            execute(args, cwd=build_dir)

    def compile_and_load(self, filename):
        """
        Performs the necessary build steps to compile the source and
        wrappers for the :param filename: and load it dynamically.
        """
        item = self.get_item(filename)
        target = 'lib%s.so' % item.path.stem
        build_dir = str(self.build_dir) if self.build_dir else None

        # First, ensure all base objects are built
        self.build(filename, target=target)

        # Execute the first-level wrapper (f90wrap)
        info('Python-wrapping %s' % item)
        modname = item.path.stem
        f90wrap_args = self.toolchain.f90wrap_args(modname=modname,
                                                   source=str(item.path))
        execute(f90wrap_args, cwd=build_dir)

        pywrapper = (self.build_dir/('f90wrap_%s.f90' % item.path.stem))
        source = pywrapper if pywrapper.exists() else 'f90wrap_toplevel.f90'
        lib_dirs = ['%s' % self.build_dir.absolute()]
        f2py_args = self.toolchain.f2py_args(modname=modname, source=source,
                                             libs=[modname], lib_dirs=lib_dirs)
        execute(f2py_args, cwd=build_dir)

        # Handle import paths and load the compiled module
        if str(self.build_dir.absolute()) not in sys.path:
            sys.path.insert(0, str(self.build_dir.absolute()))
        return import_module(item.path.stem)
