import sys
import re
from abc import ABCMeta, abstractproperty
from pathlib import Path
from functools import lru_cache
import networkx as nx
from collections import deque, OrderedDict
from importlib import import_module

from loki.logging import info, debug  # The only upwards dependency!

from loki.build.tools import as_tuple
from loki.build.compiler import delete
from loki.build.toolchain import _default_toolchain


__all__ = ['Builder', 'BuildItem']


_re_use = re.compile('use\s+(?P<use>\w+)', re.IGNORECASE)
_re_include = re.compile('\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile('module\s+(\w+).*end module', re.IGNORECASE | re.DOTALL)
_re_subroutine = re.compile('subroutine\s+(\w+).*end subroutine', re.IGNORECASE | re.DOTALL)


class BuildItem(object):
    """
    Abstract representation of a compilable item (source file or library)
    that provides quick access to dependencies and provided definitions.

    A :class:`BuildItem` is used to automtically establish dependency trees.
    """

    __metaclass__ = ABCMeta

    @abstractproperty
    def dependencies(self):
        """
        List of tuple of :class:`BuildItem`s that are required to build this item.
        """
        pass

    @abstractproperty
    def definitions(self):
        """
        List of tuple of symbols defined by this build item.
        """
        pass


class Obj(BuildItem):
    """
    A single source object representing a single C or Fortran source file.
    """

    def __init__(self, filename, builder=None):
        self.path = Path(filename)
        self.builder = builder

        with self.path.open() as f:
            source = f.read()

        self.modules = [m.lower() for m in _re_module.findall(source)]
        self.subroutines = [m.lower() for m in _re_subroutine.findall(source)]

        self.uses = [m.lower() for m in _re_use.findall(source)]
        self.includes = [m.lower() for m in _re_include.findall(source)]

    def __repr__(self):
        return 'Obj<%s>' % self.path.name

    @property
    def dependencies(self):
        """
        Names of build items that this item depends on.
        """
        uses = ['%s.F90' % u for u in self.uses]
        includes = [Path(incl).stem for incl in self.includes]
        includes = [Path(incl).stem if '.intfb' in incl else incl
                    for incl in includes]
        includes = ['%s.F90' % incl for incl in includes]
        return as_tuple(uses + includes)

    @property
    def definitions(self):
        """
        Names of provided subroutine and modules.
        """
        return as_tuple(self.modules + self.subroutines)

    def build(self):
        """
        Execute the respective build command according to the given
        :param toochain:.

        Please note that this does not build any dependencies.
        """
        build_dir = str(self.builder.build_dir)
        toolchain = self.builder.toolchain or _default_toolchain

        debug('Building obj %s' % self)
        use_c = self.path.suffix.lower() in ['.c', '.cc']
        toolchain.build(source=self.path.absolute(), use_c=use_c, cwd=build_dir)

    def wrap(self):
        """
        Wrap the compiled object using ``f90wrap`` and return the loaded module.
        """
        build_dir = str(self.builder.build_dir)
        toolchain = self.builder.toolchain or _default_toolchain

        module = self.path.stem
        source = [str(self.path)]
        toolchain.f90wrap(modname=module, source=source, cwd=build_dir)

        # Execute the second-level wrapper (f2py-f90wrap)
        wrapper = 'f90wrap_%s.f90' % self.path.stem
        if self.modules is None or len(self.modules) == 0:
            wrapper = 'f90wrap_toplevel.f90'
        toolchain.f2py(modname=module, source=[wrapper, '%s.o' % self.path.stem],
                       cwd=build_dir)

        return self.builder.load_module(module)


class Lib(BuildItem):
    """
    A library linked from multiple objects.
    """

    def __init__(self, name, objects=None, builder=None):
        self.name = name
        self.path = Path('lib%s.so' % name)
        self.builder = builder
        self.objects = objects or []

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

        debug('Building lib %s' % self)
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


class Builder(object):
    """
    A :class:`Builder` that compiles binaries or libraries, while performing
    automated dependency discovery from one or more source paths.

    :param sources: One or more paths to search for source files
    :param includes: One or more paths to that include header files
    """

    def __init__(self, source_dirs, include_dirs=None, root_dir=None,
                 build_dir=None, toolchain=None):
        # TODO: Make configurable and supply more presets
        self.toolchain = toolchain or _default_toolchain

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

    def find_path(self, filename):
        """
        Scan all source paths for source files and create a :class:`BuildItem`.

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

    @lru_cache(maxsize=None)
    def Obj(self, filename):
        path = self.find_path(filename)
        if path is None:
            raise RuntimeError('Could not establish path for %s' % filename)
        return Obj(filename=path, builder=self)

    def Lib(self, name, objects=None):
        objs = [o if isinstance(o, Obj) else self.Obj(o)
                for o in as_tuple(objects)]
        return Lib(name, objects=objs, builder=self)

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
            self.toolchain.link(objs=objs, target=target, cwd=build_dir)

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
        info('Python-wrapping %s' % items[0])
        sourcepaths = [str(i.path) for i in items]
        self.toolchain.f90wrap(modname=modname, source=sourcepaths, cwd=build_dir)

        # Execute the second-level wrapper (f2py-f90wrap)
        wrappers = ['f90wrap_%s.f90' % item.path.stem for item in items]
        wrappers += ['f90wrap_toplevel.f90']  # Include the generic wrapper
        wrappers = [w for w in wrappers if (self.build_dir/w).exists()]

        # Resolve final compilation libraries and include dirs
        libs = libs or [modname]
        lib_dirs = lib_dirs or ['%s' % self.build_dir.absolute()]
        incl_dirs = incl_dirs or []

        self.toolchain.f2py(modname=modname, source=wrappers,
                            libs=libs, lib_dirs=lib_dirs,
                            incl_dirs=incl_dirs, cwd=build_dir)

        self.load_module(module=modname)
