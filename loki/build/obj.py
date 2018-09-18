import re
from pathlib import Path
from cached_property import cached_property
from fastcache import clru_cache
from collections import Iterable

from loki.build.tools import as_tuple, flatten
from loki.build.logging import _default_logger, debug
from loki.build.compiler import _default_compiler


__all__ = ['Obj']


_re_use = re.compile('^\s*use\s+(?P<use>\w+)', re.IGNORECASE | re.MULTILINE)
_re_include = re.compile('\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile('module\s+(\w+).*end module', re.IGNORECASE | re.DOTALL)
_re_subroutine = re.compile('subroutine\s+(\w+).*end subroutine', re.IGNORECASE | re.DOTALL)


@clru_cache(maxsize=1024, typed=False)
class Obj(object):
    """
    A single source object representing a single C or Fortran source file.
    """

    # Default source and header extension recognized
    # TODO: Make configurable!
    _src_ext = ['.F90', '.F']
    _h_ext = ['.h']

    def __init__(self, name, builder=None, source_dirs=None):
        self.name = name
        self.builder = builder

        self.source_path = Path(name)

        if not self.source_path.exists():
            # Given name is not a full source path
            if source_dirs is None:
                raise RuntimeError('Could not create object: %s' % name)

            src_pattern = ['**/%s%s' % (name, ext) for ext in self._src_ext]
            src_path = self.find_path(pattern=src_pattern, source_dirs=source_dirs)

            if src_path is None or isinstance(src_path, Iterable):
                debug('Could not find source file for %s' % self)
                self.source_path = None
            else:
                self.source_path = Path(src_path)

    @staticmethod
    def find_path(pattern, source_dirs=None):
        """
        Scan all source paths for source files according to glob pattern.
        """
        filepaths = flatten(list(list(s.glob(pattern)) for s in as_tuple(source_dirs))
                            for pattern in as_tuple(pattern))
        if len(filepaths) == 0:
            return None
        elif len(filepaths) == 1:
            return filepaths[0]
        else:
            return filepaths

    def __repr__(self):
        return 'Obj<%s>' % self.name

    @cached_property
    def source(self):
        if self.source_path is not None:
            with self.source_path.open() as f:
                source = f.read()
            return source
        else:
            return None

    @cached_property
    def modules(self):
        return [m.lower() for m in _re_module.findall(self.source)]

    @cached_property
    def subroutines(self):
        return [m.lower() for m in _re_subroutine.findall(self.source)]

    @cached_property
    def uses(self):
        return [m.lower() for m in _re_use.findall(self.source)]

    @cached_property
    def includes(self):
        return [m.lower() for m in _re_include.findall(self.source)]

    @property
    def dependencies(self):
        """
        Names of build items that this item depends on.
        """
        if self.source is None:
            return ()

        # For C-style header includes, drop the `.h`
        includes = [Path(incl).stem for incl in self.includes]
        # Hack: Also drop the `.intfb` part for interface blocks
        includes = [Path(incl).stem if '.intfb' in incl else incl
                    for incl in includes]
        return as_tuple(set(self.uses + includes))

    @property
    def definitions(self):
        """
        Names of provided subroutine and modules.
        """
        return as_tuple(self.modules + self.subroutines)

    def build(self, builder=None, logger=None, compiler=None):
        """
        Execute the respective build command according to the given
        :param toochain:.

        Please note that this does not build any dependencies.
        """
        logger = logger or builder.logger
        compiler = compiler or builder.compiler
        buildpath = builder.build_dir if builder else Path.cwd()
        build_dir = builder.build_dir

        use_c = self.source_path.suffix.lower() in ['.c', '.cc']
        compiler.compile(source=self.source_path.absolute(), use_c=use_c, cwd=build_dir)

    def wrap(self):
        """
        Wrap the compiled object using ``f90wrap`` and return the loaded module.
        """
        build_dir = str(self.builder.build_dir)
        compiler = self.builder.compiler or _default_compiler

        module = self.source_path.stem
        source = [str(self.source_path)]
        compiler.f90wrap(modname=module, source=source, cwd=build_dir)

        # Execute the second-level wrapper (f2py-f90wrap)
        wrapper = 'f90wrap_%s.f90' % self.source_path.stem
        if self.modules is None or len(self.modules) == 0:
            wrapper = 'f90wrap_toplevel.f90'
        compiler.f2py(modname=module, source=[wrapper, '%s.o' % self.source_path.stem],
                       cwd=build_dir)

        return self.builder.load_module(module)
