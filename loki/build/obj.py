from pathlib import Path
import re

from loki.build.tools import as_tuple
from loki.build.logging import _default_logger
from loki.build.toolchain import _default_toolchain


__all__ = ['Obj']


_re_use = re.compile('use\s+(?P<use>\w+)', re.IGNORECASE)
_re_include = re.compile('\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile('module\s+(\w+).*end module', re.IGNORECASE | re.DOTALL)
_re_subroutine = re.compile('subroutine\s+(\w+).*end subroutine', re.IGNORECASE | re.DOTALL)


class Obj(object):
    """
    A single source object representing a single C or Fortran source file.
    """

    def __init__(self, filename, builder=None, logger=None):
        self.path = Path(filename)
        self.builder = builder
        self.logger = logger or _default_logger

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

        self.logger.debug('Building obj %s' % self)
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
