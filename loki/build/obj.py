import re
from pathlib import Path
from cached_property import cached_property
from fastcache import clru_cache
from collections import Iterable

from loki.build.tools import as_tuple, flatten, execute
from loki.build.logging import _default_logger, debug
from loki.build.compiler import _default_compiler


__all__ = ['Obj']


_re_use = re.compile('^\s*use\s+(?P<use>\w+)', re.IGNORECASE | re.MULTILINE)
_re_include = re.compile('\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile('module\s+(\w+).*end module', re.IGNORECASE | re.DOTALL)
_re_subroutine = re.compile('subroutine\s+(\w+).*end subroutine', re.IGNORECASE | re.DOTALL)


def cached_func(func):
    return clru_cache(maxsize=None, typed=False, unhashable='ignore')(func)


class Obj(object):
    """
    A single source object representing a single C or Fortran source file.
    """

    # Default source and header extension recognized
    # TODO: Make configurable!
    _src_ext = ['.F90', '.F']
    _h_ext = ['.h']

    def __new__(cls, *args, name=None, source_dir=None, **kwargs):
        # Name is either provided or inferred from source_path
        name = name or Path(kwargs.get('source_path')).stem

        # Return an instance cached on the derived or provided name
        # TODO: We could make the path relative to a "cache path" here...
        return Obj.__xnew_cached_(cls, name)

    def __new_stage2_(cls, name):
        obj = super(Obj, cls).__new__(cls)
        obj.name = name
        return obj

    __xnew_cached_ = staticmethod(cached_func(__new_stage2_))

    def __init__(self, name=None, source_path=None, builder=None, source_dirs=None):
        self.builder = builder
        self.q_task = None

        if not hasattr(self, 'source_path'):
            # If this is the first time, establish the source path
            self.source_path = Path(source_path or self.name)

            if not self.source_path.exists():
                debug('Could not find source file for %s' % self)
                self.source_path = None

    def __repr__(self):
        return 'Obj<%s>' % self.name

    @cached_property
    def source(self):
        if self.source_path is not None:
            # TODO: Make encoding a global config item.
            with self.source_path.open(encoding='latin1') as f:
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
        if self.source is None:
            return []
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

    def build(self, builder=None, logger=None, compiler=None, workqueue=None):
        """
        Execute the respective build command according to the given
        :param toochain:.

        Please note that this does not build any dependencies.
        """
        logger = logger or builder.logger
        compiler = compiler or builder.compiler
        build_dir = builder.build_dir
        include_dirs = builder.include_dirs if builder else None

        if self.source_path is None:
            raise RuntimeError('No source file found for %s' % self)

        use_c = self.source_path.suffix.lower() in ['.c', '.cc']
        source = self.source_path.absolute()
        target = (build_dir/self.name).with_suffix('.o')
        args = compiler.compile_args(source=source, include_dirs=include_dirs,
                                     use_c=use_c, target=target, mod_dir=build_dir)
        if workqueue is not None:
            self.q_task = workqueue.execute(args)
        else:
            execute(args)

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
