import re
from pathlib import Path

try:
    from functools import cached_property
except ImportError:
    try:
        from cached_property import cached_property
    except ImportError:
        def cached_property(func):
            return func

from loki.logging import debug
from loki.tools import cached_func


__all__ = ['Header']


_re_use = re.compile(r'^\s*use\s+(?P<use>\w+)', re.IGNORECASE | re.MULTILINE)
_re_include = re.compile(r'\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile(r'module\s+(\w+).*end module', re.IGNORECASE | re.DOTALL)
_re_subroutine = re.compile(r'subroutine\s+(\w+).*end subroutine', re.IGNORECASE | re.DOTALL)


class Header:

    _ext = ['.intfb.h', '.h']

    def __new__(cls, *args, name=None, **kwargs):  # pylint: disable=unused-argument
        # Name is either provided or inferred from source_path
        name = name or Path(kwargs.get('source_path')).stem
        # Hack: Remove the .intfb from the name
        if 'intfb' in name:
            name = Path(name).stem
        name = name.lower()

        # Return an instance cached on the derived or provided name
        # TODO: We could make the path relative to a "cache path" here...
        return Header.__xnew_cached_(cls, name)

    def __new_stage2_(self, name):
        obj = super().__new__(self)
        obj.name = name
        return obj

    __xnew_cached_ = staticmethod(cached_func(__new_stage2_))

    def __init__(self, name=None, source_path=None):  # pylint: disable=unused-argument
        if not hasattr(self, 'source_path'):
            # If this is the first time, establish the source path
            self.source_path = Path(source_path or self.name)  # pylint: disable=no-member

            if not self.source_path.exists():
                debug('Could not find source file for %s', self)
                self.source_path = None

    def __repr__(self):
        return f'Header<{self.name}>'  # pylint: disable=no-member

    @cached_property
    def source(self):
        if self.source_path is not None:
            # TODO: Make encoding a global config item.
            with self.source_path.open(encoding='latin1') as f:
                source = f.read()
            return source
        return None

    @cached_property
    def uses(self):
        if self.source is None:
            return []
        return [m.lower() for m in _re_use.findall(self.source)]

    @cached_property
    def includes(self):
        return [m.lower() for m in _re_include.findall(self.source)]
