from pathlib import Path
import re
from collections import Iterable
from cachetools import cached, LRUCache

from loki.build.tools import as_tuple


__all__ = ['Builder', 'BuildUnit']


_re_use = re.compile('use\s+(?P<use>\w+)', re.IGNORECASE)
_re_include = re.compile('\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile('module\s+(\w+).*end module', re.IGNORECASE|re.DOTALL)
_re_subroutine = re.compile('subroutine\s+(\w+).*end subroutine', re.IGNORECASE|re.DOTALL)


class BuildUnit(object):
    """
    A simple representation of a source file that provides quick
    access to dependencies and provided modules/subroutines.

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
        return 'BU<%s>' % self.path.name


class Builder(object):
    """
    A :class:`Builder` that compiles binaries or libraries, while performing
    automated dependency discovery from one or more source paths.

    :param sources: One or more paths to search for source files
    :param includes: One or more paths to that include header files
    """

    def __init__(self, sources, includes=None):
        self.sources = [Path(p) for p in as_tuple(sources)]
        self.includes = [Path(p) for p in as_tuple(includes)]

    @cached(LRUCache(maxsize=1024))
    def get_unit(self, filename):
        """
        Scan a source path for potential build units and populate cache.

        :param path: A :class:`pathlib.Path` object for which to retrieve the :class:`BuildUnit`
        """
        filepath = Path(filename)
        for s in self.sources:
            filepaths = [BuildUnit(fp) for fp in s.glob('**/%s' % filename)]
            if len(filepaths) == 0:
                return None
            elif len(filepaths) == 1:
                return filepaths[0]
            else:
                return filepaths
