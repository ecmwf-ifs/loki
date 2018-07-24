from pathlib import Path
import re
from cachetools import cached, LRUCache

from loki.build.tools import as_tuple
from loki.build.compiler import execute
from loki.build.toolchain import _default_toolchain


__all__ = ['Builder', 'BuildUnit']


_re_use = re.compile('use\s+(?P<use>\w+)', re.IGNORECASE)
_re_include = re.compile('\#include\s+["\']([\w\.]+)[\"\']', re.IGNORECASE)
# Please note that the below regexes are fairly expensive due to .* with re.DOTALL
_re_module = re.compile('module\s+(\w+).*end module', re.IGNORECASE | re.DOTALL)
_re_subroutine = re.compile('subroutine\s+(\w+).*end subroutine', re.IGNORECASE | re.DOTALL)


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

    @cached(LRUCache(maxsize=1024))
    def get_unit(self, filename):
        """
        Scan all source paths for source files and create a :class:`BuildUnit`.

        :param filename: Name of the source file we are looking for.
        """
        for s in self.source_dirs:
            filepaths = [BuildUnit(fp) for fp in s.glob('**/%s' % filename)]
            if len(filepaths) == 0:
                return None
            elif len(filepaths) == 1:
                return filepaths[0]
            else:
                return filepaths

    def __getitem__(self, *args, **kwargs):
        return self.get_unit(*args, **kwargs)

    def build(self, filename, target=None):
        unit = self.get_unit(filename)
        args = self.toolchain.build_args(source=unit.path.absolute(),
                                         include_dirs=self.include_dirs,
                                         target=target)

        # TODO: Need better way to deal with build dirs
        build_dir = str(self.build_dir) if self.build_dir else None
        execute(args, cwd=build_dir)
