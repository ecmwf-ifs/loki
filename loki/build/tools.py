import os
import shutil
from subprocess import run, PIPE, STDOUT, CalledProcessError
from collections.abc import Iterable
from pathlib import Path
from fastcache import clru_cache

from loki.build.logging import debug, error


__all__ = ['execute', 'delete', 'as_tuple', 'filter_ordered', 'flatten']


def execute(args, cwd=None, env=None):
    debug('Executing: %s', ' '.join(args))
    cwd = cwd if cwd is None else str(cwd)
    try:
        run(args, check=True, stdout=PIPE, stderr=STDOUT, cwd=cwd, env=env)
    except CalledProcessError as e:
        error('Execution failed with:')
        error(e.output.decode("utf-8"))
        raise e


def delete(filename, force=False):
    filepath = Path(filename)
    debug('Deleting %s', filepath)
    if force:
        shutil.rmtree('%s' % filepath, ignore_errors=True)
    else:
        if filepath.exists():
            os.remove('%s' % filepath)


def as_tuple(item, type=None, length=None):
    """
    Force item to a tuple.

    Partly extracted from: https://github.com/OP2/PyOP2/.
    """
    # Stop complaints about `type` in this function
    # pylint: disable=redefined-builtin

    # Empty list if we get passed None
    if item is None:
        t = ()
    elif isinstance(item, str):
        t = (item,)
    else:
        # Convert iterable to list...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except (TypeError, NotImplementedError):
            t = (item,) * (length or 1)
    if length and not len(t) == length:
        raise ValueError("Tuple needs to be of length %d" % length)
    if type and not all(isinstance(i, type) for i in t):
        raise TypeError("Items need to be of type %s" % type)
    return t


def filter_ordered(elements, key=None):
    """
    Filter elements in a list while preserving order.

    Partly extracted from: https://github.com/opesci/devito.

    :param key: Optional conversion key used during equality comparison.
    """
    seen = set()
    if key is None:
        key = lambda x: x
    return [e for e in elements if not (key(e) in seen or seen.add(key(e)))]


def flatten(l):
    """
    Flatten a hierarchy of nested lists into a plain list.

    Partly extracted from: https://github.com/opesci/devito.
    """
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


def find_paths(directory, pattern, ignore=None, sort=True):
    """
    Utility function to generate a list of file paths based on include
    and exclude patterns applied to a root directory.

    :param root: Root director from which to glob files.
    :param includes: A glob pattern generating files to include in the list.
    :param excludes: A glob pattern generating files to exclude from the list.
    """
    directory = Path(directory)
    excludes = flatten(directory.rglob(e) for e in as_tuple(ignore))

    files = []
    for incl in as_tuple(pattern):
        files += [f for f in directory.rglob(incl) if f not in excludes]

    return sorted(files) if sort else files


def cached_func(func):
    return clru_cache(maxsize=None, typed=False, unhashable='ignore')(func)
