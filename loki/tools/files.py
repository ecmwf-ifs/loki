# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import re
import sys
import pickle
import shutil
import fnmatch
import tempfile
from functools import wraps
from hashlib import md5
from pathlib import Path
from importlib import import_module, reload, invalidate_caches

from loki.logging import debug, info
from loki.tools.util import as_tuple, flatten
from loki.config import config


__all__ = [
    'gettempdir', 'filehash', 'delete', 'find_paths', 'find_files',
    'disk_cached', 'load_module'
]


def gettempdir():
    """
    Create a Loki-specific tempdir in the systems temporary directory.
    """
    if config['tmp-dir']:
        tmpdir = Path(config['tmp-dir'])
    else:
        tmpdir = Path(tempfile.gettempdir())/'loki'

    if not tmpdir.exists():
        tmpdir.mkdir()

    return tmpdir


def filehash(source, prefix=None, suffix=None):
    """
    Generate a filename from a hash of ``source`` with an optional ``prefix``.
    """
    prefix = '' if prefix is None else prefix
    suffix = '' if suffix is None else suffix
    return f'{prefix}{str(md5(source.encode()).hexdigest())}{suffix}'


def delete(filename, force=False):
    filepath = Path(filename)
    debug(f'Deleting {filepath}')
    if force:
        shutil.rmtree(f'{filepath}', ignore_errors=True)
    else:
        if filepath.exists():
            os.remove(f'{filepath}')


def find_paths(directory, pattern, ignore=None, sort=True):
    """
    Utility function to generate a list of file paths based on include
    and exclude patterns applied to a root directory.

    Parameters
    ----------
    directory : str or :any:`pathlib.Path`
        Root directory from which to glob files.
    pattern : list of str
        A list of glob patterns generating files to include in the list.
    ignore : list of str, optional
        A list of glob patterns generating files to exclude from the list.
    sort : bool, optional
        Flag to indicate alphabetic ordering of files

    Returns
    -------
    list :
        The list of file names
    """
    directory = Path(directory)
    excludes = flatten(directory.rglob(e) for e in as_tuple(ignore))

    files = []
    for incl in as_tuple(pattern):
        files += [f for f in directory.rglob(incl) if f not in excludes]

    return sorted(files) if sort else files


def find_files(pattern, srcdir='.'):
    """
    Case-insensitive alternative for glob patterns that recursively
    walks all sub-directories and matches a case-insensitive regex pattern.

    Basic idea from:
    http://stackoverflow.com/questions/8151300/ignore-case-in-glob-on-linux
    """
    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    return [Path(dirpath)/fname for dirpath, _, fnames in os.walk(str(srcdir))
            for fname in fnames if rule.match(fname)]


def disk_cached(argname, suffix='cache'):
    """
    A function that creates a decorator which will cache the result of a function

    :param argname: Name of the argument that holds the filename
    """
    def decorator(fn):

        @wraps(fn)
        def cached(*args, **kwargs):
            """
            Wrapper that will cache the output of a function on disk.

            The first argument is assumed to be the name of the file
            that needs to be cached, and the cache will be put next
            to that file with the suffix ``.cache``.
            """
            filename = kwargs[argname]
            cachefile = f'{filename}.{suffix}'

            # Read cached file from disc if it's been cached before
            if config['disk-cache'] and os.path.exists(cachefile):
                # Only use cache if it is newer than the file
                filetime = os.path.getmtime(filename)
                cachetime = os.path.getmtime(cachefile)
                if cachetime >= filetime:
                    with open(cachefile, 'rb') as cachehandle:
                        info(f'Loading cache: "{cachefile}"')
                        return pickle.load(cachehandle)

            # Execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # Write to cache file
            if config['disk-cache']:
                with open(cachefile, 'wb') as cachehandle:
                    info(f'Saving cache: "{cachefile}"')
                    pickle.dump(res, cachehandle)

            return res
        return cached
    return decorator


def load_module(module, path=None):
    """
    Handle import paths and load the compiled module
    """
    if path and str(path) not in sys.path:
        sys.path.insert(0, str(path))
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
