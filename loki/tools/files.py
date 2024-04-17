# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import atexit
import fnmatch
from functools import wraps
from hashlib import md5
from importlib import import_module, reload, invalidate_caches
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile

from loki.logging import debug, info
from loki.tools.util import as_tuple, flatten
from loki.config import config


__all__ = [
    'LokiTempdir', 'gettempdir', 'filehash', 'delete', 'find_paths',
    'find_files', 'disk_cached', 'load_module',
    'write_env_launch_script', 'local_loki_setup',
    'local_loki_cleanup'
]


class LokiTempdir:
    """
    Data structure to hold an instance of :class:`tempfile.TemporaryDirectory`
    to provide a Loki-specific temporary directory that is automatically
    cleaned up when the Python interpreter is terminated

    This class provides the temporary directory creation that :any:`gettempdir`
    relies upon.
    """

    def __init__(self):
        self.tmp_dir = None
        atexit.register(self.cleanup)

    def create(self):
        """
        Create the temporary directory
        """
        if self.tmp_dir is not None:
            # The temporary directory has already been initialised
            return

        # Determine the basedir...
        if config['tmp-dir']:
            basedir = Path(config['tmp-dir'])
        else:
            basedir = Path(tempfile.gettempdir())/'loki'

        # ...and make sure it exists
        basedir.mkdir(parents=True, exist_ok=True)

        # Pick a unique prefix
        prefix = f'{os.getpid()!s}_'

        self.tmp_dir = tempfile.TemporaryDirectory(prefix=prefix, dir=basedir) # pylint: disable=consider-using-with
        debug(f'Created temporary directory {self.tmp_dir.name}')

    def get(self):
        """
        Get the temporary directory path

        Returns
        -------
        pathlib.Path
        """
        if self.tmp_dir is None:
            self.create()
        return Path(self.tmp_dir.name)

    def cleanup(self):
        """
        Clean up the temporary directory
        """
        if self.tmp_dir is not None:
            name = self.tmp_dir.name
            self.tmp_dir.cleanup()
            self.tmp_dir = None
            debug(f'Cleaned up temporary directory {name}')


TMP_DIR = LokiTempdir()
"""
An instance of :class:`LokiTempdir` representing the
temporary directory that the current Loki instance uses.
"""


def gettempdir():
    """
    Get a Loki-specific tempdir

    Throughout the lifetime of the Python interpreter process, this will always
    return the same temporary directory.

    The base directory, under which the temporary directory resides, can be
    specified by setting the environment variable ``LOKI_TMP_DIR``. Otherwise
    the platform default will be used, observing the rules specified by
    :any:`tempfile.gettempdir`.

    The temporary directory is created, managed, and cleaned up by an instance of
    :any:`LokiTempdir`. Loki will choose a process-specific temporary directory
    under the base directory to avoid race conditions between concurrently running
    Loki instances. The initialisation mechanism is lazy, creating the
    temporary directory only when this function is called for the first time.
    """
    return TMP_DIR.get()


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


def write_env_launch_script(here, binary, args):
    # Write a script to source env.sh and launch the binary
    script = Path(here/f'build/run_{binary}.sh')
    script.write_text(f"""
#!/bin/bash

source env.sh >&2
bin/{binary} {' '.join(args)}
exit $?
    """.strip())
    script.chmod(0o750)

    return script


def local_loki_setup(here):
    lokidir = Path(__file__).parent.parent.parent
    target = here/'source/loki'
    backup = here/'source/loki.bak'

    # Do not overwrite any existing Loki copy
    if target.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(target, backup)

    return str(lokidir.resolve()), target, backup


def local_loki_cleanup(target, backup):
    if target.is_symlink():
        target.unlink()
    if not target.exists() and backup.exists():
        shutil.move(backup, target)
