import os
import re
import pickle
import fnmatch
import tempfile
from functools import wraps
from hashlib import md5
from pathlib import Path

from loki.logging import info
from loki.config import config


__all__ = ['gettempdir', 'filehash', 'find_files', 'disk_cached']


def gettempdir():
    """
    Create a Loki-specific tempdir in the systems temporary directory.
    """
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
    return '%s%s%s' % (prefix, str(md5(source.encode()).hexdigest()), suffix)


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
            cachefile = '%s.%s' % (filename, suffix)

            # Read cached file from disc if it's been cached before
            if config['disk-cache'] and os.path.exists(cachefile):
                # Only use cache if it is newer than the file
                filetime = os.path.getmtime(filename)
                cachetime = os.path.getmtime(cachefile)
                if cachetime >= filetime:
                    with open(cachefile, 'rb') as cachehandle:
                        info("Loading cache: '%s'" % cachefile)
                        return pickle.load(cachehandle)

            # Execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # Write to cache file
            if config['disk-cache']:
                with open(cachefile, 'wb') as cachehandle:
                    info("Saving cache: '%s'" % cachefile)
                    pickle.dump(res, cachehandle)

            return res
        return cached
    return decorator
