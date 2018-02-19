import os
import pickle
from collections import Iterable

__all__ = ['flatten', 'disk_cached']


def flatten(l):
    """
    Flatten a hierarchy of nested lists into a plain list.
    """
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


def disk_cached(argname):
    """
    A function that creates a decorator which will cache the result of a function

    :param argname: Name of the argument that holds the filename
    """
    def decorator(fn):
        def wrapped(*args, **kwargs):
            """
            Wrapper that will cache the output of a function on disk.

            The first argument is assumed to be the name of the file
            that needs to be cached, and the cache will be put next
            to that file with the suffix ``.cache``.
            """
            cachefile = '%s.cache' % kwargs[argname]
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as cachehandle:
                    print("Loading cache: '%s'" % cachefile)
                    return pickle.load(cachehandle)

            # Execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # Write to cache file
            with open(cachefile, 'wb') as cachehandle:
                print("Saving cache: '%s'" % cachefile)
                pickle.dump(res, cachehandle)

            return res
        return wrapped
    return decorator
