import os
import pickle
from collections import Iterable

from ecir.helpers import assemble_continued_statement_from_list

__all__ = ['flatten', 'chunks', 'disk_cached', 'as_tuple', 'extract_lines']


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


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
            filename = kwargs[argname]
            cachefile = '%s.cache' % filename
            if os.path.exists(cachefile):
                # Only use cache if it is newer than the file
                filetime = os.path.getmtime(filename)
                cachetime = os.path.getmtime(cachefile)
                if cachetime >= filetime:
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


def as_tuple(item, type=None, length=None):
    """
    Force item to a tuple.

    Partly extracted from: https://github.com/OP2/PyOP2/.
    """
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


def extract_lines(ast, source, full_lines=False):
    """
    Extract the marked string from source text.
    """
    attrib = ast.attrib if hasattr(ast, 'attrib') else ast
    lstart = int(attrib['line_begin'])
    lend = int(attrib['line_end'])
    cstart = int(attrib['col_begin'])
    cend = int(attrib['col_end'])

    source = source.splitlines(keepends=True)

    if full_lines:
        return ''.join(source[lstart-1:lend])

    if lstart == lend:
        lines = [source[lstart-1][cstart:cend]]
    else:
        lines = source[lstart-1:lend]
        firstline = lines[0][cstart:]
        lastline = lines[-1][:cend]
        lines = [firstline] + lines[1:-1] + [lastline]


    # Scan for line continuations and honour inline
    # comments in between continued lines
    def continued(line):
        return line.strip().endswith('&')
    def is_comment(line):
        return line.strip().startswith('!')

    # We only honour line continuation if we're not parsing a comment
    if not is_comment(lines[-1]):
        while continued(lines[-1]) or is_comment(lines[-1]):
            lend += 1
            lines.append(source[lend-1])

    return ''.join(lines)
