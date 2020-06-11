import os
import time
import pickle
import tempfile
from copy import deepcopy
from functools import wraps
from pathlib import Path
from hashlib import md5


from loki.logging import log, info, INFO


__all__ = ['as_tuple', 'flatten', 'chunks', 'disk_cached', 'gettempdir', 'truncate_string',
           'JoinableStringList']


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


def is_iterable(o):
    """
    Checks if an item is truly iterable using duck typing.

    This was added because :class:`pymbolic.primitives.Expression` provide an ``__iter__`` method
    that throws an exception to avoid being iterable. However, with that method defined it is
    identified as a :class:`collections.Iterable` and thus this is a much more reliable test than
    ``isinstance(obj, collections.Iterable)``.
    """
    try:
        iter(o)
    except TypeError:
        return False
    else:
        return True


def flatten(l, is_leaf=None):
    """
    Flatten a hierarchy of nested lists into a plain list.

    :param callable is_leaf: Optional function that gets called for each iterable element
                             to decide if it is to be considered as a leaf that does not
                             need further flattening.
    """
    if is_leaf is None:
        is_leaf = lambda el: False
    newlist = []
    for el in l:
        if is_iterable(el) and not (isinstance(el, (str, bytes)) or is_leaf(el)):
            for sub in flatten(el, is_leaf):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
            if os.path.exists(cachefile):
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
            with open(cachefile, 'wb') as cachehandle:
                info("Saving cache: '%s'" % cachefile)
                pickle.dump(res, cachehandle)

            return res
        return cached
    return decorator


def timeit(log_level=INFO, argname=None):
    argname = as_tuple(argname)

    def decorator(fn):

        @wraps(fn)
        def timed(*args, **kwargs):
            ts = time.time()
            result = fn(*args, **kwargs)
            te = time.time()

            argvals = ', '.join(kwargs[arg] for arg in argname)
            log('[%s: %s] Executed in %.2fs' % (fn.__name__, argvals, (te - ts)),
                level=log_level)
            return result

        return timed
    return decorator


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


def truncate_string(string, length=16, continuation='...'):
    """
    Truncates a string to have a maximum given number of characters and indicates the
    truncation by continuation characters '...'.
    This is used, for example, in the representation strings of IR nodes.
    """
    if len(string) > length:
        return string[:length - len(continuation)] + continuation
    return string


class JoinableStringList:
    """
    Helper class that takes a list of items and joins them into a long string,
    when converting the object to a string using custom separators.
    Long lines are wrapped automatically.

    The behaviour is essentially the same as `sep.join(items)` but with the
    automatic wrapping of long lines. `items` can contain st

    :param items: the list (or tuple) of items (that can be instances of
                  `str` or `JoinableStringList`) that is to be joined.
    :type items: list or tuple
    :param str sep: the separator to be inserted between consecutive items.
    :param int width: the line width after which long lines should be wrapped.
    :param cont: the line continuation string to be inserted on a line break.
    :type cont: (str, str) or str
    :param bool separable: an indicator whether this can be split up to fill
                           lines or should stay as a unit (this is for cosmetic
                           purposes only, as too long lines will be wrapped
                           in any case).
    """

    def __init__(self, items, sep, width, cont, separable=True):
        super().__init__()

        assert is_iterable(items)
        assert isinstance(sep, str)
        if isinstance(cont, str):
            cont = cont.splitlines(keepends=True)
            if len(cont) == 1:
                cont += ['']
        assert is_iterable(cont) and len(cont) == 2
        assert all(width > len(c) for c in cont)

        self.items = [item for item in items if item is not None]
        self.sep = sep
        self.width = width
        self.cont = cont
        self.separable = separable

    def _add_item_to_line(self, line, item):
        """
        Append the given item to the line.

        :param str line: the line to which the item is appended.
        :param item: the item that is appended.
        :type item: str or `JoinableStringList`

        :return: the updated line and a list of preceeding lines that have
                 been wrapped in the process.
        :rtype: (str, list)
        """
        # Let's see if we can fit the current item plus separator
        # onto the line and have enough space left for a line break
        new_line = '{!s}{!s}'.format(line, item)
        if len(new_line) + len(self.cont[0]) <= self.width:
            return new_line, []

        # Putting the current item plus separator and potential line break
        # onto the current line exceeds the allowed width: we need to break.
        item_line = '{!s}{!s}'.format(self.cont[1], item)
        item_fits_in_line = len(item_line) + len(self.cont[0]) <= self.width

        # First, let's see if we have a JoinableStringList object that we can split up.
        # However, we'll split this up only if allowed or if the item won't fit
        # on a line
        if isinstance(item, type(self)) and (item.separable or not item_fits_in_line):
            line, new_item = item._to_str(line=line, stop_on_continuation=True)
            new_line, lines = self._add_item_to_line(self.cont[1], new_item)
            return new_line, [line + self.cont[0], *lines]

        # Otherwise, let's put it on a new line if the item as a whole fits on the next line
        if item_fits_in_line:
            return item_line, [line + self.cont[0]]

        # The new item does not fit onto a line at all and it is not a JoinableStringList
        # for which we know how to split it: let's try our best anyways
        # TODO: This is not safe for strings currently and may still exceed
        #       the line limit if the chunks are too big! Safest option would
        #       be to have expression mapper etc. all return JoinableStringList instances
        #       and accept violations for the remaining cases.
        chunk_list = str(item).split(' ')
        chunk_list[:-1] = [chunk + ' ' for chunk in chunk_list[:-1]]

        # First, add as much as possible to the previous line
        next_chunk = 0
        for idx, chunk in enumerate(chunk_list):
            new_line = line + chunk
            if len(new_line) + len(self.cont[0]) > self.width:
                next_chunk = idx
                break
            line = new_line

        # Now put the rest on new lines
        lines = []
        if line != self.cont[1]:
            lines += [line + self.cont[0]]
            line = self.cont[1]
        for chunk in chunk_list[next_chunk:]:
            new_line = line + chunk
            if len(new_line) + len(self.cont[0]) > self.width and line != self.cont[1]:
                lines += [line + self.cont[0]]
                line = self.cont[1] + chunk
            else:
                line = new_line
        return line, lines

    def _to_str(self, line='', stop_on_continuation=False):
        """
        Join all items into a long string using the given separator and wrap lines if
        necessary.

        :param str line: the line this should be appended to.
        :param bool stop_on_continuation: if True, only items up to the line width are
            appended

        :return: the joined string and a `JoinableStringList` object with the remaining
                 items, if any, or None.
        :rtype: (str, JoinableStringList or NoneType)
        """
        if not self.items:
            return '', None
        if len(self.items) == 1:
            return str(self.items[0]), None
        lines = []
        # Add all items one after another
        for idx, item in enumerate(self.items):
            if str(item) == '':
                # Skip empty items
                continue
            sep = self.sep if idx + 1 < len(self.items) else ''
            old_line = line
            line, _lines = self._add_item_to_line(line, item + sep)
            if stop_on_continuation and _lines:
                return old_line, type(self)(self.items[idx:], sep=self.sep, width=self.width,
                                            cont=self.cont, separable=self.separable)
            lines += _lines
        return ''.join([*lines, line]), None

    def __add__(self, other):
        """
        Concatenate this object and a string or another py:class:`JoinableStringList`.

        :param other: the object to append.
        :type other: str or JoinableStringList
        """
        if isinstance(other, type(self)):
            return type(self)([self, other], sep='', width=self.width, cont=self.cont,
                              separable=False)
        if isinstance(other, str):
            obj = deepcopy(self)
            if obj.items:
                obj.items[-1] += other
            else:
                obj.items = [other]
            return obj
        raise TypeError('Concatenation only for strings or items of same type.')

    def __radd__(self, other):
        """
        Concatenate a string and this object.

        :param other: the str to prepend.
        :type other: str
        """
        if isinstance(other, str):
            obj = deepcopy(self)
            if obj.items:
                obj.items[0] = other + obj.items[0]
            else:
                obj.items = [other]
            return obj
        raise TypeError('Concatenation only for strings.')

    def __str__(self):
        """
        Convert to a string.
        """
        return self._to_str()[0]
