import time
from functools import wraps
from collections import OrderedDict
from subprocess import run, PIPE, STDOUT, CalledProcessError

from loki.logging import log, debug, error, INFO


__all__ = ['as_tuple', 'is_iterable', 'flatten', 'chunks', 'timeit',
           'execute', 'CaseInsensitiveDict', 'strip_inline_comments']


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


def timeit(log_level=INFO, getter=None):
    """
    Timing decorator that logs the time taken in a specific function call.

    Parameters
    ==========
    * ``log_level``: The lvel at which to log the resulting time
    * ``getter``: (List of) lambda function to extract additinal strings for
                  the log message. Each getter will be invoked on ``**kwargs``.
    """
    getter = as_tuple(getter)

    def decorator(fn):

        @wraps(fn)
        def timed(*args, **kwargs):
            ts = time.time()
            result = fn(*args, **kwargs)
            te = time.time()

            argvals = ', '.join(g(kwargs) for g in getter)
            log('[Loki::%s: %s] Executed in %.2fs' % (fn.__name__, argvals, (te - ts)),
                level=log_level)
            return result

        return timed
    return decorator


def execute(command, silent=True, **kwargs):
    """
    Execute a single command within a given director or envrionment.

    Parameters:
    ===========
    ``command``: String or list of strings with the command to execute
    ``silent``: Silences output by redirecting stdout/stderr (default: ``True``)
    ``cwd`` Directory in which to execute command (will be stringified)
    """

    cwd = kwargs.pop('cwd', None)
    cwd = cwd if cwd is None else str(cwd)

    if silent:
        kwargs['stdout'] = kwargs.pop('stdout', PIPE)
        kwargs['stderr'] = kwargs.pop('stderr', STDOUT)

    # Some string mangling to support lists and strings
    if isinstance(command, list):
        command = ' '.join(command)
    if isinstance(command, str):
        command = command.split(' ')

    debug('[Loki] Executing: %s', ' '.join(command))
    try:
        return run(command, check=True, cwd=cwd, **kwargs)
    except CalledProcessError as e:
        error('Execution failed with:')
        error(str(e.output))
        raise e


class CaseInsensitiveDict(OrderedDict):
    """
    Dict that ignores the casing of string keys.

    Basic idea from:
    https://stackoverflow.com/questions/2082152/case-insensitive-dictionary
    """
    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def get(self, key, default=None):
        return super().get(key.lower(), default)

    def __contains__(self, key):
        return super().__contains__(key.lower())


def strip_inline_comments(source, comment_char='!', str_delim='"\''):
    """
    Strip inline comments from a source string and return the modified string.

    Note: this does only work reliably for Fortran strings at the moment (where quotation
    marks are escaped by double quotes and thus the string status is kept correct automatically).

    :param str source: the source line(s) to be stripped.
    :param str comment_char: the character that marks the beginning of a comment.
    :param str str_delim: one or multiple characters that are valid string delimiters.
    """
    if comment_char not in source:
        # No comment, we can bail out early
        return source

    # Split the string into lines and look for the start of comments
    source_lines = source.splitlines()

    def update_str_delim(open_str_delim, string):
        """Run through the string and update the string status."""
        for ch in string:
            if ch in str_delim:
                if open_str_delim == '':
                    # This opens a string
                    open_str_delim = ch
                elif open_str_delim == ch:
                    # TODO: Handle escaping of quotes in general. Fortran just works (TM)
                    # This closes a string
                    open_str_delim = ''
                # else: character is string delimiter but we are inside an open string
                # with a different character used => ignored
        return open_str_delim

    # If we are inside a string this holds the delimiter character that was used
    # to open the current string environment:
    #  '': if not inside a string
    #  'x':  inside a string with x being the opening string delimiter
    open_str_delim = ''

    # Run through lines to strip inline comments
    clean_lines = []
    for line in source_lines:
        end = line.find(comment_char)
        open_str_delim = update_str_delim(open_str_delim, line[:end])

        while end != -1:
            if not open_str_delim:
                # We have found the start of the inline comment, add the line up until there
                clean_lines += [line[:end].rstrip()]
                break
            # We are inside an open string, idx does not mark the start of a comment
            start, end = end, line.find(comment_char, end + 1)
            open_str_delim = update_str_delim(open_str_delim, line[start:end])
        else:
            # No comment char found in current line, keep original line
            clean_lines += [line]
            open_str_delim = update_str_delim(open_str_delim, line[end:])

    return '\n'.join(clean_lines)
