# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import signal
import sys
import operator as op
import weakref
from functools import lru_cache
from collections import OrderedDict
from collections.abc import Sequence
from shlex import split
from subprocess import run, PIPE, STDOUT, CalledProcessError
from contextlib import contextmanager
from pathlib import Path

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from loki.logging import debug, error


__all__ = ['as_tuple', 'is_iterable', 'is_subset', 'flatten', 'chunks',
           'execute', 'CaseInsensitiveDict', 'strip_inline_comments',
           'binary_insertion_sort', 'cached_func', 'optional', 'LazyNodeLookup',
           'yaml_include_constructor', 'auto_post_mortem_debugger', 'set_excepthook',
           'timeout']


def as_tuple(item, type=None, length=None):
    """
    Force item to a tuple, even if `None` is provided.
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
        raise ValueError(f'Tuple needs to be of length {length: d}')
    if type and not all(isinstance(i, type) for i in t):
        raise TypeError(f'Items need to be of type {type}')
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
    return True


def is_subset(a, b, ordered=True, subsequent=False):
    """
    Check if all items in iterable :data:`a` are contained in iterable :data:`b`.

    Parameters
    ----------
    a : iterable
        The iterable whose elements are searched in :data:`b`.
    b : iterable
        The iterable of which :data:`a` is tested to be a subset.
    ordered : bool, optional
        Require elements to appear in the same order in :data:`a` and :data:`b`.
    subsequent : bool, optional
        If set to `False`, then other elements are allowed to sit in :data:`b`
        in-between the elements of :data:`a`. Only relevant when using
        :data:`ordered`.

    Returns
    -------
    bool :
        `True` if all elements of :data:`a` are found in :data:`b`, `False`
        otherwise.
    """
    if not ordered:
        return set(a) <= set(b)

    if not isinstance(a, Sequence):
        raise ValueError('a is not a Sequence')
    if not isinstance(b, Sequence):
        raise ValueError('b is not a Sequence')
    if not a:
        return False

    # Search for the first element of a in b and make sure a fits in the
    # remainder of b
    try:
        idx = b.index(a[0])
    except ValueError:
        return False
    if len(a) > (len(b) - idx):
        return False

    if subsequent:
        # Now compare the sequences one by one and bail out if they don't match
        for i, j in zip(a, b[idx:]):
            if i != j:
                return False
        return True

    # When allowing intermediate elements, we search for the next element
    # in the remainder of b after the previous element
    for i in a[1:]:
        try:
            idx = b.index(i, idx+1)
        except ValueError:
            return False
    return True


def flatten(l, is_leaf=None):
    """
    Flatten a hierarchy of nested lists into a plain list.

    :param callable is_leaf: Optional function that gets called for each iterable element
                             to decide if it is to be considered as a leaf that does not
                             need further flattening.
    """
    if is_leaf is None:
        is_leaf = lambda el: False  # pylint: disable=unnecessary-lambda-assignment
    newlist = []
    for el in l:
        if is_iterable(el) and not (isinstance(el, (str, bytes)) or is_leaf(el)):
            for sub in flatten(el, is_leaf):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


def filter_ordered(elements, key=None):
    """
    Filter elements in a list while preserving order.

    :param key: Optional conversion key used during equality comparison.
    """
    seen = set()
    if key is None:
        key = lambda x: x  # pylint: disable=unnecessary-lambda-assignment
    return [e for e in elements if not (key(e) in seen or seen.add(key(e)))]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def execute(command, silent=True, **kwargs):
    """
    Execute a single command within a given directory or environment

    Parameters
    ----------
    command` : str or list of str
        The command to execute
    silent : bool, optional
        Suppress output by redirecting stdout/stderr (default: `True`)
    stdout : file object, optional
        Redirect stdout to this file object (Note: :data:`silent` overwrites this)
    stderr : file object, optional
        Redirect stdout to this file object (Note: :data:`silent` overwrites this)
    cwd : str or :class:`pathlib.Path`
        Directory in which to execute :data:`command` (will be stringified)
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
        command = split(command, posix=False)

    debug('[Loki] Executing: %s', ' '.join(command))
    try:
        return run(command, check=True, cwd=cwd, **kwargs)
    except CalledProcessError as e:
        command_str = ' '.join(command)
        error(f'Error: Execution of {command[0]} failed:')
        error(f'  Full command: {command_str}')
        output_str = ''
        if e.stdout:
            output_str += e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
        if e.stderr:
            output_str += '\n'
            output_str += e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
        if output_str:
            error(f'  Output of the command:\n\n{output_str}')
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


def binary_search(items, val, start, end, lt=op.lt):
    """
    Search for the insertion position of a value in a given
    range of items.

    :param list items: the list of items to search.
    :param val: the value for which to seek the position.
    :param int start: first index for search range in ``items``.
    :param int end: last index (inclusive) for search range in ``items``.
    :param lt: the "less than" comparison operator to use. Default is the
        standard ``<`` operator (``operator.lt``).

    :return int: the insertion position for the value.

    This implementation was adapted from
    https://www.geeksforgeeks.org/binary-insertion-sort/.
    """
    # we need to distinugish whether we should insert before or after the
    # left boundary. imagine [0] is the last step of the binary search and we
    # need to decide where to insert -1
    if start == end:
        if lt(val, items[start]):
            return start
        return start + 1

    # this occurs if we are moving beyond left's boundary meaning the
    # left boundary is the least position to find a number greater than val
    if start > end:
        return start

    pos = (start + end) // 2
    if lt(items[pos], val):
        return binary_search(items, val, pos+1, end, lt=lt)
    if lt(val, items[pos]):
        return binary_search(items, val, start, pos-1, lt=lt)
    return pos


def binary_insertion_sort(items, lt=op.lt):
    """
    Sort the given list of items using binary insertion sort.

    In the best case (already sorted) this has linear running time O(n) and
    on average and in the worst case (sorted in reverse order) a quadratic
    running time O(n*n).

    A binary search is used to find the insertion position, which reduces
    the number of required comparison operations. Hence, this sorting function
    is particularly useful when comparisons are expensive.

    :param list items: the items to be sorted.
    :param lt: the "less than" comparison operator to use. Default is the
        standard ``<`` operator (``operator.lt``).

    :return: the list of items sorted in ascending order.

    This implementation was adapted from
    https://www.geeksforgeeks.org/binary-insertion-sort/.
    """
    for i in range(1, len(items)):
        val = items[i]
        pos = binary_search(items, val, 0, i-1, lt=lt)
        items = items[:pos] + [val] + items[pos:i] + items[i+1:]
    return items


def cached_func(func):
    """
    Decorator that memoizes (caches) the result of a function
    """
    return lru_cache(maxsize=None, typed=False)(func)


@contextmanager
def optional(condition, context_manager, *args, **kwargs):
    """
    Apply the context manager only when a condition is fulfilled.

    Based on https://stackoverflow.com/a/41251962.

    Parameters
    ----------
    condition : bool
        The condition that needs to be fulfilled to apply the context manager.
    context_manager :
        The context manager to apply.
    """
    if condition:
        with context_manager(*args, **kwargs) as y:
            yield y
    else:
        yield


class LazyNodeLookup:
    """
    Utility class for indirect, :any:`weakref`-style lookups

    References to IR nodes are usually not stable as the IR may be
    rebuilt at any time. This class offers a way to refer to a node
    in an IR by encoding how it can be found instead.

    .. note::
       **Example:**
       Reference a declaration node that contains variable "a"

       .. code-block::

          from loki import LazyNodeLookup, FindNodes, Declaration
          # Assume this has been initialized before
          # routine = ...

          # Create the reference
          query = lambda x: [d for d in FindNodes(VariableDeclaration).visit(x.spec) if 'a' in d.symbols][0]
          decl_ref = LazyNodeLookup(routine, query)

          # Use the reference (this carries out the query)
          decl = decl_ref()

    Parameters
    ----------
    anchor :
        The "stable" anchor object to which :attr:`query` is applied to find the object.
        This is stored internally as a :any:`weakref`.
    query :
        A function object that accepts a single argument and should return the lookup
        result. To perform the lookup, :attr:`query` is called with :attr:`anchor`
        as argument.
    """

    def __init__(self, anchor, query):
        self._anchor = weakref.ref(anchor)
        self.query = query

    @property
    def anchor(self):
        return self._anchor()

    def __call__(self):
        return self.query(self.anchor)


def yaml_include_constructor(loader, node):
    """
    Add support for ``!include`` tags to YAML load

    Activate via ``yaml.add_constructor("!include", yaml_include_constructor)``
    or ``yaml.add_constructor("!include", yaml_include_constructor, yaml.SafeLoader)``
    (for use with ``yaml.safe_load``).

    Adapted from JUBE2 (https://fz-juelich.de/jsc/jube) and
    http://code.activestate.com/recipes/577612-yaml-include-support/

    This allows to include other YAML files or parts of them inside a YAML file:

    .. code-block:: yaml

        # include.yml
        tag0:
          foo: bar

        tag1:
          baz: bar

    .. code-block:: yaml

        # main.yml
        nested: !include include.yml

        nested_filtered: !include include.yml:["tag0"]

    which is equivalent to the following:

    ..code-block:: yaml

        nested:
          tag0:
            foo: bar
          tag1:
            baz: bar
        nested_filtered:
          baz: bar
    """
    if not HAVE_YAML:
        error('Pyyaml is not installed')
        raise RuntimeError

    # Load the content of the included file
    yaml_node_data = node.value.split(":")
    file = Path(yaml_node_data[0])
    try:
        with file.open() as inputfile:
            content = yaml.load(inputfile.read(), type(loader))
    except OSError:
        error(f'Cannot open include file {file}')
        return f'!include {node.value}'

    # Filter included content if subscripts given
    if len(yaml_node_data) > 1:
        try:
            subscripts = yaml_node_data[1].strip().lstrip('[').rstrip(']').split('][')

            for subscript in subscripts:
                if subscript.isnumeric():
                    content = content[int(subscript)]
                elif subscript[0] == subscript[-1] and subscript[0] in '"\'':
                    content = content[subscript.strip('"\'')]
                else:
                    content = content[subscript]
        except KeyError as e:
            error(f'Cannot extract {yaml_node_data[1]} from {file}')
            raise e

    return content


def auto_post_mortem_debugger(type, value, tb):  # pylint: disable=redefined-builtin
    """
    Exception hook that automatically attaches a debugger

    Activate by calling ``set_excepthook(hook=auto_post_mortem_debugger)``.

    Adapted from https://code.activestate.com/recipes/65287/
    """
    is_interactive = hasattr(sys, 'ps1')
    no_tty = not sys.stderr.isatty() or not sys.stdin.isatty() or not sys.stdout.isatty()
    if is_interactive or no_tty or type == SyntaxError:
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback # pylint: disable=import-outside-toplevel
        import pdb # pylint: disable=import-outside-toplevel
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)   # pylint: disable=no-member


def set_excepthook(hook=None):
    """
    Set an exception hook that is called for uncaught exceptions

    This can be called with :meth:`auto_post_mortem_debugger` to automatically
    attach a debugger (Pdb or, if installed, Pdb++) when exceptions occur.

    With :data:`hook` set to `None`, this will restore the default exception
    hook ``sys.__excepthook``.
    """
    if hook is None:
        sys.excepthook = sys.__excepthook__
    else:
        sys.excepthook = hook


@contextmanager
def timeout(time_in_s, message=None):
    """
    Context manager that specifies a timeout for the code section in its body

    This is implemented by installing a signal handler for :any:`signal.SIGALRM`
    and scheduling that signal for :data:`time_in_s` in the future.
    For that reason, this context manager cannot be nested.

    A value of 0 for :data:`time_in_s` will not install any timeout.

    The following example illustrates the usage, which will result in a
    :any:`RuntimeError` being raised.

    .. code-block::
       with timeout(5):
           sleep(10)

    Parameters
    ----------
    time_in_s : int
        Timeout in seconds after which to interrupt the code
    message : str
        A custom error message to use if a timeout occurs
    """
    if message is None:
        message = f"Timeout reached after {time_in_s} second(s)"

    def timeout_handler(signum, frame): # pylint: disable=unused-argument
        raise RuntimeError(message)

    if time_in_s > 0:
        handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time_in_s)
    try:
        yield
    finally:
        if time_in_s > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)
