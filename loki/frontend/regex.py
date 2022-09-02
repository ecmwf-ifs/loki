"""
The implementation of a regex parser frontend

This is intended to allow for fast, partial extraction of IR objects
from Fortran source files without the need to generate a complete
parse tree.
"""

import re

from loki import ir
from loki.frontend import Source
from loki.logging import PERF
from loki.tools import timeit, as_tuple

__all__ = ['parse_regex_source', 'HAVE_REGEX']


HAVE_REGEX = True
"""Indicate that the regex frontend is available."""


@timeit(log_level=PERF)
def parse_regex_source(source, scope=None):
    """
    Generate a reduced Loki IR from regex parsing of the given Fortran source

    Currently, this only extracts :any:`Module` and :any:`Subroutine` objects.
    Any non-matched source code snippets are retained as :any:`RawSource` objects.

    Parameters
    ----------
    source : str or :any:`Source`
        The raw source string
    scope : :any:`Scope`, optional
        The enclosing parent scope
    """
    candidates = (match_module, match_subroutine_function)
    if not isinstance(source, Source):
        lines = (1, source.count('\n') + 1)
        source = Source(lines=lines, string=source)
    ir_ = match_block_candidates(source, candidates, scope=scope)
    return ir.Section(body=as_tuple(ir_), source=source)


def match_block_candidates(source, candidates, scope=None):
    """
    Apply match functions to :data:`source` recursively

    Parameters
    ----------
    source : :any:`Source`
        A source file object with the string to match
    candidates : list
        The list of candidate match functions to call
    scope : :any:`Scope`
        The parent scope for this source code snippet
    """
    blocks = []
    for idx, candidate in enumerate(candidates):
        while source.string:
            pre, match, source = candidate(source, scope=scope)
            if not match:
                assert pre is None
                break
            if pre:
                # See if any of the other candidates match before this match
                blocks += match_block_candidates(pre, candidates[idx+1:], scope=scope)
            blocks += [match]
    if source.string.strip():
        blocks += [ir.RawSource(text=source.string, source=source)]
    return blocks


_whitespace_comment_lineend_pattern = r'(?:[ \t]*|[ \t]*\![\S \t]*)$\n'
"""Helper pattern capturing the common case of a comment until line end with optional leading white space."""

_comment_pattern = re.compile(_whitespace_comment_lineend_pattern, re.MULTILINE)
"""Compiled version of :any:`_whitespace_comment_lineend_pattern`."""

_contains_pattern = (
    r'(?P<contains>(?P<contains_keyword>^[ \t]*?contains'  # Optional contains keyword
) + _whitespace_comment_lineend_pattern + ( # optional whitespace/comment until line end
    r')(?P<contains_body>(?:'  # Group around a contained function/subroutine
    r'(?:^'
) + _whitespace_comment_lineend_pattern + ( # Allow for multiple empty lines/comments before subroutines
    r')*'
    r'^[ \t\w()]*?' # parameter/white space/attribute banter before subroutine/function keyword
    r'(?:subroutine\s\w+.*?end\s+subroutine|function\s\w+.*?end\s+function)' # keyword to end keyword
    r'(?:\s+\w+)?'  # optionally repeated routine name
) + _whitespace_comment_lineend_pattern + ( # optional whitespace/comment until line end
    r'.*?'  # Allow for arbitrary characters between subroutines/functions
    r')*?)'  # End group around contained function/subroutine (0+ times repeated)
    r')?'  # End optional contains sections
)
"""Helper pattern capturing the ``contains`` part of a module or subroutine/function."""

_re_module = re.compile(
    (
        r'^(?:[ \t]*?)'  # Whitespace before module keyword
        r'module\s+(?P<name>\w+)[\S \t]*$'  # Module keyword and module name
        r'(?P<spec>.*?)'  # Module spec
    ) + _contains_pattern + (  # Module body (`contains`` section)
        r'end\s+module(?:\s*(?P=name))?'  # End keyword with optionally module name repeated after end keyword
    ) + _whitespace_comment_lineend_pattern + ( # optional whitespace/comment until line end
        r'?' # ...with the '\n' optional
    ),
    re.IGNORECASE | re.DOTALL | re.MULTILINE
)
"""Pattern to match module definitions."""
print(_re_module.pattern)

_re_subroutine_function = re.compile(
    (
        r'^(?:[ \t\w()]*?)'  # Whitespace and subroutine/function attributes before subroutine/function keyword
        r'(?P<keyword>subroutine|function)\s+(?P<name>\w+)'  # Subroutine/function keyword and name
        r'(?P<args>\s*\((?:(?:\s|\![\S \t]*$)*\w+\s*,?)+\))?[\S \t]*$' # Arguments
        r'(?P<spec>.*?)'  # Spec and body of routine
    ) + _contains_pattern + (  # Contains section
        r'end\s+(?P=keyword)(?:\s*(?P=name))?'  # End keyword with optionally routine name repeated after end keyword
    ) + _whitespace_comment_lineend_pattern + ( # optional whitespace/comment until line end...
        r'?' # ...with the '\n' optional
    ),
    re.IGNORECASE | re.DOTALL | re.MULTILINE
)
"""Pattern to match subroutine/function definitions."""


def match_module(source, scope):  # pylint:disable=unused-argument
    """
    Search for a module definition in :data:`source`

    This will only match the first occurence of a module, repeated calls using
    the remainder string are necessary to match further occurences.

    Parameters
    ----------
    source : :any:`Source`
        The source file object containing a string to match
    scope : :any:`Scope`
        The enclosing scope a matched object is embedded into

    Returns
    -------
    :any:`Source`, :any:`Module`, :any:`Source`
        A 3-tuple containing a source object with the source string before the match, the
        matched module object, and a source object with the source string after the match.
        If no match is found, the first two entries are `None` and the last is the original
        :data:`source` object.
    """
    from loki import Module  # pylint: disable=import-outside-toplevel
    match = _re_module.search(source.string)
    if not match:
        return None, None, source

    if match['spec']:
        spec = ir.RawSource(text=match['spec'], source=source.clone_with_span((match.span('spec'))))
    else:
        spec = None
    module = Module(name=match['name'], spec=spec, source=source.clone_with_span(match.span()))

    if match['contains']:
        keyword_source = source.clone_with_span(match.span('contains_keyword'))
        contains = [ir.Intrinsic(text=match['contains_keyword'], source=keyword_source)]
        if match['contains_body']:
            contains_source = source.clone_with_span(match.span('contains_body'))
            contains += match_block_candidates(contains_source, [match_subroutine_function], scope=module)
        # pylint: disable=unnecessary-dunder-call
        module.__init__(
            name=module.name, spec=module.spec, contains=contains,
            parent=module.parent, source=module.source, symbol_attrs=module.symbol_attrs
        )

    return (
        source.clone_with_span((0, match.span()[0])),
        module,
        source.clone_with_span((match.span()[1], len(source.string)))
    )

def match_subroutine_function(source, scope):
    """
    Search for a subroutine or function definition in :data:`source`

    This will only match the first occurence of a subroutine or function,
    repeated calls using the remainder string are necessary to match
    further occurences.

    Parameters
    ----------
    source : :any:`Source`
        The source file object containing a string to match
    scope : :any:`Scope`
        The enclosing scope a matched object is embedded into

    Returns
    -------
    :any:`Source`, :any:`Subroutine`, :any:`Source`
        A 3-tuple containing a source object with the source string before the match, the
        matched subroutine object, and a source object with the source string after the match.
        If no match is found, the first two entries are `None` and the last is the original
        :data:`source` object.
    """
    from loki import Subroutine  # pylint: disable=import-outside-toplevel
    match = _re_subroutine_function.search(source.string)
    if not match:
        return None, None, source

    if match['args']:
        args = match['args'].strip('() \t\n')
        args = _comment_pattern.sub('', args)
        args = tuple(arg.strip() for arg in args.split(','))
    else:
        args = ()
    is_function = match['keyword'].lower() == 'function'
    if match['spec']:
        spec = ir.RawSource(text=match['spec'], source=source.clone_with_span((match.span('spec'))))
    else:
        spec = None

    routine = Subroutine(
        name=match['name'], args=args, is_function=is_function, spec=spec,
        source=source.clone_with_span(match.span()), parent=scope
    )

    if match['contains']:
        keyword_source = source.clone_with_span(match.span('contains_keyword'))
        contains = [ir.Intrinsic(text=match['contains_keyword'], source=keyword_source)]
        if match['contains_body']:
            contains_source = source.clone_with_span(match.span('contains_body'))
            contains += match_block_candidates(contains_source, [match_subroutine_function], scope=routine)
        # pylint: disable=unnecessary-dunder-call
        routine.__init__(
            name=routine.name, args=routine._dummies, is_function=routine.is_function,
            spec=routine.spec, contains=contains,
            parent=routine.parent, source=routine.source, symbol_attrs=routine.symbol_attrs
        )

    return (
        source.clone_with_span((0, match.span()[0])),
        routine,
        source.clone_with_span((match.span()[1], len(source.string)))
    )
