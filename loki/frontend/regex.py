"""
The implementation of a regex parser frontend

This is intended to allow for fast, partial extraction of IR objects
from Fortran source files without the need to generate a complete
parse tree.
"""

import re

from loki import ir
from loki.expression import symbols as sym
from loki.frontend.util import REGEX
from loki.frontend.source import Source, source_to_lines, join_source_list
from loki.logging import PERF
from loki.scope import SymbolAttributes
from loki.tools import timeit, as_tuple
from loki.types import BasicType, ProcedureType

__all__ = ['parse_regex_source', 'HAVE_REGEX']


HAVE_REGEX = True
"""Indicate that the regex frontend is available."""


@timeit(log_level=PERF)
def parse_regex_source(source, scope=None, lazy_frontend=None):
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
    lazy_frontend : :any:`Frontend`, optional
        The frontend to use when triggering a full parse.
    """
    candidates = (match_module, match_subroutine_function)
    if not isinstance(source, Source):
        lines = (1, source.count('\n') + 1)
        source = Source(lines=lines, string=source)
    ir_ = match_block_candidates(source, candidates, scope=scope, lazy_frontend=lazy_frontend)
    return ir.Section(body=as_tuple(ir_), source=source)


def match_block_candidates(source, candidates, scope=None, lazy_frontend=None):
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
    lazy_frontend : :any:`Frontend`, optional
        The frontend to use when triggering a full parse.
    """
    blocks = []
    for idx, candidate in enumerate(candidates):
        while source.string:
            pre, match, source = candidate(source, scope=scope, lazy_frontend=lazy_frontend)
            if not match:
                assert pre is None
                break
            if pre:
                # See if any of the other candidates match before this match
                blocks += match_block_candidates(pre, candidates[idx+1:], scope=scope, lazy_frontend=lazy_frontend)
            blocks += [match]
    if source.string.strip():
        blocks += [ir.RawSource(text=source.string, source=source)]
    return blocks


def match_statement_candidates(source, candidates, scope=None):
    """
    Apply single-statement match functions to :data:`source`

    Parameters
    ----------
    source : :any:`Source`
        A source file object with the string to match
    candidates : list
        The list of candidate match functions to call
    scope : :any:`Scope`
        The parent scope for this source code snippet
    """
    source_lines = source_to_lines(source)

    ir_ = []
    source_stack = []
    for line in source_lines:
        for candidate in candidates:
            match = candidate(line, scope=scope)
            if match:
                if source_stack:
                    s = join_source_list(source_stack)
                    ir_ += [ir.RawSource(s.string, source=s)]
                    source_stack = []
                ir_ += [match]
                break
        else:
            source_stack += [line]
    if source_stack:
        s = join_source_list(source_stack)
        ir_ += [ir.RawSource(s.string, source=s)]
    return ir_


def match_block_statement_candidates(source, block_candidates, statement_candidates, scope=None, lazy_frontend=None):
    """
    Apply match functions to :data:`source` recursively

    This tries to first match any block candidates, and then runs through
    unmatched source sections trying to match statement candidates

    Parameters
    ----------
    source : :any:`Source`
        A source file object with the string to match
    block_candidates : list
        The list of candidate match functions for blocks to call
    statement_candidates : list
        The list of candidate match functions for individual statements to call
    scope : :any:`Scope`
        The parent scope for this source code snippet
    lazy_frontend : :any:`Frontend`, optional
        The frontend to use when triggering a full parse.
    """
    blocks = []
    for idx, candidate in enumerate(block_candidates):
        while source.string:
            pre, match, source = candidate(source, scope=scope)
            if not match:
                assert pre is None
                break
            if pre:
                # See if any of the other candidates match before this match
                blocks += match_block_statement_candidates(
                    pre, block_candidates[idx+1:], statement_candidates, scope=scope, lazy_frontend=lazy_frontend
                )
            blocks += [match]
    if source.string.strip():
        blocks += match_statement_candidates(source, statement_candidates, scope=scope)
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
        r'^[ \t]*'  # Whitespace before module keyword
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


def match_module(source, scope, lazy_frontend=None):
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
    lazy_frontend : :any:`Frontend`, optional
        The frontend to use when triggering a full parse.

    Returns
    -------
    :any:`Source`, :any:`Module`, :any:`Source`
        A 3-tuple containing a source object with the source string before the match, the
        matched module object, and a source object with the source string after the match.
        If no match is found, the first two entries are `None` and the last is the original
        :data:`source` object.
    """
    from loki import Module  # pylint: disable=import-outside-toplevel,cyclic-import
    match = _re_module.search(source.string)
    if not match:
        return None, None, source

    module = Module(name=match['name'], source=source.clone_with_span(match.span()), parent=scope)
    if match['spec']:
        block_candidates = (match_typedef, )
        statement_candidates = (match_import, )
        spec = match_block_statement_candidates(
            source.clone_with_span(match.span('spec')), block_candidates, statement_candidates,
            scope=module, lazy_frontend=lazy_frontend
        )
    else:
        spec = None

    if match['contains']:
        keyword_source = source.clone_with_span(match.span('contains_keyword'))
        contains = [ir.Intrinsic(text=match['contains_keyword'], source=keyword_source)]
        if match['contains_body']:
            contains_source = source.clone_with_span(match.span('contains_body'))
            contains += match_block_candidates(
                contains_source, [match_subroutine_function], scope=module, lazy_frontend=lazy_frontend
            )
    else:
        contains = None

    # pylint: disable=unnecessary-dunder-call
    module.__init__(
        name=module.name, spec=spec, contains=contains,
        parent=module.parent, source=module.source, symbol_attrs=module.symbol_attrs,
        incomplete=lazy_frontend is not None, frontend=lazy_frontend or REGEX
    )

    return (
        source.clone_with_span((0, match.span()[0])),
        module,
        source.clone_with_span((match.span()[1], len(source.string)))
    )


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


def match_subroutine_function(source, scope, lazy_frontend=None):
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
    lazy_frontend : :any:`Frontend`, optional
        The frontend to use when triggering a full parse.

    Returns
    -------
    :any:`Source`, :any:`Subroutine`, :any:`Source`
        A 3-tuple containing a source object with the source string before the match, the
        matched subroutine object, and a source object with the source string after the match.
        If no match is found, the first two entries are `None` and the last is the original
        :data:`source` object.
    """
    from loki import Subroutine  # pylint: disable=import-outside-toplevel,cyclic-import
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

    routine = Subroutine(
        name=match['name'], args=args, is_function=is_function,
        source=source.clone_with_span(match.span()), parent=scope
    )
    if match['spec']:
        candidates = (match_import, )
        spec = match_statement_candidates(source.clone_with_span(match.span('spec')), candidates, scope=routine)
    else:
        spec = None

    if match['contains']:
        keyword_source = source.clone_with_span(match.span('contains_keyword'))
        contains = [ir.Intrinsic(text=match['contains_keyword'], source=keyword_source)]
        if match['contains_body']:
            contains_source = source.clone_with_span(match.span('contains_body'))
            contains += match_block_candidates(
                contains_source, [match_subroutine_function], scope=routine, lazy_frontend=lazy_frontend
            )
    else:
        contains = None

    # pylint: disable=unnecessary-dunder-call
    routine.__init__(
        name=routine.name, args=routine._dummies, is_function=routine.is_function,
        spec=spec, contains=contains,
        parent=routine.parent, source=routine.source, symbol_attrs=routine.symbol_attrs,
        incomplete=lazy_frontend is not None, frontend=lazy_frontend or REGEX
    )

    return (
        source.clone_with_span((0, match.span()[0])),
        routine,
        source.clone_with_span((match.span()[1], len(source.string)))
    )


_re_typedef = re.compile(
    (
        r'^[ \t]*'  # Whitespace before type keyword
        r'type(?:\s*,\s*[\w\(\)]+)*?'  # type keyword with optional parameters
        r'(?:\s*::\s*|\s+)'  # optional `::` separator or white space
        r'(?P<name>\w+)[\S \t]*$'  # Type name
        r'(?P<spec>.*?)'  # Type spec
        r'(?P<contains>contains.*?)?'  # Optional procedure bindings part (after ``contains`` keyword)
        r'\s+end\s+type(?:\s*(?P=name))?'  # End keyword with optionally type name repeated
    ),
    re.IGNORECASE | re.DOTALL | re.MULTILINE
)
"""
Pattern to match derived type definitions via ``type...end type`` keywords.

Spec and typebound-procedures-part (``contains`` and everything thereafter)
are matched in separate groups for subsequent processing with :any:`_re_proc_binding`.
"""


def match_typedef(source, scope):
    """
    Search for a derived type definition in :data:`source`

    This will only match the first occurence of a derived type definition,
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
    :any:`Source`, :any:`TypeDef`, :any:`Source`
        A 3-tuple containing a source object with the source string before the match, the
        matched typedef object, and a source object with the source string after the match.
        If no match is found, the first two entries are `None` and the last is the original
        :data:`source` object.
    """
    match = _re_typedef.search(source.string)
    if not match:
        return None, None, source
    typedef = ir.TypeDef(
        name=match['name'], body=(), parent=scope,
        source=source.clone_with_span(match.span())
    )
    if match['spec']:
        spec_source = source.clone_with_span(match.span('spec'))
        spec = [ir.RawSource(text=spec_source.string, source=spec_source)]
    else:
        spec = []
    if match['contains']:
        candidates = (match_procedure_binding, match_generic_binding)
        contains = match_statement_candidates(source.clone_with_span(match.span('contains')), candidates, scope=typedef)
    else:
        contains = []
    typedef._update(body=as_tuple(spec + contains))

    return (
        source.clone_with_span((0, match.span()[0])),
        typedef,
        source.clone_with_span((match.span()[1], len(source.string)))
    )


_re_proc_binding = re.compile(
    (
        r'^[ \t]*procedure'  # Match ``procedure keyword after optional white space
        r'(?P<attributes>(?:[ \t]*,[ \t]*\w+)*?)'  # Optional attributes
        r'(?:[ \t]*::)?'  # Optional `::` delimiter
        r'[ \t]*'  # Some white space
        r'(?P<bindings>'  # Beginning of bindings group
        r'\w+(?:[ \t]*=>[ \t]*\w+)?'  # Binding name with optional binding name specifier (via ``=>``)
        r'(?:[ \t]*,[ \t]*' # Optional group for additional bindings, separated by ``,``
        r'\w+(?:[ \t]*=>[ \t]*\w+)?'  # Additional binding name with optional binding name specifier
        r')*'  # End of optional group for additional bindings
        r')'  # End of bindings group
    ),
    re.IGNORECASE
)
"""
Pattern to match type-bound procedure declarations within the
typebound-procedures-part of a derived type definition via ``procedure...`` keyword.

It matches the full binding-name, i.e., including any optional rename such as
``name => bind_name``.
"""


def match_procedure_binding(source, scope):
    """
    Search for procedure bindings in derived types

    Parameters
    ----------
    source : :any:`Source`
        The source file object containing a single-line string to match
    scope : :any:`Scope`
        The enclosing scope a matched object is embedded into

    Returns
    -------
    :any:`ProcedureDeclaration` or NoneType
        The matched object or `None`
    """
    match = _re_proc_binding.search(source.string)
    if not match:
        return None
    bindings = match['bindings'].replace(' ', '').split(',')
    bindings = [s.split('=>') for s in bindings]

    symbols = []
    for s in bindings:
        if len(s) == 1:
            type_ = SymbolAttributes(ProcedureType(name=s[0]))
            symbols += [sym.Variable(name=s[0], type=type_, scope=scope)]
        else:
            type_ = SymbolAttributes(ProcedureType(name=s[1]))
            initial = sym.Variable(name=s[1], type=type_, scope=scope.parent)
            symbols += [sym.Variable(name=s[0], type=type_.clone(initial=initial), scope=scope)]
    return ir.ProcedureDeclaration(symbols=symbols, source=source.clone_with_span(match.span()))


_re_generic_binding = re.compile(
    (
        r'^[ \t]*generic'  # Match ``generic`` keyword after optional white space
        r'(?P<attributes>(?:[ \t]*,[ \t]*\w+)*?)'  # Optional attributes
        r'(?:[ \t]*::)?'  # Optional `::` delimiter
        r'[ \t]*'  # Some white space
        r'(?P<name>\w+)'  # Binding name
        r'[ \t]*=>[ \t]*'  # Separator ``=>``
        r'(?P<bindings>\w+(?:[ \t]*,[ \t]*\w+)*)*'  # Match binding name list
    ),
    re.IGNORECASE
)
"""
Pattern to match generic name declarations for type-bound procedures within the
typebound-procedures part of a derived type definition via ``generic...`` keyword.

It matches the generic name and the binding name list
"""


def match_generic_binding(source, scope):
    """
    Search for generic bindings in derived types

    Parameters
    ----------
    source : :any:`Source`
        The source file object containing a single-line string to match
    scope : :any:`Scope`
        The enclosing scope a matched object is embedded into

    Returns
    -------
    :any:`ProcedureDeclaration` or NoneType
        The matched object or `None`
    """
    match = _re_generic_binding.search(source.string)
    if not match:
        return None
    bindings = match['bindings'].replace(' ', '').split(',')
    name = match['name']
    type_ = SymbolAttributes(ProcedureType(name=name, is_generic=True), bind_names=as_tuple(bindings))
    symbols = (sym.Variable(name=name, type=type_, scope=scope),)
    return ir.ProcedureDeclaration(symbols=symbols, generic=True, source=source.clone_with_span(match.span()))


_re_imports = re.compile(
    r'^ *use +(?P<module>\w+)(?: *, *(?P<only>only *:)?'  # The use statement including an optional ``only``
    r'(?P<imports>(?: *\w+(?: *=> *\w+)? *,?)+))?',  # The optional list of names (with optional renames)
    re.IGNORECASE
)
"""Pattern to match Fortran imports (``USE`` statements)"""


def match_import(source, scope):
    """
    Search for imports via Fortran's ``USE`` statement

    Parameters
    ----------
    source : :any:`Source`
        The source file object containing a single-line string to match
    scope : :any:`Scope`
        The enclosing scope a matched object is embedded into

    Returns
    -------
    :any:`Import` or NoneType
        The matched object or `None`
    """
    match = _re_imports.search(source.string)
    if not match:
        return None
    module = match['module']

    type_ = SymbolAttributes(BasicType.DEFERRED, imported=True)
    if match['imports']:
        imports = match['imports'].replace(' ', '').split(',')
        imports = [s.split('=>') for s in imports]
        if match['only']:
            rename_list = None
            symbols = []
            for s in imports:
                if len(s) == 1:
                    symbols += [sym.Variable(name=s[0], type=type_, scope=scope)]
                else:
                    symbols += [sym.Variable(name=s[0], type=type_.clone(use_name=s[1]), scope=scope)]
        else:
            rename_list = [
                (s[1], sym.Variable(name=s[0], type=type_.clone(use_name=s[1]), scope=scope))
                for s in imports
            ]
            symbols = None
    else:
        rename_list = None
        symbols = None
    return ir.Import(module, symbols=symbols, rename_list=rename_list, source=source.clone_with_span(match.span()))
