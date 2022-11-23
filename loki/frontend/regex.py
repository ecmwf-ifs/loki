"""
The implementation of a regex parser frontend

This is intended to allow for fast, partial extraction of IR objects
from Fortran source files without the need to generate a complete
parse tree.
"""
from abc import abstractmethod
from enum import Flag, auto
import re
from codetiming import Timer

from loki import ir
from loki.expression import symbols as sym
from loki.frontend.source import Source, FortranReader
from loki.logging import debug
from loki.scope import SymbolAttributes
from loki.tools import as_tuple
from loki.types import BasicType, ProcedureType, DerivedType

__all__ = ['RegexParserClass', 'parse_regex_source', 'HAVE_REGEX']


HAVE_REGEX = True
"""Indicate that the regex frontend is available."""


class RegexParserClass(Flag):
    """
    Classes to configure active patterns in the :any:`REGEX` frontend

    Every :class:`Pattern` in the frontend is categorized as one of these classes.
    By specifying some (or all of them) as ``parser_classes`` to :any:`parse_regex_source`,
    pattern matching can be switched on and off for some pattern classes, and thus the overall
    parse time reduced.
    """
    ProgramUnitClass = auto()
    ImportClass = auto()
    TypeDefClass = auto()
    DeclarationClass = auto()
    CallClass = auto()
    AllClasses = ProgramUnitClass | ImportClass | TypeDefClass | DeclarationClass | CallClass  # pylint: disable=unsupported-binary-operation


class Pattern:
    """
    Base class for patterns used in the :any:`REGEX` frontend

    Parameters
    ----------
    pattern : str
        The regex pattern used for matching
    flags : re.RegexFlag
        Regular expression flag(s) to use when compiling and matching the pattern
    """

    def __init__(self, pattern, flags=None):
        self.pattern = re.compile(pattern, flags)

    @abstractmethod
    def match(self, reader, parser_classes, scope):
        """
        Match the stored pattern against the source string in the reader object

        This method must be implemented by every child class to provide the
        matching logic. It is not necessary to check the selected :data:`parser_classes`
        here, as the relevant :meth:`match` method will only be called for :class:`Pattern`
        classes that are active. :data:`parser_classes` is only passed here to forward it
        to use it when matching recursively.

        If this match method matches against a single line, it should return a :any:`Node`
        if matched successfully, or otherwise `None`.

        If this match method matches a block, e.g. a :any:`Subroutine`, then this should
        return a 3-tupel ``(pre, node, new_reader)``, with each entry:

        - ``pre`` : A :any:`FortranReader` object representing any unmatched source code
                    fragments prior to the matched object. Can be `None` if there are none
                    or if there was no match.
        - ``node``: The object created as the result of a successful match, or `None`.
        - ``new_reader``: A :any:`FortranReader` object representing any unmatched source
                          code fragments past the matched object. Can be `None` if there
                          are none and should be the original :data:`reader` object if
                          there was no match.

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """

    @classmethod
    def match_block_candidates(cls, reader, candidates, parser_classes=None, scope=None):
        """
        Attempt to match block candidates

        It will automatically skip :data:`candidates` that are inactive due to the chosen
        :data:`parser_classes`.

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        candidates : list of str
            The list of candidate classes to match
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        if parser_classes is None:
            parser_classes = RegexParserClass.AllClasses
        ir_ = []

        # Extract source bits that would be swept under the rag when sanitizing
        head = reader.source_from_head()
        tail = reader.source_from_tail()

        for idx, candidate_name in enumerate(candidates):
            candidate = PATTERN_REGISTRY[candidate_name]
            if not candidate.parser_class & parser_classes:
                continue
            while reader:
                pre, match, reader = candidate.match(reader, parser_classes=parser_classes, scope=scope)
                if not match:
                    assert pre is None
                    break
                if pre:
                    # See if any of the other candidates match before this match
                    ir_ += cls.match_block_candidates(
                        pre, candidates[idx+1:], parser_classes=parser_classes, scope=scope
                    )
                ir_ += [match]

        if reader:
            source = reader.to_source(include_padding=True)
            ir_ += [ir.RawSource(text=source.string, source=source)]
        if head is not None and (not ir_ or ir_[0].source.lines[0] > head.lines[1]):
            # Insert the header bit only if the recursion hasn't already taken care of it
            ir_ = [ir.RawSource(text=head.string, source=head)] + ir_
        if tail is not None:
            ir_ += [ir.RawSource(text=tail.string, source=tail)]
        return ir_

    @classmethod
    def match_statement_candidates(cls, reader, candidates, parser_classes=None, scope=None):
        """
        Attempt to match single-line statement candidates

        It will automatically skip :data:`candidates` that are inactive due to the chosen
        :data:`parser_classes`.

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        candidates : list of str
            The list of candidate classes to match
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        if parser_classes is None:
            parser_classes = RegexParserClass.AllClasses
        # Extract source bits that would be swept under the rag when sanitizing
        head = reader.source_from_head()

        filtered_candidates = [PATTERN_REGISTRY[candidate_name] for candidate_name in candidates]
        filtered_candidates = [
            candidate for candidate in filtered_candidates if candidate.parser_class & parser_classes
        ]
        if not filtered_candidates:
            return []

        ir_ = []
        last_match = -1
        for idx, _ in enumerate(reader):
            for candidate in filtered_candidates:
                match = candidate.match(reader, parser_classes=parser_classes, scope=scope)
                if match:
                    if last_match - idx > 1:
                        span = (reader.sanitized_spans[last_match + 1], reader.sanitized_spans[idx])
                        source = reader.source_from_sanitized_span(span)
                        ir_ += [ir.RawSource(source.string, source=source)]
                    last_match = idx
                    ir_ += [match]
                    break

        if head is not None and ir_:
            ir_ = [ir.RawSource(text=head.string, source=head)] + ir_

        tail_span = (reader.sanitized_spans[last_match + 1], None)
        source = reader.source_from_sanitized_span(tail_span, include_padding=True)
        if source:
            ir_ += [ir.RawSource(source.string, source=source)]
        return ir_

    @classmethod
    def match_block_statement_candidates(
        cls, reader, block_candidates, statement_candidates, parser_classes=None, scope=None
    ):
        """
        Attempt to match block candidates and subsequently attempt to match statement candidates
        on unmatched sections

        It will automatically skip :data:`candidates` that are inactive due to the chosen
        :data:`parser_classes`.

        This is essentially equivalent to :meth:`match_block_candidates` but applies
        :meth:`match_statement_candidates` to the unmatched tail source instead of returning
        it as a :any:`RawSource` object straight away.

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        block_candidates : list of str
            The list of block candidate classes to match
        statement_candidates : list of str
            The list of statement candidate classes to match
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        if parser_classes is None:
            parser_classes = RegexParserClass.AllClasses
        # Extract source bits that would be swept under the rag when sanitizing
        head = reader.source_from_head()

        ir_ = []
        for idx, candidate_name in enumerate(block_candidates):
            candidate = PATTERN_REGISTRY[candidate_name]
            if not candidate.parser_class & parser_classes:
                continue
            while reader:
                pre, match, reader = candidate.match(reader, parser_classes=parser_classes, scope=scope)
                if not match:
                    assert pre is None
                    break
                if pre:
                    # See if any of the other candidates match before this match
                    ir_ += cls.match_block_statement_candidates(
                        pre, block_candidates[idx+1:], statement_candidates, scope=scope
                    )
                ir_ += [match]

        if head is not None and ir_ and reader:
            # Insert the head source bits only if we have matched something, otherwise
            # the statement candidate matching will take care of this
            ir_ = [ir.RawSource(text=head.string, source=head)] + ir_

        if reader:
            ir_ += cls.match_statement_candidates(
                reader, statement_candidates, parser_classes=parser_classes, scope=scope
            )
        return ir_

    _pattern_opening_parenthesis = re.compile(r'\(')
    _pattern_closing_parenthesis = re.compile(r'\)')
    _pattern_quoted_string = re.compile(r'(?:\'.*?\')|(?:".*?")')

    @classmethod
    def _remove_quoted_string_nested_parentheses(cls, string):
        """
        Remove any quoted strings and parentheses with their content in the given string
        """
        string = cls._pattern_quoted_string.sub('', string)
        p_open = [match.start() for match in cls._pattern_opening_parenthesis.finditer(string)]
        p_close = [match.start() for match in cls._pattern_closing_parenthesis.finditer(string)]
        if len(p_open) > len(p_close):
            # Note: fparser's reader has currently problems with opening
            # quotes in comments in combination with line continuation, thus
            # potentially failing to sanitize the string correctly.
            # In that case, we'll just discard everything after the first
            # opening parenthesis, well aware that we're potentially
            # loosing information...
            # See https://github.com/stfc/fparser/issues/264
            return string[:p_open[0]]
        assert len(p_open) == len(p_close)
        if not p_close:
            return string

        # We match pairs of parentheses starting at the end by pushing and popping from a stack.
        # Whenever the stack runs out, we have fully resolved a set of (nested) parenthesis and
        # record the corresponding span
        spans = []
        stack = [p_close.pop()]
        while p_open:
            if not p_close or p_open[-1] > p_close[-1]:
                assert stack
                start = p_open.pop()
                end = stack.pop()
                if not stack:
                    spans.append((start, end))
            else:
                stack.append(p_close.pop())

        # We should now be left with no parentheses anymore and can build the new string
        # by using everything between these parenthesis "spans"
        assert not (stack or p_open or p_close)
        spans.reverse()
        new_string = string[:spans[0][0]]
        for (_, start), (end, _) in zip(spans[:-1], spans[1:]):
            new_string += string[start+1:end]
        new_string += string[spans[-1][1]+1:]
        return new_string


@Timer(logger=debug, text=lambda s: f'[Loki::REGEX] Executed parse_regex_source in {s:.2f}s')
def parse_regex_source(source, parser_classes=None, scope=None):
    """
    Generate a reduced Loki IR from regex parsing of the given Fortran source

    The IR nodes that should be matched can be configured via :data:`parser_classes`.
    Any non-matched source code snippets are retained as :any:`RawSource` objects.

    Parameters
    ----------
    source : str or :any:`Source`
        The raw source string
    parser_classes : RegexParserClass
        Active parser classes for matching
    scope : :any:`Scope`, optional
        The enclosing parent scope
    """
    if parser_classes is None:
        parser_classes = RegexParserClass.AllClasses
    candidates = ('ModulePattern', 'SubroutineFunctionPattern')
    if isinstance(source, Source):
        reader = FortranReader(source.string)
    else:
        reader = FortranReader(source)
    ir_ = Pattern.match_block_candidates(reader, candidates, parser_classes=parser_classes, scope=scope)
    return ir.Section(body=as_tuple(ir_), source=source)


class ModulePattern(Pattern):
    """
    Pattern to match :any:`Module` objects
    """

    parser_class = RegexParserClass.ProgramUnitClass

    def __init__(self):
        super().__init__(
            r'^module[ \t]+(?P<name>\w+)\b.*?$'
            r'(?P<spec>.*?)'
            r'(?P<contains>^contains\n(?:'
            r'(?:[ \t\w()]*?subroutine.*?^end[ \t]*subroutine\b(?:[ \t]\w+)?\n)|'
            r'(?:[ \t\w()]*?function.*?^end[ \t]*function\b(?:[ \t]\w+)?\n)|'
            r'(?:^#\w+.*?\n)'
            r')*)?'
            r'^end[ \t]*module\b(?:[ \t](?P=name))?',
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a :any:`Module`

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        from loki import Module  # pylint: disable=import-outside-toplevel,cyclic-import
        match = self.pattern.search(reader.sanitized_string)
        if not match:
            return None, None, reader

        # Check if the Module node has been created before by looking it up in the scope
        module = None
        name = match['name']
        if scope is not None and name in scope.symbol_attrs:
            module_type = scope.symbol_attrs[name]  # Look-up only in current scope!
            if module_type and module_type.dtype.module != BasicType.DEFERRED:
                module = module_type.dtype.module

        if module is None:
            source = reader.source_from_sanitized_span(match.span())
            module = Module(name=name, source=source, parent=scope)

        if match['spec'] and match['spec'].strip():
            block_candidates = ('TypedefPattern',)
            statement_candidates = ('ImportPattern', 'VariableDeclarationPattern')
            spec = self.match_block_statement_candidates(
                reader.reader_from_sanitized_span(match.span('spec'), include_padding=True),
                block_candidates, statement_candidates, parser_classes=parser_classes, scope=module
            )
        else:
            spec = None

        if match['contains']:
            contains = [ir.Intrinsic(text='CONTAINS')]
            span = match.span('contains')
            span = (span[0] + 8, span[1])  # Skip the "contains" keyword as it has been added
            candidates = ['SubroutineFunctionPattern']
            contains += self.match_block_candidates(
                reader.reader_from_sanitized_span(span, include_padding=True),
                candidates, parser_classes=parser_classes, scope=module
            )
        else:
            contains = None

        module.__init__(  # pylint: disable=unnecessary-dunder-call
            name=module.name, spec=spec, contains=contains, parent=module.parent,
            source=module.source, symbol_attrs=module.symbol_attrs, incomplete=True
        )

        if match.span()[0] > 0:
            pre = reader.reader_from_sanitized_span((0, match.span()[0]), include_padding=True)
        else:
            pre = None
        return pre, module, reader.reader_from_sanitized_span((match.span()[1], None), include_padding=True)


class SubroutineFunctionPattern(Pattern):
    """
    Pattern to match :any:`Subroutine` objects
    """

    parser_class = RegexParserClass.ProgramUnitClass

    def __init__(self):
        super().__init__(
            r'^[ \t\w()]*?(?P<keyword>subroutine|function)[ \t]+(?P<name>\w+)\b.*?$'
            r'(?P<spec>.*?)'
            r'(?P<contains>^contains\n(?:'
            r'(?:[ \t\w()]*?subroutine.*?^end[ \t]*subroutine\b(?:[ \t]\w+)?\n)|'
            r'(?:[ \t\w()]*?function.*?^end[ \t]*function\b(?:[ \t]\w+)?\n)|'
            r'(?:^#\w+.*?\n)'
            r')*)?'
            r'^end[ \t]*(?P=keyword)\b(?:[ \t](?P=name))?',
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a :any:`Subroutine`

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        from loki import Subroutine  # pylint: disable=import-outside-toplevel,cyclic-import
        match = self.pattern.search(reader.sanitized_string)
        if not match:
            return None, None, reader

        # Check if the Subroutine node has been created before by looking it up in the scope
        routine = None
        name = match['name']
        if scope is not None and name in scope.symbol_attrs:
            proc_type = scope.symbol_attrs[name]  # Look-up only in current scope!
            if proc_type and getattr(proc_type.dtype, 'procedure', BasicType.DEFERRED) != BasicType.DEFERRED:
                routine = proc_type.dtype.procedure

        if routine is None:
            is_function = match['keyword'].lower() == 'function'
            source = reader.source_from_sanitized_span(match.span())
            routine = Subroutine(
                name=name, args=(), is_function=is_function, source=source, parent=scope
            )

        if match['spec']:
            statement_candidates = ('ImportPattern', 'VariableDeclarationPattern', 'CallPattern')
            spec = self.match_statement_candidates(
                reader.reader_from_sanitized_span(match.span('spec'), include_padding=True),
                statement_candidates, parser_classes=parser_classes, scope=routine
            )
        else:
            spec = None

        if match['contains']:
            contains = [ir.Intrinsic(text='CONTAINS')]
            span = match.span('contains')
            span = (span[0] + 8, span[1])  # Skip the "contains" keyword as it has been added
            block_children = ['SubroutineFunctionPattern']
            contains += self.match_block_candidates(
                reader.reader_from_sanitized_span(span), block_children, parser_classes=parser_classes, scope=routine
            )
        else:
            contains = None

        routine.__init__(  # pylint: disable=unnecessary-dunder-call
            name=routine.name, args=routine._dummies, is_function=routine.is_function,
            spec=spec, contains=contains, parent=routine.parent, source=routine.source,
            symbol_attrs=routine.symbol_attrs, incomplete=True
        )

        if match.span()[0] > 0:
            pre = reader.reader_from_sanitized_span((0, match.span()[0]), include_padding=True)
        else:
            pre = None
        return pre, routine, reader.reader_from_sanitized_span((match.span()[1], None), include_padding=True)


class TypedefPattern(Pattern):
    """
    Pattern to match :any:`TypeDef` objects
    """

    parser_class = RegexParserClass.TypeDefClass

    def __init__(self):
        super().__init__(
            r'type(?:[ \t]*,[ \t]*[\w\(\)]+)*?'  # type keyword with optional parameters
            r'(?:[ \t]*::[ \t]*|[ \t]+)'  # optional `::` separator or white space
            r'(?P<name>\w+)\b.*?$'  # Type name
            r'(?P<spec>.*?)'  # Type spec
            r'(?P<contains>^contains\n.*?)?'  # Optional procedure bindings part (after ``contains`` keyword)
            r'^end[ \t]*type\b(?:[ \t]+(?P=name))?',  # End keyword with optionally type name repeated
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a :any:`TypeDef`

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        match = self.pattern.search(reader.sanitized_string)
        if not match:
            return None, None, reader

        source = reader.source_from_sanitized_span(match.span())
        typedef = ir.TypeDef(name=match['name'], body=(), parent=scope, source=source)

        if match['spec'] and match['spec'].strip():
            statement_candidates = ('VariableDeclarationPattern',)
            spec = self.match_statement_candidates(
                reader.reader_from_sanitized_span(match.span('spec'), include_padding=True),
                statement_candidates, parser_classes=parser_classes, scope=typedef
            )
        else:
            spec = []

        if match['contains']:
            contains = [ir.Intrinsic(text='CONTAINS')]
            span = match.span('contains')
            span = (span[0] + 8, span[1])  # Skip the "contains" keyword as it has been added

            statement_candidates = ('ProcedureBindingPattern', 'GenericBindingPattern')
            contains += self.match_statement_candidates(
                reader.reader_from_sanitized_span(span, include_padding=True),
                statement_candidates, parser_classes=parser_classes, scope=typedef
            )
        else:
            contains = []
        typedef._update(body=as_tuple(spec + contains))

        if match.span()[0] > 0:
            pre = reader.reader_from_sanitized_span((0, match.span()[0]), include_padding=True)
        else:
            pre = None
        return pre, typedef, reader.reader_from_sanitized_span((match.span()[1], None), include_padding=True)


class ProcedureBindingPattern(Pattern):
    """
    Pattern to match procedure bindings
    """

    parser_class = RegexParserClass.TypeDefClass

    def __init__(self):
        super().__init__(
            r'^procedure\b'  # Match ``procedure`` keyword
            r'(?P<attributes>(?:[ \t]*,[ \t]*\w+)*?)'  # Optional attributes
            r'(?:[ \t]*::)?'  # Optional `::` delimiter
            r'[ \t]*'  # Some white space
            r'(?P<bindings>'  # Beginning of bindings group
            r'\w+(?:[ \t]*=>[ \t]*\w+)?'  # Binding name with optional binding name specifier (via ``=>``)
            r'(?:[ \t]*,[ \t]*' # Optional group for additional bindings, separated by ``,``
            r'\w+(?:[ \t]*=>[ \t]*\w+)?'  # Additional binding name with optional binding name specifier
            r')*'  # End of optional group for additional bindings
            r')',  # End of bindings group
            re.IGNORECASE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a procedure binding

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        line = reader.current_line
        match = self.pattern.search(line.line)
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

        return ir.ProcedureDeclaration(symbols=symbols, source=reader.source_from_current_line)


class GenericBindingPattern(Pattern):
    """
    Pattern to match generic bindings
    """

    parser_class = RegexParserClass.TypeDefClass

    def __init__(self):
        super().__init__(
            r'^generic'  # Match ``generic`` keyword
            r'(?P<attributes>(?:[ \t]*,[ \t]*\w+)*?)'  # Optional attributes
            r'(?:[ \t]*::)?'  # Optional `::` delimiter
            r'[ \t]*'  # Some white space
            r'(?P<name>\w+)'  # Binding name
            r'[ \t]*=>[ \t]*'  # Separator ``=>``
            r'(?P<bindings>\w+(?:[ \t]*,[ \t]*\w+)*)*',  # Match binding name list
            re.IGNORECASE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a generic procedure binding

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        line = reader.current_line
        match = self.pattern.search(line.line)
        if not match:
            return None

        bindings = match['bindings'].replace(' ', '').split(',')
        name = match['name']
        type_ = SymbolAttributes(ProcedureType(name=name, is_generic=True), bind_names=as_tuple(bindings))
        symbols = (sym.Variable(name=name, type=type_, scope=scope),)
        return ir.ProcedureDeclaration(symbols=symbols, generic=True, source=reader.source_from_current_line())


class ImportPattern(Pattern):
    """
    Pattern to match :any:`Import` nodes
    """

    parser_class = RegexParserClass.ImportClass

    def __init__(self):
        super().__init__(
            r'^use +(?P<module>\w+)(?: *, *(?P<only>only *:)?'  # The use statement including an optional ``only``
            r'(?P<imports>(?: *\w+(?: *=> *\w+)? *,?)+))?',  # The optional list of names (with optional renames)
            re.IGNORECASE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a :any:`Import`

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        line = reader.current_line
        match = self.pattern.search(line.line)
        if not match:
            return None

        module = match['module']
        type_ = SymbolAttributes(BasicType.DEFERRED, imported=True)
        if match['imports']:
            imports = match['imports'].replace(' ', '').split(',')
            imports = [s.split('=>') for s in imports]
            imports = [s for s in imports if s and s[0]]
            if match['only']:
                rename_list = None
                symbols = []
                for s in imports:
                    if not s[0]:
                        continue
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

        return ir.Import(
            module, symbols=as_tuple(symbols), rename_list=rename_list,
            source=reader.source_from_current_line()
        )


class VariableDeclarationPattern(Pattern):
    """
    Pattern to match :any:`VariableDeclaration` nodes

    For the moment, this only matches variable declarations for derived type objects
    (via ``TYPE`` or ``CLASS`` keywords).
    """

    parser_class = RegexParserClass.DeclarationClass

    def __init__(self):
        super().__init__(
            r'^(?:type|class)[ \t]*\([ \t]*(?P<typename>\w+)[ \t]*\)'  # TYPE or CLASS keyword with typename
            r'(?:[ \t]*,[ \t]*[a-z]+(?:\(.*?\))?)*'  # Optional attributes
            r'(?:[ \t]*::)?'  # Optional `::` delimiter
            r'[ \t]*'  # Some white space
            r'(?P<variables>\w+\b.*?)$',  # Variable names
            re.IGNORECASE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a :any:`Import`

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        line = reader.current_line
        match = self.pattern.search(line.line)
        if not match:
            return None

        type_ = SymbolAttributes(DerivedType(match['typename']))
        variables = self._remove_quoted_string_nested_parentheses(match['variables'])  # Remove dimensions
        variables = variables.replace(' ', '').split(',')  # Variable names without white space
        variables = tuple(sym.Variable(name=v, type=type_, scope=scope) for v in variables)
        return ir.VariableDeclaration(variables, source=reader.source_from_current_line())


class CallPattern(Pattern):
    """
    Pattern to match :any:`CallStatement` nodes
    """

    parser_class = RegexParserClass.CallClass

    def __init__(self):
        super().__init__(
            r'^(?P<conditional>if[ \t]*\(.*?\)[ \t]*)?'  # Optional inline-conditional preceeding the call
            r'call',  # Call keyword
            re.IGNORECASE
        )

    def match(self, reader, parser_classes, scope):
        """
        Match the provided source string against the pattern for a :any:`CallStatement`

        Parameters
        ----------
        reader : :any:`FortranReader`
            The reader object containing a sanitized Fortran source
        parser_classes : RegexParserClass
            Active parser classes for matching
        scope : :any:`Scope`
            The parent scope for the current source fragment
        """
        line = reader.current_line
        match = self.pattern.search(line.line)
        if not match:
            return None

        # Extract the called routine name
        call = line.line[match.span()[1]:].strip()
        if not call:
            return None
        call = self._remove_quoted_string_nested_parentheses(call)  # Remove arguments and dimension expressions
        call = call.replace(' ', '')  # Remove any white space

        name_parts = call.split('%')
        name = sym.Variable(name=name_parts[0], scope=scope)
        for cname in name_parts[1:]:
            name = sym.Variable(name=name.name + '%' + cname, parent=name, scope=scope)  # pylint:disable=no-member

        scope.symbol_attrs[call] = scope.symbol_attrs[call].clone(dtype=ProcedureType(name=call, is_function=False))

        source = reader.source_from_current_line()
        if match['conditional']:
            span = match.span('conditional')
            return [
                ir.RawSource(text=match['conditional'], source=source.clone_with_span(span)),
                ir.CallStatement(name=name, arguments=(), source=source.clone_with_span((span[1], None)))
            ]
        return ir.CallStatement(name=name, arguments=(), source=source)


PATTERN_REGISTRY = {
    name: globals()[name]() for name in dir()
    if name.endswith('Pattern') and name != 'Pattern'
}
"""
A global registry of all available patterns

This exists to ensure every :any:`Pattern` implementation is only instantiated
once to ensure the corresponding regular expressions are not compiled multiple times.
"""
