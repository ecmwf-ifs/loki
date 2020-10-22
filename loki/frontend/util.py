from itertools import groupby
from enum import IntEnum
from pathlib import Path
import codecs

from loki.visitors import Visitor, NestedTransformer, FindNodes
from loki.ir import (Assignment, Comment, CommentBlock, Declaration, Pragma, Loop, Intrinsic)
from loki.frontend.source import Source
from loki.types import BasicType, SymbolType
from loki.expression import Literal, Variable
from loki.tools import as_tuple
from loki.logging import warning

__all__ = ['Frontend', 'OFP', 'OMNI', 'FP', 'inline_comments', 'cluster_comments',
           'inline_pragmas', 'process_dimension_pragmas', 'read_file']


class Frontend(IntEnum):
    OMNI = 1
    OFP = 2
    FP = 3

    def __str__(self):
        return self.name.lower()  # pylint: disable=no-member


OMNI = Frontend.OMNI
OFP = Frontend.OFP
FP = Frontend.FP  # The STFC FParser


class SequenceFinder(Visitor):
    """
    Utility visitor that finds repeated nodes of the same type in
    lists/tuples within a given tree.
    """

    def __init__(self, node_type):
        super(SequenceFinder, self).__init__()
        self.node_type = node_type

    @classmethod
    def default_retval(cls):
        return []

    def visit_tuple(self, o, **kwargs):
        groups = []
        for c in o:
            # First recurse...
            subgroups = self.visit(c)
            if subgroups is not None and len(subgroups) > 0:
                groups += subgroups
        for t, group in groupby(o, type):
            # ... then add new groups
            g = tuple(group)
            if t is self.node_type and len(g) > 1:
                groups.append(g)
        return groups

    visit_list = visit_tuple


class PatternFinder(Visitor):
    """
    Utility visitor that finds a pattern of nodes given as tuple/list
    of types within a given tree.
    """

    def __init__(self, pattern):
        super(PatternFinder, self).__init__()
        self.pattern = pattern

    @classmethod
    def default_retval(cls):
        return []

    @staticmethod
    def match_indices(pattern, sequence):
        """ Return indices of matched patterns in sequence. """
        matches = []
        for i, elem in enumerate(sequence):
            if elem == pattern[0]:
                if tuple(sequence[i:i+len(pattern)]) == tuple(pattern):
                    matches.append(i)
        return matches

    def visit_tuple(self, o, **kwargs):
        matches = []
        for c in o:
            # First recurse...
            submatches = self.visit(c)
            if submatches is not None and len(submatches) > 0:
                matches += submatches
        types = list(map(type, o))
        idx = self.match_indices(self.pattern, types)
        for i in idx:
            matches.append(o[i:i+len(self.pattern)])
        return matches

    visit_list = visit_tuple


def inline_comments(ir):
    """
    Identify inline comments and merge them onto statements
    """
    pairs = PatternFinder(pattern=(Assignment, Comment)).visit(ir)
    pairs += PatternFinder(pattern=(Declaration, Comment)).visit(ir)
    mapper = {}
    for pair in pairs:
        # Comment is in-line and can be merged
        # Note, we need to re-create the statement node
        # so that Transformers don't throw away the changes.
        if pair[0]._source and pair[1]._source:
            if pair[1]._source.lines[0] == pair[0]._source.lines[1]:
                mapper[pair[0]] = pair[0]._rebuild(comment=pair[1])
                mapper[pair[1]] = None  # Mark for deletion
    return NestedTransformer(mapper, invalidate_source=False).visit(ir)


def cluster_comments(ir):
    """
    Cluster comments into comment blocks
    """
    comment_mapper = {}
    comment_groups = SequenceFinder(node_type=Comment).visit(ir)
    for comments in comment_groups:
        # Build a CommentBlock and map it to first comment
        # and map remaining comments to None for removal
        if all(c._source is not None for c in comments):
            if all(c.source.string is not None for c in comments):
                string = '\n'.join(c.source.string for c in comments)
            else:
                string = None
            lines = (comments[0].source.lines[0], comments[-1].source.lines[1])
            source = Source(lines=lines, string=string, file=comments[0].source.file)
        else:
            source = None
        block = CommentBlock(comments, label=comments[0].label, source=source)
        comment_mapper[comments[0]] = block
        for c in comments[1:]:
            comment_mapper[c] = None
    return NestedTransformer(comment_mapper, invalidate_source=False).visit(ir)


def inline_pragmas(ir):
    """
    Find pragmas and merge them onto declarations and subroutine calls

    Note: Pragmas in derived types are already associated with the
    declaration due to way we parse derived types.
    """
    pairs = PatternFinder(pattern=(Pragma, Declaration)).visit(ir)
    # TODO: Generally pragma inlining does not repsect type restriction
    # (eg. omp do pragas to loops) or "post_pragmas". This needs a deeper
    # rethink, so diabling the problematic corner case for now.
    # pairs += PatternFinder(pattern=(Pragma, CallStatement)).visit(ir)
    pairs += PatternFinder(pattern=(Pragma, Loop)).visit(ir)
    mapper = {}
    for pair in pairs:
        # Merge pragma with declaration and delete
        mapper[pair[0]] = None  # Mark for deletion
        mapper[pair[1]] = pair[1]._rebuild(pragma=pair[0])
    return NestedTransformer(mapper, invalidate_source=False).visit(ir)


def inline_labels(ir):
    """
    Find labels and merge them onto the following node.

    Note: This is currently only required for OMNI and OFP frontends which
    has labels as nodes next to the corresponding statement without
    any connection between both.
    """
    pairs = PatternFinder(pattern=(Comment, Assignment)).visit(ir)
    pairs += PatternFinder(pattern=(Comment, Intrinsic)).visit(ir)
    pairs += PatternFinder(pattern=(Comment, Loop)).visit(ir)
    mapper = {}
    for pair in pairs:
        if pair[0].source and pair[0].text == '__STATEMENT_LABEL__':
            if pair[1].source and pair[1].source.lines[0] == pair[0].source.lines[1]:
                mapper[pair[0]] = None  # Mark for deletion
                mapper[pair[1]] = pair[1]._rebuild(label=pair[0].label.lstrip('0'))

    # Remove any stale labels
    for comment in FindNodes(Comment).visit(ir):
        if comment.text == '__STATEMENT_LABEL__':
            mapper[comment] = None
    return NestedTransformer(mapper, invalidate_source=False).visit(ir)


def process_dimension_pragmas(ir):
    """
    Process any '!$loki dimension' pragmas to override deferred dimensions

    Note that this assumes `inline_pragmas` has been run on :param ir: to
    attach any pragmas to the `Declaration` nodes.
    """
    for decl in FindNodes(Declaration).visit(ir):
        if decl.pragma and decl.pragma.keyword == 'loki' and decl.pragma.content.startswith('dimension'):
            for v in decl.variables:
                # Found dimension override for variable
                dims = decl.pragma.content.split('dimension(')[-1]
                dims = dims.split(')')[0].split(',')
                dims = [d.strip() for d in dims]
                shape = []
                for d in dims:
                    if d.isnumeric():
                        shape += [Literal(value=int(d), type=BasicType.INTEGER)]
                    else:
                        _type = SymbolType(BasicType.INTEGER)
                        shape += [Variable(name=d, scope=v.scope, type=_type)]
                v.type = v.type.clone(shape=as_tuple(shape))
    return ir


def read_file(file_path):
    """
    Reads a file and returns the content as string.

    This convenience function is provided to catch read errors due to bad
    character encodings in the file. It skips over these characters and
    prints a warning for the first occurence of such a character.
    """
    filepath = Path(file_path)
    try:
        with filepath.open('r') as f:
            source = f.read()
    except UnicodeDecodeError as excinfo:
        warning('Skipping bad character in input file "%s": %s',
                str(filepath), str(excinfo))
        kwargs = {'mode': 'r', 'encoding': 'utf-8', 'errors': 'ignore'}
        with codecs.open(filepath, **kwargs) as f:
            source = f.read()
    return source


def import_external_symbols(module, symbol_names, scope):
    """
    Import variable and type symbols from an external definition :param module:.

    This ensures that all symbols are copied over to the local scope, in order
    to ensure correct variable and type derivation.
    """
    symbols = []
    for name in symbol_names:
        symbol = None
        if module and name in module.symbols:
            symbol = Variable(name=name, type=module.symbols[name], scope=scope.symbols)

        elif module and name in module.types:
            symbol = module.types[name].dtype
            scope.types[name] = module.types[name]
        else:
            symbol = Variable(name=name, scope=scope.symbols)
        symbols.append(symbol)

    return as_tuple(symbols)
