from enum import IntEnum
from pathlib import Path
import codecs

from loki.visitors import NestedTransformer, FindNodes, PatternFinder, SequenceFinder
from loki.ir import Assignment, Comment, CommentBlock, Declaration, Loop, Intrinsic
from loki.frontend.source import Source
from loki.expression import Variable
from loki.tools import as_tuple
from loki.logging import warning

__all__ = [
    'Frontend', 'OFP', 'OMNI', 'FP', 'inline_comments', 'cluster_comments', 'read_file',
]


class Frontend(IntEnum):
    OMNI = 1
    OFP = 2
    FP = 3

    def __str__(self):
        return self.name.lower()  # pylint: disable=no-member


OMNI = Frontend.OMNI
OFP = Frontend.OFP
FP = Frontend.FP  # The STFC FParser


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
            symbol = Variable(name=name, type=module.symbols[name], scope=scope)

        elif module and name in module.types:
            symbol = module.types[name]
            scope.types[name] = module.types[name]
        else:
            symbol = Variable(name=name, scope=scope)
        symbols.append(symbol)

    return as_tuple(symbols)
