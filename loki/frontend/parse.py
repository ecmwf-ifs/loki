"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

from itertools import groupby
from enum import IntEnum

from loki.frontend.ofp2ir import OFP2IR
from loki.frontend.omni2ir import convert_omni2ir
from loki.visitors import Visitor, NestedTransformer
from loki.ir import Statement, Call, Comment, CommentBlock, Declaration, Pragma
from loki.tools import timeit
from loki.logging import DEBUG

__all__ = ['parse', 'Frontend', 'OMNI', 'OFP']


class Frontend(IntEnum):
    OMNI = 1
    OFP = 2

    def __str__(self):
        return self.name.lower()


OMNI = Frontend.OMNI
OFP = Frontend.OFP


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

    def visit_tuple(self, o):
        groups = []
        for c in o:
            # First recurse...
            subgroups = self.visit(c)
            if subgroups is not None and len(subgroups) > 0:
                groups += subgroups
        for t, group in groupby(o, lambda o: type(o)):
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

    def match_indices(self, pattern, sequence):
        """ Return indices of matched patterns in sequence. """
        matches = []
        for i in range(len(sequence)):
            if sequence[i] == pattern[0]:
                if tuple(sequence[i:i+len(pattern)]) == tuple(pattern):
                    matches.append(i)
        return matches

    def visit_tuple(self, o):
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


@timeit(log_level=DEBUG)
def parse(ast, type_map=None, symbol_map=None, raw_source=None, frontend=OFP, cache=None):
    """
    Generate an internal IR from the raw OFP (Open Fortran Parser)
    output.

    The internal IR is intended to represent the code at a much higher
    level than the raw langage constructs that OFP returns.
    """
    if frontend == OFP:
        ir = OFP2IR(raw_source, cache=cache).visit(ast)
    elif frontend == OMNI:
        ir = convert_omni2ir(ast, type_map=type_map, symbol_map=symbol_map,
                             raw_source=raw_source, cache=cache)
    else:
        raise NotImplementedError('Unknown frontend: %s' % frontend)

    # Identify inline comments and merge them onto statements
    pairs = PatternFinder(pattern=(Statement, Comment)).visit(ir)
    pairs += PatternFinder(pattern=(Declaration, Comment)).visit(ir)
    mapper = {}
    for pair in pairs:
        # Comment is in-line and can be merged
        # Note, we need to re-create the statement node
        # so that Transformers don't throw away the changes.
        mapper[pair[0]] = pair[0]._rebuild(comment=pair[1])
        mapper[pair[1]] = None  # Mark for deletion
    ir = NestedTransformer(mapper).visit(ir)

    # Cluster comments into comment blocks
    comment_mapper = {}
    comment_groups = SequenceFinder(node_type=Comment).visit(ir)
    for comments in comment_groups:
        # Build a CommentBlock and map it to first comment
        # and map remaining comments to None for removal
        block = CommentBlock(comments)
        comment_mapper[comments[0]] = block
        for c in comments[1:]:
            comment_mapper[c] = None
    ir = NestedTransformer(comment_mapper).visit(ir)

    # Find pragmas and merge them onto declarations and subroutine calls
    # Note: Pragmas in derived types are already associated with the
    # declaration due to way we parse derived types.
    pairs = PatternFinder(pattern=(Pragma, Declaration)).visit(ir)
    pairs += PatternFinder(pattern=(Pragma, Call)).visit(ir)
    mapper = {}
    for pair in pairs:
        # Merge pragma with declaration and delete
        mapper[pair[0]] = None  # Mark for deletion
        mapper[pair[1]] = pair[1]._rebuild(pragma=pair[0])
    ir = NestedTransformer(mapper).visit(ir)

    return ir
