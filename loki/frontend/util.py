# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from enum import IntEnum
from pathlib import Path
import codecs
from codetiming import Timer
from more_itertools import split_after

from loki.expression import (
    symbols as sym, SubstituteExpressionsMapper, ExpressionRetriever
)
from loki.ir import (
    NestedTransformer, FindNodes, PatternFinder, Transformer,
    Assignment, Comment, CommentBlock, VariableDeclaration,
    ProcedureDeclaration, Loop, Intrinsic, Pragma
)
from loki.frontend.source import join_source_list
from loki.logging import warning, perf, error
from loki.tools import group_by_class, replace_windowed, as_tuple


__all__ = [
    'Frontend', 'OFP', 'OMNI', 'FP', 'REGEX', 'available_frontends',
    'read_file', 'InlineCommentTransformer',
    'ClusterCommentTransformer', 'CombineMultilinePragmasTransformer',
    'sanitize_ir'
]


class Frontend(IntEnum):
    """
    Enumeration to identify available frontends.
    """
    #: The OMNI compiler frontend
    OMNI = 1
    #: The Open Fortran Parser
    OFP = 2
    #: Fparser 2 from STFC
    FP = 3
    #: Reduced functionality parsing using regular expressions
    REGEX = 4

    def __str__(self):
        return self.name.lower()  # pylint: disable=no-member

OMNI = Frontend.OMNI
OFP = Frontend.OFP
FP = Frontend.FP
REGEX = Frontend.REGEX


def available_frontends(xfail=None, skip=None, include_regex=False):
    """
    Provide list of available frontends to parametrize tests with

    To run tests for every frontend, an argument :attr:`frontend` can be added to
    a test with the return value of this function as parameter.

    For any unavailable frontends where ``HAVE_<frontend>`` is `False` (e.g.
    because required dependencies are not installed), :attr:`test` is marked as
    skipped.

    Use as

    ..code-block::
        @pytest.mark.parametrize('frontend', available_frontends(xfail=[OMNI, (OFP, 'Because...')]))
        def my_test(frontend):
            source = Sourcefile.from_file('some.F90', frontend=frontend)
            # ...

    Parameters
    ----------
    xfail : list, optional
        Provide frontends that are expected to fail, optionally as tuple with reason
        provided as string. By default `None`
    skip : list, optional
        Provide frontends that are always skipped, optionally as tuple with reason
        provided as string. By default `None`
    include_regex : bool, optional
        Include the :any:`REGEX` frontend in the list. By default `false`.
    """
    if xfail:
        xfail = dict((as_tuple(f) + (None,))[:2] for f in xfail)
    else:
        xfail = {}

    if skip:
        skip = dict((as_tuple(f) + (None,))[:2] for f in skip)
    else:
        skip = {}

    try:
        import pytest  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        error('Pytest is not installed.')
        raise e

    from loki import frontend  # pylint: disable=import-outside-toplevel,cyclic-import

    # Unavailable frontends
    unavailable_frontends = {
        f: f'{f} is not available' for f in Frontend
        if not getattr(frontend, f'HAVE_{str(f).upper()}')
    }
    skip.update(unavailable_frontends)

    # Build the list of parameters
    params = []
    for f in Frontend:
        if f in skip:
            params += [pytest.param(f, marks=pytest.mark.skip(reason=skip[f]))]
        elif f in xfail:
            params += [pytest.param(f, marks=pytest.mark.xfail(reason=xfail[f]))]
        elif f != REGEX or include_regex:
            params += [f]

    return params


def match_type_pattern(pattern, sequence):
    """
    Match elements in a sequence according to a pattern of their types.

    Parameters
    ----------
    patter: list of type
        A list of types of the pattern to match
    sequence : list
        The list of items from which to match elements
    """
    idx = []
    types = tuple(map(type, sequence))
    for i, elem in enumerate(types):
        if elem == pattern[0]:
            if tuple(types[i:i+len(pattern)]) == tuple(pattern):
                idx.append(i)

    # Return a list of element matches
    return [sequence[i:i+len(pattern)] for i in idx]


class InlineCommentTransformer(Transformer):
    """
    Identify inline comments and merge them onto statements
    """

    def visit_tuple(self, o, **kwargs):
        pairs = match_type_pattern(pattern=(Assignment, Comment), sequence=o)
        pairs += match_type_pattern(pattern=(VariableDeclaration, Comment), sequence=o)
        pairs += match_type_pattern(pattern=(ProcedureDeclaration, Comment), sequence=o)

        for pair in pairs:
            # Comment is in-line and can be merged
            if pair[0].source and pair[1].source:
                if pair[1].source.lines[0] == pair[0].source.lines[1]:
                    new = pair[0]._rebuild(comment=pair[1])
                    o = replace_windowed(o, pair, new)

        # Then recurse over the new nodes
        visited = tuple(self.visit(i, **kwargs) for i in o)

        # Strip empty sublists/subtuples or None entries
        return tuple(i for i in visited if i is not None and as_tuple(i))

    visit_list = visit_tuple


class ClusterCommentTransformer(Transformer):
    """
    Combines consecutive sets of :any:`Comment` into a :any:`CommentBlock`.
    """

    def visit_tuple(self, o, **kwargs):
        """
        Find groups of :any:`Comment` and inject into the tuple.
        """
        cgroups = group_by_class(o, Comment)
        for group in cgroups:
            # Combine the group into a CommentBlock
            source = join_source_list(tuple(p.source for p in group))
            block = CommentBlock(comments=group, label=group[0].label, source=source)
            o = replace_windowed(o, group, subs=(block,))

        # Then recurse over the new nodes
        visited = tuple(self.visit(i, **kwargs) for i in o)

        # Strip empty sublists/subtuples or None entries
        return tuple(i for i in visited if i is not None and as_tuple(i))

    visit_list = visit_tuple


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


class CombineMultilinePragmasTransformer(Transformer):
    """
    Combine multiline :any:`Pragma` nodes into single ones.
    """

    def visit_tuple(self, o, **kwargs):
        """
        Finds multi-line pragmas and combines them in-place.
        """
        pgroups = group_by_class(o, Pragma)

        for group in pgroups:
            # Separate sets of consecutive multi-line pragmas
            pred = lambda p: not p.content.rstrip().endswith('&')  # pylint: disable=unnecessary-lambda-assignment
            for pragmaset in split_after(group, pred=pred):
                # Combine into a single pragma and add to map
                source = join_source_list(tuple(p.source for p in pragmaset))
                content = ' '.join(p.content.rstrip(' &') for p in pragmaset)
                new_pragma = Pragma(
                    keyword=pragmaset[0].keyword, content=content, source=source
                )
                o = replace_windowed(o, pragmaset, subs=(new_pragma,))

        visited = tuple(self.visit(i, **kwargs) for i in o)

        # Strip empty sublists/subtuples or None entries
        return tuple(i for i in visited if i is not None and as_tuple(i))


class RangeIndexTransformer(Transformer):
    """
    :any:`Transformer` that replaces ``arr(1:n)`` notations with
    ``arr(n)`` in :any:`VariableDeclaration`.
    """

    retriever = ExpressionRetriever(lambda e: isinstance(e, (sym.Array)))

    @staticmethod
    def is_one_index(dim):
        return isinstance(dim, sym.RangeIndex) and dim.lower == 1 and dim.step is None

    def visit_VariableDeclaration(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Gets all :any:`Array` symbols and adjusts dimension and shape.
        """
        vmap = {}
        for v in self.retriever.retrieve(o.symbols):
            dimensions = tuple(d.upper if self.is_one_index(d) else d for d in v.dimensions)
            _type = v.type
            if _type.shape:
                shape = tuple(d.upper if self.is_one_index(d) else d for d in _type.shape)
                _type = _type.clone(shape=shape)
            vmap[v] = v.clone(dimensions=dimensions, type=_type)

        mapper = SubstituteExpressionsMapper(vmap, invalidate_source=self.invalidate_source)
        return o.clone(symbols=mapper(o.symbols, recurse_to_declaration_attributes=True))


@Timer(logger=perf, text=lambda s: f'[Loki::Frontend] Executed sanitize_ir in {s:.2f}s')
def sanitize_ir(_ir, frontend, pp_registry=None, pp_info=None):
    """
    Utility function to sanitize internal representation after creating it
    from the parse tree of a frontend

    It carries out post-processing according to :data:`pp_info` and applies
    the following operations:

    * :any:`inline_comments` to attach inline-comments to IR nodes
    * :any:`ClusterCommentTransformer` to combine multi-line comments into :any:`CommentBlock`
    * :any:`CombineMultilinePragmasTransformer` to combine multi-line pragmas into a
      single node

    Parameters
    ----------
    _ir : :any:`Node`
        The root node of the internal representation tree to be processed
    frontend : :any:`Frontend`
        The frontend from which the IR was created
    pp_registry: dict, optional
        Registry of pre-processing items to be applied
    pp_info : optional
        Information from internal preprocessing step that was applied to work around
        parser limitations and that should be re-inserted
    """
    # Apply postprocessing rules to re-insert information lost during preprocessing
    if pp_info is not None and pp_registry is not None:
        for r_name, rule in pp_registry.items():
            info = pp_info.get(r_name, None)
            _ir = rule.postprocess(_ir, info)

    # Perform some minor sanitation tasks
    _ir = InlineCommentTransformer(inplace=True, invalidate_source=False).visit(_ir)
    _ir = ClusterCommentTransformer(inplace=True, invalidate_source=False).visit(_ir)

    if frontend == OMNI:
        # Revert OMNI's array dimension expansion from `a(n)` => `arr(1:n)`
        _ir = RangeIndexTransformer(invalidate_source=False).visit(_ir)

    if frontend in (OMNI, OFP):
        _ir = inline_labels(_ir)

    if frontend in (FP, OFP):
        _ir = CombineMultilinePragmasTransformer(inplace=True, invalidate_source=False).visit(_ir)

    return _ir
