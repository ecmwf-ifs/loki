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
from itertools import groupby
from more_itertools import replace

from loki.visitors import (
    NestedTransformer, FindNodes, PatternFinder, SequenceFinder, Transformer
)
from loki.ir import (
    Assignment, Comment, CommentBlock, VariableDeclaration, ProcedureDeclaration,
    Loop, Intrinsic, Pragma
)
from loki.frontend.source import Source
from loki.logging import warning, perf

__all__ = [
    'Frontend', 'OFP', 'OMNI', 'FP', 'REGEX', 'inline_comments',
    'ClusterCommentTransformer', 'read_file',
    'combine_multiline_pragmas', 'sanitize_ir'
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


def inline_comments(ir):
    """
    Identify inline comments and merge them onto statements
    """
    pairs = PatternFinder(pattern=(Assignment, Comment)).visit(ir)
    pairs += PatternFinder(pattern=(VariableDeclaration, Comment)).visit(ir)
    pairs += PatternFinder(pattern=(ProcedureDeclaration, Comment)).visit(ir)
    mapper = {}
    for pair in pairs:
        # Comment is in-line and can be merged
        # Note, we need to re-create the statement node
        # so that Transformers don't throw away the changes.
        if pair[0].source and pair[1].source:
            if pair[1].source.lines[0] == pair[0].source.lines[1]:
                mapper[pair[0]] = pair[0]._rebuild(comment=pair[1])
                mapper[pair[1]] = None  # Mark for deletion
    return NestedTransformer(mapper, invalidate_source=False).visit(ir)


class ClusterCommentTransformer(Transformer):
    """
    Combines consecutive sets of :any:`Comment` into a :any:`CommentBlock`.
    """

    def visit_tuple(self, o, **kwargs):
        """
        Find groups of :any:`Comment` and inject into the tuple.
        """
        cgroups = tuple(
            tuple(g) for k, g in groupby(o, key=lambda x: x.__class__)
            if k == Comment
        )
        cgroups = tuple(g for g in cgroups if len(g) > 1)

        for group in cgroups:
            # Combine the group into a CommentBlock
            source = join_source_list(tuple(p.source for p in group))
            block = CommentBlock(comments=group, label=group[0].label, source=source)
            pred = lambda *args: args == group
            o = tuple(replace(
                o, pred=pred, substitutes=(block,), window_size=len(group)
            ))

        # Then recurse over the new nodes
        return tuple(self.visit(i, **kwargs) for i in o)

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


def combine_multiline_pragmas(ir):
    """
    Combine multiline pragmas into single pragma nodes
    """
    pragma_mapper = {}
    pragma_groups = SequenceFinder(node_type=Pragma).visit(ir)
    for pragma_list in pragma_groups:
        collected_pragmas = []
        for pragma in pragma_list:
            if not collected_pragmas:
                if pragma.content.rstrip().endswith('&'):
                    # This is the beginning of a multiline pragma
                    collected_pragmas = [pragma]
            else:
                # This is the continuation of a multiline pragma
                collected_pragmas += [pragma]

                if pragma.keyword != collected_pragmas[0].keyword:
                    raise RuntimeError('Pragma keyword mismatch after line continuation: ' +
                                       f'{collected_pragmas[0].keyword} != {pragma.keyword}')

                if not pragma.content.rstrip().endswith('&'):
                    # This is the last line of a multiline pragma
                    content = [p.content.strip()[:-1].rstrip() for p in collected_pragmas[:-1]]
                    content = ' '.join(content) + ' ' + pragma.content.strip()

                    if all(p.source is not None for p in collected_pragmas):
                        if all(p.source.string is not None for p in collected_pragmas):
                            string = '\n'.join(p.source.string for p in collected_pragmas)
                        else:
                            string = None
                        lines = (collected_pragmas[0].source.lines[0], collected_pragmas[-1].source.lines[1])
                        source = Source(lines=lines, string=string, file=pragma.source.file)
                    else:
                        source = None

                    new_pragma = Pragma(keyword=pragma.keyword, content=content, source=source)
                    pragma_mapper[collected_pragmas[0]] = new_pragma
                    pragma_mapper.update({p: None for p in collected_pragmas[1:]})

                    collected_pragmas = []

    return NestedTransformer(pragma_mapper, invalidate_source=False).visit(ir)


@Timer(logger=perf, text=lambda s: f'[Loki::Frontend] Executed sanitize_ir in {s:.2f}s')
def sanitize_ir(_ir, frontend, pp_registry=None, pp_info=None):
    """
    Utility function to sanitize internal representation after creating it
    from the parse tree of a frontend

    It carries out post-processing according to :data:`pp_info` and applies
    the following operations:

    * :any:`inline_comments` to attach inline-comments to IR nodes
    * :any:`ClusterCommentTransformer` to combine multi-line comments into :any:`CommentBlock`
    * :any:`combine_multiline_pragmas` to combine multi-line pragmas into a
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
    _ir = inline_comments(_ir)
    _ir = ClusterCommentTransformer(invalidate_source=False).visit(_ir)

    if frontend in (OMNI, OFP):
        _ir = inline_labels(_ir)

    if frontend in (FP, OFP):
        _ir = combine_multiline_pragmas(_ir)

    return _ir
