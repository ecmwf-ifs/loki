from enum import IntEnum
from pathlib import Path
import codecs
from loki.expression.expression import SubstituteExpressions

from loki.visitors import (
    Transformer, NestedTransformer, FindNodes, PatternFinder, SequenceFinder
)
from loki.ir import (
    Assignment, Comment, CommentBlock, Declaration, Loop, Intrinsic, Pragma,
    StatementFunction
)
from loki.expression import Scalar, Array, InlineCall, FindVariables
from loki.types import ProcedureType, SymbolAttributes
from loki.tools import LazyNodeLookup
from loki.frontend.source import Source
from loki.logging import warning

__all__ = [
    'Frontend', 'OFP', 'OMNI', 'FP', 'inline_comments', 'cluster_comments', 'read_file',
    'combine_multiline_pragmas', 'inject_statement_functions'
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

    def __str__(self):
        return self.name.lower()  # pylint: disable=no-member

OMNI = Frontend.OMNI
OFP = Frontend.OFP
FP = Frontend.FP


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
                    raise RuntimeError('Pragma keyword mismatch after line continuation: {} != {}'.format(
                        collected_pragmas[0].keyword, pragma.keyword))

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


def inject_statement_functions(routine):
    """
    Identify statement function definitions and correct their
    representation in the IR

    Fparser misinterprets statement function definitions as array
    assignments and may put them into the subroutine's body instead of
    the spec. This function tries to identify them, correct the type of
    the symbols representing statement functions (as :any:`ProcedureSymbol`)
    and store their definition as :any:`StatementFunction`.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine object for which statement functions should be
        injected
    """
    def create_stmt_func(assignment):
        arguments = assignment.lhs.dimensions
        variable = assignment.lhs.clone(dimensions=None)
        return StatementFunction(variable, arguments, assignment.rhs, variable.type)

    def create_type(stmt_func):
        procedure_query = lambda x: [f for f in FindNodes(StatementFunction).visit(x.spec)
                                        if f.variable == str(stmt_func.variable)][0]
        procedure = LazyNodeLookup(routine, procedure_query)
        return SymbolAttributes(ProcedureType(is_function=True, procedure=procedure))

    # Only locally declared scalar variables are potential candidates
    candidates = [str(v).lower() for v in routine.variables if isinstance(v, Scalar)]

    # First suspects: Array assignments in the spec
    spec_map = {}
    for assignment in FindNodes(Assignment).visit(routine.spec):
        if isinstance(assignment.lhs, Array) and assignment.lhs.name.lower() in candidates:
            stmt_func = create_stmt_func(assignment)
            spec_map[assignment] = stmt_func
            routine.symbols[str(stmt_func.variable)] = create_type(stmt_func)

    # Other suspects: Array assignments at the beginning of the body
    spec_appendix = []
    body_map = {}
    for node in routine.body.body:
        if isinstance(node, (Comment, CommentBlock)):
            spec_appendix += [node]
        if isinstance(node, Assignment) and isinstance(node.lhs, Array) and node.lhs.name.lower() in candidates:
            stmt_func = create_stmt_func(node)
            spec_appendix += [stmt_func]
            body_map[node] = None
            routine.symbols[str(stmt_func.variable)] = create_type(stmt_func)
        else:
            break

    if spec_map or body_map:
        # All statement functions
        stmt_funcs = {node.lhs.name.lower() for node in spec_map}
        stmt_funcs |= {node.lhs.name.lower() for node in body_map}

        # Find any use of the statement functions in the body and replace
        # them with function calls
        expr_map = {}
        for variable in FindVariables().visit(routine.body):
            if variable.name.lower() in stmt_funcs and isinstance(variable, Array):
                parameters = variable.dimensions
                expr_map[variable] = InlineCall(variable.clone(dimensions=None), parameters=parameters)

        # Make sure we remove comments from the body if we append them to spec
        if any(isinstance(node, StatementFunction) for node in spec_appendix):
            body_map.update({node: None for node in spec_appendix if isinstance(node, (Comment, CommentBlock))})

        # Apply transformer with the built maps
        if spec_map:
            routine.spec = Transformer(spec_map).visit(routine.spec)
        if body_map:
            routine.body = Transformer(body_map).visit(routine.body)
            if spec_appendix:
                routine.spec.append(spec_appendix)
        if expr_map:
            routine.body = SubstituteExpressions(expr_map).visit(routine.body)

        # And make sure all symbols have the right type
        routine.rescope_variables()