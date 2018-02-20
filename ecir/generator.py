"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

import re
from collections import deque
from itertools import groupby

from ecir.ir import Loop, Statement, Conditional, Comment, CommentBlock, Variable, Expression
from ecir.visitors import Visitor, Transformer, NestedTransformer
from ecir.helpers import assemble_continued_statement_from_list
from ecir.tools import as_tuple

__all__ = ['IRGenerator']


def extract_lines(attrib, source, full_lines=False):
    """
    Extract the marked string from source text.
    """
    lstart = int(attrib['line_begin'])
    lend = int(attrib['line_end'])
    cstart = int(attrib['col_begin'])
    cend = int(attrib['col_end'])

    if isinstance(source, str):
        source = source.splitlines(keepends=False)

    if full_lines:
        return ''.join(source[lstart-1:lend])

    if lstart == lend:
        if len(source[lstart-1]) < cend-1:
            # Final line has line continuations (&), assemble it
            # Note: We trim the final character, since the utility adds a newline
            line = assemble_continued_statement_from_list(lstart-1, source, return_orig=False)[1][:-1]
        else:
            line = source[lstart-1]
        return line[cstart:cend]
    else:
        lines = source[lstart-1:lend]
        firstline = lines[0][cstart:]
        lastline = lines[-1][:cend]
        return ''.join([firstline] + lines[1:-1] + [lastline])


class IRGenerator(Visitor):

    def __init__(self, raw_source):
        super(IRGenerator, self).__init__()

        self._raw_source = raw_source

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        if instance.tag in self._handlers:
            return self._handlers[instance.tag]
        else:
            return super(IRGenerator, self).lookup_method(instance)

    def visit_Element(self, o):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c) for c in o.getchildren())
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        else:
            return children if len(children) > 0 else None

    visit_body = visit_Element

    def visit_loop(self, o):
        source = extract_lines(o.attrib, self._raw_source, full_lines=True)
        variable = o.find('header/index-variable').attrib['name']
        try:
            lower = self.visit(o.find('header/index-variable/lower-bound'))[0]
            upper = self.visit(o.find('header/index-variable/upper-bound'))[0]
        except:
            lower = None
            upper = None
        body = as_tuple(self.visit(o.find('body')))
        return Loop(variable=variable, body=body, source=source, bounds=(lower, upper))

    def visit_if(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        conditions = tuple(self.visit(h) for h in o.findall('header'))
        bodies = tuple(self.visit(b) for b in o.findall('body'))
        if len(bodies) > 2:
            raise NotImplementedError('Else-if conditionals not yet supported')
        then_body = bodies[0]
        else_body = bodies[-1] if len(bodies) > 1 else None
        return Conditional(source=source, condition=conditions[0],
                           then_body=then_body, else_body=else_body)

    def visit_comment(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        return Comment(source=source)

    def visit_assignment(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, source=source)

    def visit_name(self, o):
        indices = tuple(self.visit(i) for i in o.findall('subscripts/subscript/name'))
        return Variable(name=o.attrib['id'],
                        indices=indices if len(indices) > 0 else None)

    def visit_literal(self, o):
        return o.attrib['value']

    def visit_value(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        return Expression(source=source)

    def visit_operation(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        return Expression(source=source)


class SequenceFinder(Visitor):
    """
    Utility visitor that finds repeated nodes of the same type in
    lists/tuples within a given tree.
    """

    def __init__(self, node_type):
        super(SequenceFinder, self).__init__()
        self.node_type = node_type

    def visit_tuple(self, o):
        groups = []
        for c in o:
            subgroups = self.visit(c)
            # print("SUB-GROUPS %s" % str(subgroups))
            if subgroups is not None and len(subgroups) > 0:
                groups += subgroups
        # print("pre-GROUPS %s" % str(groups))
        for t, group in groupby(o, lambda o: type(o)):
            g = tuple(group)
            if t is self.node_type and len(g) > 1:
                groups.append(g)
        # print("final-GROUPS %s" % str(groups))
        return groups# if len(groups) > 0 else None

    visit_list = visit_tuple


def generate(ofp_ast, raw_source):
    """
    Generate an internal IR from the raw OFP (Open Fortran Parser)
    output.

    The internal IR is intended to represent the code at a much higher
    level than the raw langage constructs that OFP returns.
    """

    # Parse the OFP AST into a raw IR
    ir = IRGenerator(raw_source).visit(ofp_ast)

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

    return ir
