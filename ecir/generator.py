"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

import re
from collections import deque
from itertools import groupby

from ecir.ir import Loop, Statement, Conditional, Comment, CommentBlock, Variable, Expression, Index
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
        bodies = tuple([self.visit(b)] for b in o.findall('body'))
        ncond = len(conditions)
        else_body = bodies[-1] if len(bodies) > ncond else None
        return Conditional(source=source, conditions=conditions,
                           bodies=bodies[:ncond], else_body=else_body)

    def visit_comment(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        return Comment(source=source)

    def visit_assignment(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, source=source)

    def visit_name(self, o):
        indices = tuple(self.visit(i) for i in o.findall('subscripts/subscript'))
        return Variable(name=o.attrib['id'],
                        indices=indices if len(indices) > 0 else None)

    def visit_literal(self, o):
        return o.attrib['value']

    def visit_value(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        return Expression(source=source)

    def visit_subscript(self, o):
        if o.find('range'):
            lower = self.visit(o.find('range/lower-bound'))
            upper = self.visit(o.find('range/upper-bound'))
            return Index(expr='%s:%s' % (lower, upper))
        elif o.find('name'):
            var = self.visit(o.find('name'))
            return Index(expr='%s' % var)
        elif o.find('operation'):
            op = self.visit(o.find('operation'))
            return Index(expr='%s' % op)
        else:
            return Index(expr=':')

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
            if subgroups is not None and len(subgroups) > 0:
                groups += subgroups
        for t, group in groupby(o, lambda o: type(o)):
            g = tuple(group)
            if t is self.node_type and len(g) > 1:
                groups.append(g)
        return groups

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
