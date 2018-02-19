"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

import re
from collections import deque

from ecir.ir import Loop, Statement, InlineComment, Variable
from ecir.visitors import GenericVisitor
from ecir.helpers import assemble_continued_statement_from_list

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
        return line[cstart:cend+1]
    else:
        lines = source[lstart-1:lend]
        firstline = lines[0][cstart:]
        lastline = lines[-1][:cend]
        return ''.join([firstline] + lines[1:-1] + [lastline])


class IRGenerator(GenericVisitor):

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
        body = self.visit(o.find('body'))
        return Loop(variable=variable, children=body, source=source,
                    bounds=(lower, upper))

    def visit_comment(self, o):
        # This seems to refer to inline-comment only...
        source = extract_lines(o.attrib, self._raw_source)
        return InlineComment(source=source)

    def visit_assignment(self, o):
        source = extract_lines(o.attrib, self._raw_source)
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, source=source)

    def visit_name(self, o):
        indices = tuple(self.visit(i) for i in o.findall('subscripts/subscript/name'))
        return Variable(name=o.attrib['id'],
                        indices=indices if len(indices) > 0 else None)

    def visit_value(self, o):
        return extract_lines(o.attrib, self._raw_source)


def generate(ofp_ast, raw_source):
    """
    Generate an internal IR from the raw OFP (Open Fortran Parser)
    output.

    The internal IR is intended to represent the code at a much higher
    level than the raw langage constructs that OFP returns.
    """

    # Parse the OFP AST into a raw IR
    ir = IRGenerator(raw_source).visit(ofp_ast)

    # TODO: Post-process, like clustering comments, etc...

    return ir
