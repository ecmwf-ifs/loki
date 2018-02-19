"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

import re
from collections import deque

from ecir.ir import Loop, InlineComment
from ecir.visitors import GenericVisitor

__all__ = ['IRGenerator']


def extract_lines(attrib, lines, single_line=False):
    """
    Extract the marked string from source text.
    """
    lstart = int(attrib['line_begin'])
    lend = int(attrib['line_end'])
    if single_line:
        assert(lstart==lend)
        cstart = int(attrib['col_begin'])
        cend = int(attrib['col_end'])
        line = lines[lstart-1]
        return line[cstart-1:cend]
    else:
        return lines[lstart-1:lend]


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
        source = extract_lines(o.attrib, self._raw_source, single_line=False)
        return Loop(children=self.visit(o.find('body')), source=''.join(source))

    def visit_comment(self, o):
        # This seems to refer to inline-comment only...
        source = extract_lines(o.attrib, self._raw_source, single_line=True)
        return InlineComment(source=source)


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
