"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

import re
from collections import deque

from ecir.visitors import GenericVisitor

__all__ = ['CodeBlock', 'Loop', 'IRGenerator']


class CodeBlock(object):
    """
    Internal representation of an arbitrary piec of source code.
    """

    def __init__(self, source):
        self._source = source


class Loop(CodeBlock):
    """
    Internal representation of a loop in source code.

    Importantly, this object will carry around an exact copy of the
    source string that defines it's body.
    """

    def __init__(self, source, children=None):
        self._source = source
        self._children = children


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
        # Extract the relevant source lines and recurse over loop body
        lstart = int(o.attrib['line_begin'])
        lend = int(o.attrib['line_end'])
        return Loop(children=self.visit(o.find('body')),
                    source=self._raw_source[lstart:lend])

    def visit_statement(self, o):
        print("STMT %s::%s" % (o.tag, o.attrib.items()))
