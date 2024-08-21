# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Pretty-printer classes for IR
"""

from sys import stdout

from loki.tools import JoinableStringList, is_iterable, as_tuple
from loki.ir.visitor import Visitor


__all__ = ['Stringifier', 'pprint']


class Stringifier(Visitor):
    """
    Convert a given IR tree to a string representation.

    This serves as base class for backends and provides a number of helpful
    routines that ease implementing automatic recursion and line wrapping.
    It doubles as a means to produce a human readable representation of the
    IR, which is useful for debugging purposes.

    Parameters
    ----------
    depth : int, optional
        The level of indentation to be applied initially.
    indent : str, optional
        The string to be prepended to a line for each level of indentation.
    linewidth : int, optional
        The line width limit after which to break a line.
    line_cont : optional
        A function handle that accepts the current indentation string
        (:attr:`Stringifier.indent`) and returns the string for line
        continuation. This is inserted between two lines when they need to
        wrap to stay within the line width limit. Defaults to newline character
        plus indentation.
    symgen : optional
        A function handle that accepts a :any:`pymbolic.primitives.Expression`
        and produces a string representation for that.
    """

    # pylint: disable=arguments-differ

    def __init__(self, depth=0, indent='  ', linewidth=90,
                 line_cont=lambda indent: '\n' + indent, symgen=str):
        super().__init__()

        self.depth = depth
        self._indent = indent
        self.linewidth = linewidth
        self.line_cont = line_cont
        self._symgen = symgen

    @property
    def symgen(self):
        """
        Formatter for expressions.
        """
        return self._symgen

    @property
    def indent(self):
        """
        Yield indentation string according to current depth.

        Returns
        -------
        str
            A string containing ``indent * depth``.
        """
        return self._indent * self.depth

    @staticmethod
    def join_lines(*lines):
        """
        Combine multiple lines into a long string, inserting line breaks in between.
        Entries that are `None` are skipped.

        Parameters
        ----------
        lines : list
             The lines to be combined.

        Returns
        -------
        str or `None`
            The combined string or `None` if an empty list was given.
        """
        if not lines:
            return None
        return '\n'.join(line for line in lines if line is not None)

    def join_items(self, items, sep=', ', separable=True):
        """
        Concatenate a list of items into :any:`JoinableStringList`.

        The return value can be passed to :meth:`format_line` or
        :meth:`format_node` or converted to a string with `str`, using
        the :any:`JoinableStringList` as an argument.
        Upon expansion, lines will be wrapped automatically to stay within
        the linewidth limit.

        Parameters
        ----------
        items : list
            The list of strings to be joined.
        sep : str, optional
            The separator to be inserted between items.
        separable : bool, optional
            Allow line breaks between individual :data:`items`.

        Returns
        -------
        :any:`JoinableStringList`
        """
        return JoinableStringList(items, sep=sep, width=self.linewidth,
                                  cont=self.line_cont(self.indent), separable=separable)

    def format_node(self, name, *items):
        """
        Default format for a node.

        Creates a string of the form ``<name[, attribute, attribute, ...]>``.
        """
        if items:
            return self.format_line('<', name, ' ', self.join_items(items), '>')
        return self.format_line('<', name, '>')

    def format_line(self, *items, comment=None, no_wrap=False, no_indent=False):
        """
        Format a line by concatenating all items and applying indentation while observing
        the allowed line width limit.

        Note that the provided comment will simply be appended to the line and no line
        width limit will be enforced for that.

        :param list items: the items to be put on that line.
        :param str comment: an optional inline comment to be put at the end of the line.
        :param bool no_wrap: disable line wrapping.
        :param bool no_indent: do not apply indentation.

        :return: the string of the current line, potentially including line breaks if
                 required to observe the line width limit.
        :rtype: str
        """
        # print(f"items: {items}")
        if not no_indent and items != ('',):
            items = [self.indent, *items]
        if no_wrap:
            # Simply concatenate items and append the comment
            line = ''.join(str(item) for item in items)
        else:
            # Use join_items to concatenate items
            line = str(self.join_items(items, sep=''))
        if comment:
            return line + comment
        return line

    def visit_all(self, item, *args, **kwargs):
        """
        Convenience function to call :meth:`visit` for all given arguments.

        If only a single argument is given that is iterable,
        :meth:`visit` is called on all of its elements instead.
        """
        if is_iterable(item) and not args:
            return as_tuple(self.visit(i, **kwargs) for i in item if i is not None)
        return as_tuple(self.visit(i, **kwargs) for i in [item, *args] if i is not None)

    # Handler for outer objects

    def visit_Module(self, o, **kwargs):
        """
        Format a :any:`Module` as

        .. code-block:: none

           <repr(Module)>
             ...spec...
             ...routines...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        spec = self.visit(o.spec, **kwargs)
        routines = self.visit(o.subroutines, **kwargs)
        self.depth -= 1
        return self.join_lines(header, spec, routines)

    def visit_Subroutine(self, o, **kwargs):
        """
        Format a :any:`Subroutine` as

        .. code-block:: none

           <repr(Subroutine)>
             ...docstring...
             ...spec...
             ...body...
             ...members...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        docstring = self.visit(o.docstring, **kwargs)
        spec = self.visit(o.spec, **kwargs)
        body = self.visit(o.body, **kwargs)
        members = self.visit(o.members, **kwargs)
        self.depth -= 1
        return self.join_lines(header, docstring, spec, body, members)

    # Handler for AST base nodes

    def visit_Node(self, o, **kwargs):
        """
        Format a :any:`Node` as

        .. code-block:: none

           <repr(Node)>
        """
        return self.format_node(repr(o))

    def visit_Expression(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Dispatch routine to expression tree stringifier
        :attr:`Stringifier.symgen`.
        """
        return self.symgen(o)

    def visit_tuple(self, o, **kwargs):
        """
        Recurse for each item in the tuple and return as separate lines.
        """
        lines = (self.visit(item, **kwargs) for item in o)
        return self.join_lines(*lines)

    visit_list = visit_tuple

    # Handler for IR nodes

    def visit_InternalNode(self, o, **kwargs):
        """
        Format :any:`InternalNode` as

        .. code-block:: none

           <repr(InternalNode)>
             ...body...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        body = self.visit(o.body, **kwargs)
        self.depth -= 1
        return self.join_lines(header, body)


    def visit_Conditional(self, o, **kwargs):
        """
        Format :any:`Conditional` as

        .. code-block:: none

           <repr(Conditional)>
             <If [condition]>
               ...
             <Else>
               ...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        conditions = [self.format_node('If', self.visit(o.condition, **kwargs))]
        if o.else_body:
            conditions.append(self.format_node('Else'))
        self.depth += 1
        bodies = self.visit_all(o.body, o.else_body, **kwargs)
        self.depth -= 1
        self.depth -= 1
        body = [item for branch in zip(conditions, bodies) for item in branch]
        return self.join_lines(header, *body)

    def visit_MultiConditional(self, o, **kwargs):
        """
        Format :any:`MultiConditional` as

        .. code-block:: none

           <repr(MultiConditional)>
             <Case [value(s)]>
               ...
             <Case [value(s)]>
               ...
             <Default>
               ...
        """
        header = self.format_node(repr(o))
        self.depth += 1
        values = []
        for expr in o.values:
            value = f'({", ".join(self.visit_all(expr, **kwargs))})'
            values += [self.format_node('Case', value)]
        if o.else_body:
            values += [self.format_node('Default')]
        self.depth += 1
        bodies = self.visit_all(*o.bodies, o.else_body, **kwargs)
        self.depth -= 1
        self.depth -= 1
        body = [item for branch in zip(values, bodies) for item in branch]
        return self.join_lines(header, *body)


def pprint(ir, stream=None):
    """
    Pretty-print the given IR using :class:`Stringifier`.

    Parameters
    ----------
    ir : :any:`Node`
        The IR node starting from which to print the tree
    stream : optional
        If given, call :meth:`Stringifier.write` on this stream instead of
        :any:`sys.stdout`
    """
    if stream is None:
        stream = stdout
    stream.write(Stringifier().visit(ir))
