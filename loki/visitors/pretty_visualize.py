# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Pretty-Graph-Visualizer classes for IR
"""
from codetiming import Timer
from graphviz import Digraph
from loki.tools import JoinableStringList, is_iterable, as_tuple
from loki.visitors.visitor import Visitor

__all__ = ["Visualizer", "pretty_visualize"]

class Visualizer(Visitor):
    """
    Convert a given IR tree to a graphical representation using "graphviz".

    This serves as base class for backends and provides a number of helpful
    routines that ease implementing automatic recursion and line wrapping.
    It is adapted from the Stringifier in "pprint.py". It doubles as a means
    to produce a human readable graph representation of the IR, which is
    useful for debugging purposes and first visualization.

    Parameters
    ----------
    linewidth : int, optional
        The line width limit after which to break a line.
    symgen : str, optional
        A function handle that accepts a :any:`pymbolic.primitives.Expression`
        and produces a string representation for that.
    """

    def __init__(
        self,
        linewidth=40,
        symgen=str,
    ):
        super().__init__()
        self.graph = Digraph()
        self.graph.attr(rankdir="LR")
        self.linewidth = linewidth
        self._symgen = symgen

    @property
    def symgen(self):
        """
        Formatter for expressions.
        """
        return self._symgen

    def join_items(self, items, sep=", ", separable=True):
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
        return JoinableStringList(
            items,
            sep=sep,
            width=self.linewidth,
            cont="\n",
            separable=separable,
        )

    def format_node(self, name, *items):
        """
        Default format for a node.

        Creates a string of the form ``<name[, attribute, attribute, ...]>``.
        """
        if items:
            return self.format_line("<", name, " ", self.join_items(items), ">")
        return self.format_line("<", name, ">")

    def format_line(self, *items, comment=None, no_wrap=False):
        """
        Format a line by concatenating all items and applying indentation while observing
        the allowed line width limit.

        Note that the provided comment will simply be appended to the line and no line
        width limit will be enforced for that.

        :param list items: the items to be put on that line.
        :param str comment: an optional inline comment to be put at the end of the line.
        :param bool no_wrap: disable line wrapping.

        :return: the string of the current line, potentially including line breaks if
                 required to observe the line width limit.
        :rtype: str
        """

        if no_wrap:
            # Simply concatenate items and append the comment
            line = "".join(str(item) for item in items)
        else:
            # Use join_items to concatenate items
            line = str(self.join_items(items, sep=""))
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

    def __add_node(self, node, **kwargs):
        """
        Adds a node to the graphical representation of the IR. Utilizes the
        formatting provided by :meth:`format_node`.

        Parameters
        ----------
        node: :any: `Node` object
        kwargs["shape"]: str, optional (default: "oval")
        kwargs["label"]: str, optional (default: format_node(repr(node)))
        kwargs["parent"]: :any: `Node` object, optional (default: None)
            If not available no edge is drawn.
        """
        label = kwargs.get("label", None)
        if label is None:
            label = self.format_node(repr(node))

        shape = kwargs.get("shape", "oval")
        label = (
            label + "‎"
        )  # dirty hack to force graphviz to utilize parenthesis around the label
        self.graph.node(str(id(node)), label=label, shape=shape)

        parent = kwargs.get("parent")
        if parent:
            self.graph.edge(str(id(parent)), str(id(node)))

    # Handler for outer objects
    def visit_Module(self, o, **kwargs):
        """
        Add a :any:`Module`, mark parent node and visit all "spec" and "subroutine" nodes.
        """
        self.__add_node(o, **kwargs)
        kwargs["parent"] = o

        self.visit(o.spec, **kwargs)
        self.visit(o.subroutines, **kwargs)

    def visit_Subroutine(self, o, **kwargs):
        """
        Add a :any:`Subroutine`, mark parent node and visit all "docstring", "spec", "body", "members" nodes.
        """
        self.__add_node(o, **kwargs)
        kwargs["parent"] = o

        self.visit(o.docstring, **kwargs)
        self.visit(o.spec, **kwargs)
        self.visit(o.body, **kwargs)
        self.visit(o.members, **kwargs)

    # Handler for AST base nodes
    def visit_Comment(self, o, **kwargs):
        """
        Enables turning off comments.
        """
        if kwargs.get("show_comments"):
            self.visit_Node(o, **kwargs)

    visit_CommentBlock = visit_Comment

    def visit_Node(self, o, **kwargs):
        """
        Add a :any:`Node`, mark parent and visit all children.
        """
        self.__add_node(o, **kwargs)
        kwargs["parent"] = o

        self.visit_all(o.children, **kwargs)

    def visit_Expression(self, o, **kwargs):
        """
        Dispatch routine to add nodes utilizing expression tree stringifier,
        mark parent and stop.
        """
        if kwargs.get("show_expressions"):
            content = self.symgen(o)
            parent = kwargs.get("parent")
            self.__add_node(o, label=content, parent=parent, shape="box")

    def visit_tuple(self, o, **kwargs):
        """
        Recurse for each item in the tuple.
        """
        return tuple(self.visit(c, **kwargs) for c in o)

    visit_list = visit_tuple

    def visit_Conditional(self, o, **kwargs):
        """
        Add a :any:`Conditional`, mark parent and visit first body then else body.
        """
        parent = kwargs.get("parent")
        label = self.symgen(o.condition)
        self.__add_node(o, label=label, parent=parent, shape="diamond")
        kwargs["parent"] = o
        self.visit_all(o.body, **kwargs)

        if o.else_body:
            self.visit_all(o.else_body, **kwargs)


def pretty_visualize(ir, **kwargs):
    """
    Pretty-print the given IR using :class:`Visualizer`.

    Parameters
    ----------
    ir : :any:`Node`
        The IR node starting from which to produce the tree
    kwargs["filename"] : str, optional, default: "graph_representation"
        The location, name and format of the output graph
    kwargs["show_comments"] : bool, optional, default: True
        Whether to show comments in the output
    kwargs["show_expressions"] : bool, optional, default: False
        Whether to further expand expressions in the output
    """
    filename = kwargs.get("filename")
    if filename is None:
        filename = "graph_representation"
        kwargs["filename"] = filename

    log = f"[Loki::Graph Visualization] Visualized to {filename}" + " in {:.2f}s"
    with Timer(text=log):
        visualizer = Visualizer()
        visualizer.visit(ir, **kwargs)

        visualizer.graph.render(**kwargs)
