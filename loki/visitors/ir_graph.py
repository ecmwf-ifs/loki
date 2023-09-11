# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
GraphCollector classes for IR
"""

from itertools import chain
from codetiming import Timer

try:
    from graphviz import Digraph, nohtml

    HAVE_IR_GRAPH = True
    """Indicate wheater the graphviz package is available."""
except ImportError:
    HAVE_IR_GRAPH = False

from loki.tools import JoinableStringList, is_iterable, as_tuple
from loki.visitors.visitor import Visitor

__all__ = ["HAVE_IR_GRAPH", "GraphCollector", "ir_graph"]


class GraphCollector(Visitor):
    """
    Convert a given IR tree to a node and edge list via the visit mechanism.

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
    show_comments : bool, optional, default: False
        Whether to show comments in the output
    show_expressions : bool, optional, default: False
        Whether to further expand expressions in the output
    """

    def __init__(
        self, linewidth=40, symgen=str, show_comments=False, show_expressions=False
    ):
        super().__init__()
        self.linewidth = linewidth
        self._symgen = symgen
        self._id = 0
        self._id_map = {}
        self.show_comments = show_comments
        self.show_expressions = show_expressions

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
        content = ""
        if items:
            content = self.format_line("<", name, " ", self.join_items(items), ">")
        else:
            content = self.format_line("<", name, ">")

        # disregard all quotes to ensure nice graphviz label behaviour
        return content.replace('"', "")

    def format_line(self, *items, comment=None, no_wrap=False):
        """
        Format a line by concatenating all items and applying indentation while observing
        the allowed line width limit.

        Note that the provided comment will simply be extended to the line and no line
        width limit will be enforced for that.

        Parameters
        ----------
        items : list
            The items to be put on that line.
        comment : str
            An optional inline comment to be put at the end of the line.
        no_wrap : bool
            Disable line wrapping.

        Returns
        -------
        str the string of the current line, potentially including line breaks if
                 required to observe the line width limit.
        """

        if no_wrap:
            # Simply concatenate items and extend the comment
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
            return chain.from_iterable(
                as_tuple(self.visit(i, **kwargs) for i in item if i is not None)
            )
        return list(
            chain.from_iterable(
                as_tuple(
                    self.visit(i, **kwargs) for i in [item, *args] if i is not None
                )
            )
        )

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

        Returns
        -------
        list[tuple[dict[str,str], dict[str,str]]]]
            A list of a tuple of a node and potentially a edge information
        """
        label = kwargs.get("label", "")
        if label == "":
            label = self.format_node(repr(node))

        shape = kwargs.get("shape", "oval")

        node_key = str(id(node))
        if node_key not in self._id_map:
            self._id_map[node_key] = str(self._id)
            self._id += 1

        node_info = {
            "name": str(self._id_map[node_key]),
            "label": nohtml(str(label)),
            "shape": str(shape),
        }

        parent = kwargs.get("parent")
        edge_info = {}
        if parent:
            parent_id = self._id_map[str(id(parent))]
            child_id = self._id_map[str(id(node))]
            edge_info = {"tail_name": str(parent_id), "head_name": str(child_id)}

        return [(node_info, edge_info)]

    # Handler for outer objects
    def visit_Module(self, o, **kwargs):
        """
        Add a :any:`Module`, mark parent node and visit all "spec" and "subroutine" nodes.

        Returns
        -------
        list[tuple[dict[str,str], dict[str,str]]]]
            An extended list of tuples of a node and potentially a edge information
        """
        node_edge_info = self.__add_node(o, **kwargs)
        kwargs["parent"] = o

        node_edge_info.extend(self.visit(o.spec, **kwargs))
        node_edge_info.extend(self.visit_all(o.contains, **kwargs))

        return node_edge_info

    def visit_Subroutine(self, o, **kwargs):
        """
        Add a :any:`Subroutine`, mark parent node and visit all "docstring", "spec", "body", "members" nodes.

        Returns
        -------
        list[tuple[dict[str,str], dict[str,str]]]]
            An extended list of tuples of a node and potentially a edge information
        """
        node_edge_info = self.__add_node(o, **kwargs)
        kwargs["parent"] = o

        node_edge_info.extend(self.visit(o.docstring, **kwargs))
        node_edge_info.extend(self.visit(o.spec, **kwargs))
        node_edge_info.extend(self.visit(o.body, **kwargs))
        node_edge_info.extend(self.visit_all(o.contains, **kwargs))

        return node_edge_info

    # Handler for AST base nodes
    def visit_Comment(self, o, **kwargs):
        """
        Enables turning off comments.

        Returns
        -------
        list[tuple[dict[str,str], dict[str,str]]]]
            An extended list of tuples of a node and potentially a edge information, or list of nothing.
        """
        if self.show_comments:
            return self.visit_Node(o, **kwargs)
        return []

    visit_CommentBlock = visit_Comment

    def visit_Node(self, o, **kwargs):
        """
        Add a :any:`Node`, mark parent and visit all children.

        Returns
        -------
        list[tuple[dict[str,str], dict[str,str]]]]
            An extended list of tuples of a node and potentially a edge information
        """
        node_edge_info = self.__add_node(o, **kwargs)
        kwargs["parent"] = o

        node_edge_info.extend(self.visit_all(o.children, **kwargs))
        return node_edge_info

    def visit_Expression(self, o, **kwargs):
        """
        Dispatch routine to add nodes utilizing expression tree stringifier,
        mark parent and stop.

        Returns
        -------
        list[tuple[dict[str,str], dict[str,str]]]]
            An extended list of tuples of a node and potentially a edge information or list of nothing.
        """
        if self.show_expressions:
            content = self.symgen(o)
            parent = kwargs.get("parent")
            return self.__add_node(o, label=content, parent=parent, shape="box")
        return []

    def visit_tuple(self, o, **kwargs):
        """
        Recurse for each item in the tuple.
        """
        return self.visit_all(o, **kwargs)

    visit_list = visit_tuple

    def visit_Conditional(self, o, **kwargs):
        """
        Add a :any:`Conditional`, mark parent and visit first body then else body.

        Returns
        -------
        list[tuple[dict[str,str], dict[str,str]]]]
            An extended list of tuples of a node and potentially a edge information
        """
        parent = kwargs.get("parent")
        label = self.symgen(o.condition)
        node_edge_info = self.__add_node(o, label=label, parent=parent, shape="diamond")
        kwargs["parent"] = o
        node_edge_info.extend(self.visit_all(o.body, **kwargs))

        if o.else_body:
            node_edge_info.extend(self.visit_all(o.else_body, **kwargs))
        return node_edge_info


def ir_graph(
    ir, linewidth=40, symgen=str, show_comments=False, show_expressions=False
):
    """
    Pretty-print the given IR using :class:`GraphCollector`.

    Parameters
    ----------
    ir : :any:`Node`
        The IR node starting from which to produce the tree
    show_comments : bool, optional, default: False
        Whether to show comments in the output
    show_expressions : bool, optional, default: False
        Whether to further expand expressions in the output
    """

    if not HAVE_IR_GRAPH:
        raise ImportError("ir_graph is not available.")

    log = "[Loki::Graph Visualization] Created graph visualization in {:.2f}s"

    with Timer(text=log):
        graph_representation = GraphCollector(linewidth, symgen, show_comments, show_expressions)
        node_edge_info = [item for item in graph_representation.visit(ir) if item is not None]

        graph = Digraph()
        graph.attr(rankdir="LR")
        for node_info, edge_info in node_edge_info:
            if node_info:
                graph.node(**node_info)
            if edge_info:
                graph.edge(**edge_info)
        return graph
