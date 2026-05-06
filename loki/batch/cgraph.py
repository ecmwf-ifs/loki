# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import networkx as nx


__all__ = ['CodeGraph']


class CodeGraph:
    """
    Definition graph representing the visible code structure.

    It is built upon a :any:`networkx.DiGraph`, where nodes are scheduler
    :any:`Item` objects and edges represent definition or containment
    relationships. For example, a :any:`FileItem` defines a :any:`ModuleItem`,
    and a :any:`ModuleItem` defines contained :any:`ProcedureItem` objects.

    Parameters
    ----------
    graph : :any:`networkx.DiGraph`, optional
        Directed graph to use as initial graph.
    """

    def __init__(self, graph=None):
        self._graph = graph or nx.DiGraph()

    def __iter__(self):
        """
        Iterate over the items in the definition graph.
        """
        return iter(self._graph)

    @property
    def items(self):
        """
        Return all :any:`Item` nodes in the definition graph.
        """
        return tuple(self._graph.nodes)

    @property
    def definitions(self):
        """
        Return all definition relationships in the graph.
        """
        return tuple(self._graph.edges)

    def add_node(self, item):
        """
        Add :data:`item` as a node to the definition graph.
        """
        self._graph.add_node(item)

    def add_nodes(self, items):
        """
        Add the given :data:`items` as nodes to the definition graph.
        """
        self._graph.add_nodes_from(items)

    def add_edge(self, edge):
        """
        Add a definition relationship :data:`edge` to the graph.
        """
        self._graph.add_edge(edge[0], edge[1])

    def add_edges(self, edges):
        """
        Add definition relationships :data:`edges` to the graph.
        """
        self._graph.add_edges_from(edges)
