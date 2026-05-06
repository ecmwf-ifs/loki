# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import copy

import networkx as nx

from loki.batch.item_factory import ItemFactory


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

    @classmethod
    def from_file_item(cls, file_item, config=None):
        """
        Create a file-local :any:`CodeGraph` from a :any:`FileItem`.

        This uses a local :any:`ItemFactory` and returns both the graph and the
        locally-created items. The caller is responsible for importing the local
        items into the canonical scheduler-owned factory.

        Parameters
        ----------
        file_item : :any:`FileItem`
            File item from which to recursively discover definitions.
        config : :any:`SchedulerConfig`, optional
            Scheduler config to use when creating items.
        """
        item_factory = ItemFactory()
        item_factory.import_items((file_item,))

        cgraph = cls()
        queue = [file_item]
        visited = set()

        while queue:
            item = queue.pop(0)
            if item in visited:
                continue
            visited.add(item)

            cgraph.add_node(item)
            for child in item.create_definition_items(item_factory=item_factory, config=config):
                cgraph.add_node(child)
                cgraph.add_edge((item, child))
                queue.append(child)

        return cgraph, tuple(item_factory.item_cache.values())

    @classmethod
    def from_file_items(cls, file_items, item_factory, config=None):
        """
        Create a :any:`CodeGraph` from a sequence of :any:`FileItem` objects.

        Each file is discovered independently with :meth:`from_file_item`, then
        merged into :data:`item_factory` using :meth:`ItemFactory.import_items`.
        This preserves a serial implementation while keeping the per-file
        discovery boundary suitable for later parallelization.

        Parameters
        ----------
        file_items : tuple of :any:`FileItem`
            File items from which to discover definition subtrees.
        item_factory : :any:`ItemFactory`
            Canonical item factory into which local file items are imported.
        config : :any:`SchedulerConfig`, optional
            Scheduler config to use when creating items.
        """
        cgraph = cls()

        for file_item in file_items:
            local_cgraph, local_items = cls.from_file_item(file_item, config=config)
            canonical = item_factory.import_items(local_items)

            for item in local_cgraph.items:
                cgraph.add_node(canonical[item])
            for parent, child in local_cgraph.definitions:
                cgraph.add_edge((canonical[parent], canonical[child]))

        return cgraph

    def __iter__(self):
        """
        Iterate over the items in the definition graph.
        """
        return iter(self._graph)

    def snapshot(self):
        """
        Return a shallow item snapshot of this graph.

        This preserves item names and graph structure at the time the snapshot
        is created, while still sharing the underlying source objects.
        """
        item_map = {item: copy(item) for item in self.items}
        graph = type(self)()
        graph.add_nodes(item_map.values())
        graph.add_edges((item_map[parent], item_map[child]) for parent, child in self.definitions)
        return graph

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
