# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import deque, defaultdict
from pathlib import Path
from codetiming import Timer
import networkx as nx

from loki.batch.configure import SchedulerConfig
from loki.batch.item import (
    InterfaceItem, ProcedureItem, ProcedureBindingItem, TypeDefItem
)
from loki.batch.sfilter import SFilter
from loki.logging import info, warning, debug
from loki.tools import as_tuple


__all__ = ['SGraph']


class SGraph:
    """
    The dependency graph underpinning the :any:`Scheduler`

    It is built upon a :any:`networkx.DiGraph` to expose the dependencies
    between :any:`Item` nodes. It is typically created from one or multiple
    `seed` items via :meth:`from_seed` by recursively chasing dependencies.

    Cyclic dependencies are broken for procedures that are marked as
    ``RECURSIVE``, which would otherwise constitute a dependency on itself.
    See :meth:`_break_cycles`.
    """

    def __init__(self):
        self._graph = nx.DiGraph()

    @classmethod
    @Timer(logger=info, text='[Loki::Scheduler] Built SGraph from seed in {:.2f}s')
    def from_seed(cls, seed, item_factory, config=None):
        """
        Create a new :any:`SGraph` using :data:`seed` as starting point.

        Parameters
        ----------
        seed : (list of) str
            The names of the root nodes
        item_factory : :any:`ItemFactory`
            The item factory to use when creating graph nodes
        config : :any:`SchedulerConfig`, optional
            The config object to use when creating items
        """
        _graph = cls()
        _graph._populate(seed, item_factory, config)
        _graph._break_cycles()
        return _graph

    def as_filegraph(self, item_factory, config=None, item_filter=None, exclude_ignored=False):
        """
        Convert the :any:`SGraph` to a dependency graph that only contains
        :any:`FileItem` nodes.

        Parameters
        ----------
        item_factory : :any:`ItemFactory`
            The item factory to use when creating graph nodes
        config : :any:`SchedulerConfig`, optional
            The config object to use when creating items
        item_filter : list of :any:`Item` subclasses, optional
            Only include files that include at least one dependency item of the
            given type. By default, all items are included.
        exclude_ignored : bool, optional
            Exclude :any:`Item`s that have the ``is_ignored`` property

        Returns
        -------
        :any:`SGraph`
            A new graph object
        """
        _graph = type(self)()
        _graph._populate_filegraph(self, item_factory, config, item_filter, exclude_ignored=exclude_ignored)
        return _graph

    def _create_item(self, name, item_factory, config):
        """
        Utility method to create a new item node with the given :data:`name`

        This may trigger on-demand creation of definition items in
        the enclosing scope.
        """
        if '#' not in name:
            name = f'#{name}'
        item = item_factory.item_cache.get(name)

        if not item:
            # We may have to create the corresponding module's definitions first to make
            # the item available in the cache
            scope_name = name[:name.index('#')]
            module_item = item_factory.item_cache.get(scope_name)
            if module_item:
                module_item.create_definition_items(item_factory=item_factory, config=config)
                item = item_factory.item_cache.get(name)

        if not item:
            # The name may be a module procedure or type that is not fully qualified,
            # so we need to search all modules for any matching routines
            if '%' in name:
                module_member_name = name[name.index('#')+1:name.index('%')]
            else:
                module_member_name = name[name.index('#')+1:]
            item = item_factory.get_or_create_module_definitions_from_candidates(
                module_member_name, config, only=(ProcedureItem, TypeDefItem)
            ) or None

            if item and '%' in name:
                # If this is a type-bound procedure, we may have to create its definitions
                for _item in item:
                    _item.create_definition_items(item_factory=item_factory, config=config)
                item = item_factory.item_cache.get(name)

        return item

    def _add_children(self, item, item_factory, config, dependencies=None):
        """
        Create items for dependencies of the :data:`item` and add them to
        the graph as a dependency of :data:`item`

        Parameters
        ----------
        item : :any:`Item`
            Create the dependencies for this item
        item_factory : :any:`ItemFactory`
            The item factory to use when creating graph nodes
        config : :any:`SchedulerConfig`, optional
            The config object to use when creating items
        dependencies : list, optional
            An initial list of already created dependencies

        Returns
        -------
        list of :any:`Item`
            The list of new items that have been added to the graph
        """
        dependencies = as_tuple(dependencies)
        for dependency in item.create_dependency_items(item_factory=item_factory, config=config):
            if not (dependency in dependencies or SchedulerConfig.match_item_keys(dependency.name, item.block)):
                dependency.config['is_ignored'] = (
                    item.is_ignored or
                    bool(SchedulerConfig.match_item_keys(dependency.name, item.ignore, match_item_parents=True))
                )
                dependencies += (dependency,)

        new_items = tuple(item_ for item_ in dependencies if item_ not in self._graph)
        if new_items:
            self.add_nodes(new_items)

        # Careful not to include cycles (from recursive TypeDefs)
        self.add_edges((item, item_) for item_ in dependencies if not item == item_)
        return new_items

    def _populate(self, seed, item_factory, config):
        """
        Build the dependency graph, initialised from :data:`seed` using :data:`item_factory`
        to create the node items

        Parameters
        ----------
        seed : (list of) str
            The names of the seed items
        item_factory : :any:`ItemFactory`
            The item factory to use when creating graph nodes
        config : :any:`SchedulerConfig`, optional
            The config object to use when creating items
        """
        queue = deque()

        # Insert the seed objects
        for name in as_tuple(seed):
            item = as_tuple(self._create_item(name, item_factory, config))
            if item:
                self.add_nodes(item)
                queue.extend(item)
            else:
                debug('No item found for seed "%s"', name)

        # Populate the graph
        while queue:
            item = queue.popleft()
            if item.expand:
                children = self._add_children(item, item_factory, config)
                if children:
                    queue.extend(children)

    def _populate_filegraph(self, sgraph, item_factory, config=None, item_filter=None, exclude_ignored=False):
        """
        Derive a dependency graph with :any:`FileItem` nodes from a given :data:`sgraph`

        Parameters
        ----------
        sgraph : :any:`SGraph`
            The dependency graph from which to derive the file graph
        item_factory : :any:`ItemFactory`
            The item factory to use when creating graph nodes
        config : :any:`SchedulerConfig`, optional
            The config object to use when creating items
        item_filter : list of :any:`Item` subclasses, optional
            Only include files that include at least one dependency item of the
            given type. By default, all items are included.
        exclude_ignored : bool, optional
            Exclude :any:`Item`s that have the ``is_ignored`` property
        """
        item_2_file_item_map = {}
        file_item_2_item_map = defaultdict(list)

        # Add the file nodes for each of the items matching the filter criterion
        for item in SFilter(sgraph, item_filter, exclude_ignored=exclude_ignored):
            file_item = item_factory.get_or_create_file_item_from_source(item.source, config)
            item_2_file_item_map[item.name] = file_item
            file_item_2_item_map[file_item.name] += [item]
            if file_item not in self._graph:
                self.add_node(file_item)

        # Update the "is_ignored" attribute for file items
        for items in file_item_2_item_map.values():
            is_ignored = all(item.is_ignored for item in items)
            item_2_file_item_map[items[0]].config['is_ignored'] = is_ignored

        # Insert edges to the file items corresponding to the successors of the items
        for item in SFilter(sgraph, item_filter, exclude_ignored=exclude_ignored):
            file_item = item_2_file_item_map[item.name]
            for child in sgraph._graph.successors(item):
                child_file_item = item_2_file_item_map.get(child.name)
                if not child_file_item or child_file_item == file_item:
                    # Skip 2 situations:
                    # 1) The child_file_item is None, i.e., not in item_2_file_item_map, if
                    #    the child does not match the item_filter
                    # 2) The child may be the same as the file if there is a dependency to
                    #    another item in the same file
                    continue
                self.add_edge((file_item, child_file_item))

    def _break_cycles(self):
        """
        Remove cyclic dependencies by deleting the first outgoing edge of
        each cyclic dependency for all procedure items with a ``RECURSIVE`` prefix
        """
        for item in self.items:  # We cannot iterate over the graph itself as we plan on changing it
            if (
                isinstance(item, ProcedureItem) and
                any('recursive' in prefix.lower() for prefix in item.ir.prefix or [])
            ):
                try:
                    while True:
                        cycle_path = nx.find_cycle(self._graph, item)
                        debug(f'Removed edge {cycle_path[0]!s} to break cyclic dependency {cycle_path!s}')
                        self._graph.remove_edge(*cycle_path[0])
                except nx.NetworkXNoCycle:
                    pass


    def __iter__(self):
        """
        Iterate over the items in the dependency graph
        """
        return iter(SFilter(self))

    @property
    def items(self):
        """
        Return all :any:`Item` nodes in the dependency graph
        """
        return tuple(self._graph.nodes)

    @property
    def dependencies(self):
        """
        Return all dependencies, i.e., edges of the dependency graph
        """
        return tuple(self._graph.edges)

    def successors(self, item, item_filter=None):
        """
        Return the list of successor nodes in the dependency tree below :any:`Item`

        This returns all immediate successors (but can be filtered accordingly using
        the item's ``targets`` property) of the item in the dependency graph

        The list of successors is provided to transformations during processing with
        the :any:`Scheduler`.

        Parameters
        ----------
        item : :any:`Item`
            The item node in the dependency graph for which to determine the successors
        item_filter : list of :any:`Item` subclasses, optional
            Filter successor items to only include items of the provided type. By default,
            all items are considered. Note that including :any:`ProcedureItem` in the
            ``item_filter`` automatically adds :any:`ProcedureBindingItem` and
            :any:`InterfaceItem` as well, since these are intermediate nodes. Their
            dependencies will also be included until they eventually resolve to a
            :any:`ProcedureItem`.
        """
        item_filter = as_tuple(item_filter) or None
        if item_filter and ProcedureItem in item_filter:
            # ProcedureBindingItem and InterfaceItem are intermediate nodes that take
            # essentially the role of an edge to ProcedureItems. Therefore
            # we need to make sure these are included if ProcedureItems are included
            if ProcedureBindingItem not in item_filter:
                item_filter = item_filter + (ProcedureBindingItem,)
            if InterfaceItem not in item_filter:
                item_filter = item_filter + (InterfaceItem,)

        successors = ()
        for child in self._graph.successors(item):
            if item_filter is None or isinstance(child, item_filter):
                if isinstance(child, (ProcedureBindingItem, InterfaceItem)):
                    successors += (child,) + self.successors(child)
                else:
                    successors += (child,)
        return successors

    @property
    def depths(self):
        """
        Return a mapping of :any:`Item` nodes to their depth (topological generation)
        in the dependency graph
        """
        topological_generations = list(nx.topological_generations(self._graph))
        depths = {
            item: i_gen
            for i_gen, gen in enumerate(topological_generations)
            for item in gen
        }
        return depths

    def add_node(self, item):
        """
        Add :data:`item` as a node to the dependency graph
        """
        self._graph.add_node(item)

    def add_nodes(self, items):
        """
        Add the given :data:`items` as nodes to the dependency graph
        """
        self._graph.add_nodes_from(items)

    def add_edge(self, edge):
        """
        Add a dependency :data:`edge` to the dependency graph
        """
        self._graph.add_edge(edge[0], edge[1])

    def add_edges(self, edges):
        """
        Add the dependency :data:`edges` to the dependency graph
        """
        self._graph.add_edges_from(edges)

    def export_to_file(self, dotfile_path):
        """
        Generate a dotfile from the current graph

        Parameters
        ----------
        dotfile_path : str or pathlib.Path
            Path to write the dotfile to. A corresponding graphical representation
            will be created with an additional ``.pdf`` appendix.
        """
        try:
            import graphviz as gviz  # pylint: disable=import-outside-toplevel
        except ImportError:
            warning('[Loki] Failed to load graphviz, skipping file export generation...')
            return

        path = Path(dotfile_path)
        graph = gviz.Digraph(format='pdf', strict=True, graph_attr=(('rankdir', 'LR'),))

        # Insert all nodes in the graph
        style = {
            'color': 'black', 'shape': 'box', 'fillcolor': 'limegreen', 'style': 'filled'
        }
        for item in self.items:
            graph.node(item.name.upper(), **style)

        # Insert all edges in the schedulers graph
        graph.edges((a.name.upper(), b.name.upper()) for a, b in self.dependencies)

        try:
            graph.render(path, view=False)
        except gviz.ExecutableNotFound as e:
            warning(f'[Loki] Failed to render callgraph due to graphviz error:\n  {e}')
