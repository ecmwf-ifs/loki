# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import networkx as nx

from loki.batch.item import Item, ExternalItem
from loki.tools import as_tuple


__all__ = ['SFilter']


class SFilter:
    """
    Filtered iterator over a :any:`SGraph`

    This class allows to change the iteration behaviour over the dependency graph
    stored in :any:`SGraph`.

    Example use::

      items = scheduler.items
      reversed_items = as_tuple(SFilter(scheduler.sgraph, reverse=True))
      procedure_bindings = as_tuple(SFilter(scheduler.sgraph, item_filter=ProcedureBindingItem))

    Parameters
    ----------
    sgraph : :any:`SGraph`
        The graph over which to iterate
    item_filter : list of :any:`Item` subclasses, optional
        Only include items that match the provided list of types
    reverse : bool, optional
        Iterate over the dependency graph in reverse direction
    exclude_ignored : bool, optional
        Exclude :any:`Item` objects that have the ``is_ignored`` property
    include_external : bool, optional
        Do not skip :any:`ExternalItem` in the iterator
    """

    def __init__(self, sgraph, item_filter=None, reverse=False, exclude_ignored=False, include_external=False):
        self.sgraph = sgraph
        self.reverse = reverse
        if item_filter:
            self.item_filter = item_filter
        else:
            self.item_filter = Item
        self.exclude_ignored = exclude_ignored
        self.include_external = include_external

    def __iter__(self):
        if self.reverse:
            self._iter = iter(reversed(list(nx.topological_sort(self.sgraph._graph))))
        else:
            self._iter = iter(nx.topological_sort(self.sgraph._graph))
        return self

    def __next__(self):
        while node := next(self._iter):
            if isinstance(node, ExternalItem):
                if self.include_external and node.origin_cls in as_tuple(self.item_filter):
                    # We found an ExternalItem that matches the item filter
                    break
                continue
            if isinstance(node, self.item_filter) and not (self.exclude_ignored and node.is_ignored):
                # We found the next item matching the filter (and which is not ignored, if applicable)
                break
        return node
