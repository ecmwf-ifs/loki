# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import CodeGraph, Item


def test_cgraph_nodes_and_definitions():
    """
    Test basic node and definition-edge handling in CodeGraph.
    """
    file_item = Item('file.f90', source=None)
    module_item = Item('some_mod', source=None)
    routine_item = Item('some_mod#routine', source=None)
    typedef_item = Item('some_mod#some_type', source=None)

    cgraph = CodeGraph()
    cgraph.add_edge((file_item, module_item))
    cgraph.add_edges(((module_item, routine_item), (module_item, typedef_item)))

    assert cgraph.items == (file_item, module_item, routine_item, typedef_item)
    assert cgraph.definitions == (
        (file_item, module_item),
        (module_item, routine_item),
        (module_item, typedef_item),
    )
    assert tuple(cgraph) == cgraph.items


def test_cgraph_add_nodes():
    """
    Test explicit node insertion in CodeGraph.
    """
    items = (Item('a', source=None), Item('b', source=None))

    cgraph = CodeGraph()
    cgraph.add_node(items[0])
    cgraph.add_nodes(items[1:])

    assert cgraph.items == items
    assert not cgraph.definitions
