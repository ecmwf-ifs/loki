# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import deque
from pathlib import Path
import networkx as nx
import pytest

from loki import (
    HAVE_FP, HAVE_OFP, REGEX, RegexParserClass, as_tuple,
    FileItem, ModuleItem, ProcedureItem, TypeDefItem, ProcedureBindingItem, GlobalVariableItem,
    Sourcefile, Section, RawSource, Import, CallStatement
)

pytestmark = pytest.mark.skipif(not HAVE_FP and not HAVE_OFP, reason='Fparser and OFP not available')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def test_file_item(here):
    proj = here/'sources/projBatch'

    def get_item(path, parser_classes):
        filepath = proj/path
        return FileItem(
            filepath.name.lower(),
            Sourcefile.from_file(filepath, frontend=REGEX, parser_classes=parser_classes)
        )

    # A file with simple module that contains a single subroutine
    item = get_item('module/a_mod.F90', RegexParserClass.EmptyClass)
    assert item.name == 'a_mod.f90'
    assert item.ir is item.source
    # The file is not parsed at all
    assert not item.source.definitions
    assert isinstance(item.source.ir, Section)
    assert len(item.source.ir.body) == 1
    assert isinstance(item.source.ir.body[0], RawSource)

    # Querying definitions triggers a round of parsing
    assert item.definitions == (item.source['a_mod'],)
    assert len(item.source.definitions) == 1
    items = item.create_definition_items(item_cache={})
    assert len(items) == 1
    assert items[0].name == 'a_mod'
    assert items[0].definitions == (item.source['a'],)

    item = get_item('module/a_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 'a_mod.f90'
    assert item.definitions == (item.source['a_mod'],)
    assert item.ir is item.source
    items = item.create_definition_items(item_cache={})
    assert len(items) == 1
    assert items[0].name == 'a_mod'
    assert items[0].definitions == (item.source['a'],)

    # A file with a simple module that contains a single typedef
    item = get_item('module/t_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 't_mod.f90'
    assert item.definitions == (item.source['t_mod'],)

    items = item.create_definition_items(item_cache={})
    assert len(items) == 1
    assert items[0].name == 't_mod'
    assert items[0].ir is item.source['t_mod']
    # No typedefs because not selected in parser classes
    assert not items[0].ir.typedefs
    # Calling definitions automatically further completes the source
    assert items[0].definitions == (
        items[0].ir['t_proc'],
        items[0].ir['my_way'],
        items[0].ir.typedefs['t1'],
        items[0].ir.typedefs['t']
    )

    # Files don't have dependencies (direct dependencies, anyway)
    assert item.dependencies is ()

    # The same file but with typedefs parsed from the get-go
    item = get_item('module/t_mod.F90', RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass)
    assert item.name == 't_mod.f90'
    assert item.definitions == (item.source['t_mod'],)

    items = item.create_definition_items(item_cache={})
    assert len(items) == 1
    assert items[0].name == 't_mod'
    assert len(items[0].ir.typedefs) == 2
    assert items[0].definitions == (
        item.source['t_proc'],
        item.source['my_way'],
        item.source['t1'],
        item.source['t']
    )

    # Filter items when calling create_definition_items()
    assert not item.create_definition_items(only=ProcedureItem, item_cache={})
    items = item.create_definition_items(only=ModuleItem, item_cache={})
    assert len(items) == 1
    assert isinstance(items[0], ModuleItem)
    assert items[0].ir == item.source['t_mod']


def test_module_item(here):
    proj = here/'sources/projBatch'

    def get_item(path, name, parser_classes):
        filepath = proj/path
        source = Sourcefile.from_file(filepath, frontend=REGEX, parser_classes=parser_classes)
        return ModuleItem(name, source=source)

    # A file with simple module that contains a single subroutine and has no dependencies on
    # the module level
    item = get_item('module/a_mod.F90', 'a_mod', RegexParserClass.ProgramUnitClass)
    assert item.name == 'a_mod'
    assert item.ir is item.source['a_mod']
    assert item.definitions == (item.source['a'],)

    items = item.create_definition_items(item_cache={})
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir == item.source['a']

    assert not item.dependencies

    # A different file with a simple module that contains a single subroutine but has an import
    # dependency on the module level
    item = get_item('module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    assert item.name == 'b_mod'
    assert item.ir is item.source['b_mod']
    assert item.definitions == (item.source['b'],)

    items = item.create_definition_items(item_cache={})
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir == item.source['b']

    dependencies = item.dependencies
    assert len(dependencies) == 1
    assert isinstance(dependencies[0], Import)
    assert dependencies[0].module == 'header_mod'

    # Make sure the dependencies are also found correctly if done without parsing definitions first
    item = get_item('module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    dependencies = item.dependencies
    assert len(dependencies) == 1 and dependencies[0].module == 'header_mod'


def test_procedure_item(here):
    proj = here/'sources/projBatch'

    def get_item(path, name, parser_classes):
        filepath = proj/path
        source = Sourcefile.from_file(filepath, frontend=REGEX, parser_classes=parser_classes)
        return ProcedureItem(name, source=source)

    # A file with a single subroutine definition that calls a routine via interface block
    item = get_item('source/comp1.F90', '#comp1', RegexParserClass.ProgramUnitClass)
    assert item.name == '#comp1'
    assert item.ir is item.source['comp1']
    assert item.definitions is ()

    assert not item.create_definition_items(item_cache={})

    dependencies = item.dependencies
    assert len(dependencies) == 5
    assert isinstance(dependencies[0], Import)
    assert dependencies[0].module == 't_mod'
    assert isinstance(dependencies[1], Import)
    assert dependencies[1].module == 'header_mod'
    assert isinstance(dependencies[2], CallStatement)
    assert dependencies[2].name == 'arg%proc'
    assert isinstance(dependencies[3], CallStatement)
    assert dependencies[3].name == 'comp2'
    assert isinstance(dependencies[4], CallStatement)
    assert dependencies[4].name == 'arg%no%way'

    # A file with a single subroutine definition that calls two routines via module imports
    item = get_item('source/comp2.F90', '#comp2', RegexParserClass.ProgramUnitClass)
    assert item.name == '#comp2'
    assert item.ir is item.source['comp2']
    assert item.definitions is ()

    assert not item.create_definition_items(item_cache={})

    dependencies = item.dependencies
    assert len(dependencies) == 7
    assert isinstance(dependencies[0], Import)
    assert dependencies[0].module == 't_mod'
    assert isinstance(dependencies[1], Import)
    assert dependencies[1].module == 'header_mod'
    assert isinstance(dependencies[2], Import)
    assert dependencies[2].module == 'a_mod'
    assert isinstance(dependencies[3], Import)
    assert dependencies[3].module == 'b_mod'
    assert isinstance(dependencies[4], CallStatement)
    assert dependencies[4].name == 'a'
    assert isinstance(dependencies[5], CallStatement)
    assert dependencies[5].name == 'b'
    assert isinstance(dependencies[6], CallStatement)
    assert dependencies[6].name == 'arg%yay%proc'


def test_typedef_item(here):
    proj = here/'sources/projBatch'

    def get_item(path, name, parser_classes):
        filepath = proj/path
        source = Sourcefile.from_file(filepath, frontend=REGEX, parser_classes=parser_classes)
        return TypeDefItem(name, source=source)

    # A file with a single type definition
    item = get_item('module/t_mod.F90', 't_mod#t', RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass)
    assert item.name == 't_mod#t'
    assert item.ir is item.source['t']
    assert item.definitions is ()

    assert not item.create_definition_items(item_cache={})
    assert item.dependencies == as_tuple(item.ir.parent.imports)


def test_item_graph(here):
    """
    Build a :any:`nx.Digraph` from a dummy call hierarchy to check the incremental parsing and
    discovery behaves as expected.

    Expected dependencies:

    .. code-block::
             + -------------- + --(imports)-->  t_mod#t  --(imports)-->  tt_mod#tt
           /                  |
       comp1  --(calls)-->  comp2  --(calls)-->  a_mod#a
         |                    |
         |                    + --(calls)-->  b_mod#b
         |                    |
         |                    + --(calls)--> tt_mod#tt%proc --(binds to) --> tt_mod#tt_proc
         |
         + --(calls)-->  t_mod#t%proc  --(binds to)--> t_mod#t_proc
         |
         + --(calls)-->  t_mod#t%no%way --(binds to)-->  t_mod#t1%way  --(binds to)-->  t_mod#my_way

    Additionally, ``comp`` depends on ``header_mod`` (for a kind-parameter ``k``), while
    all others except ``t_mod``/``t_mod#t`` depend directly on the kind-parameter ``header_mod#k``.

    """
    proj = here/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 7

    # Map item names to items
    item_cache = {}

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        source = Sourcefile.from_file(path, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)
        file_item = FileItem(name=relative_path, source=source)
        item_cache[relative_path] = file_item
        item_cache.update((item.name, item) for item in file_item.create_definition_items(item_cache={}))

    # Populate a graph from a seed routine
    seed = '#comp1'
    full_graph = nx.DiGraph()
    full_graph.add_node(item_cache[seed])

    dependencies = item_cache[seed].create_dependency_items(item_cache=item_cache)
    full_graph.add_nodes_from(dependencies)
    full_graph.add_edges_from((item_cache[seed], item) for item in dependencies)

    queue = deque()
    queue.extend(dependencies)

    while queue:
        item = queue.popleft()
        dependencies = item.create_dependency_items(item_cache=item_cache)
        new_items = [i for i in dependencies if i not in full_graph]
        if new_items:
            full_graph.add_nodes_from(new_items)
            queue.extend(new_items)
        full_graph.add_edges_from((item, dependency) for dependency in dependencies)

    expected_dependencies = {
        '#comp1': ('header_mod', 't_mod#t', '#comp2', 't_mod#t%proc', 't_mod#t%no%way'),
        '#comp2': ('header_mod#k', 't_mod#t', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc'),
        'a_mod#a': ('header_mod#k',),
        'b_mod#b': ('header_mod#k',),
        't_mod#t': ('tt_mod#tt',),
        't_mod#t%proc': ('t_mod#t_proc',),
        't_mod#t_proc': ('tt_mod#tt',),
        't_mod#t%no%way': ('t_mod#t1%way',),
        't_mod#t%yay%proc': ('tt_mod#tt%proc',),
        't_mod#t1%way': ('t_mod#my_way',),
        't_mod#my_way': ('tt_mod#tt',),
        'tt_mod#tt': ('header_mod#k',),
        'tt_mod#tt%proc': ('tt_mod#tt_proc',),
        'tt_mod#tt_proc': ('header_mod#k',),
        'header_mod': (),
        'header_mod#k': (),
    }

    assert len(full_graph) == len(expected_dependencies)
    assert all(key in full_graph for key in expected_dependencies)

    edges = tuple((a.name, b.name) for a, b in full_graph.edges)
    for node, dependencies in expected_dependencies.items():
        for dependency in dependencies:
            assert (node, dependency) in edges
    assert len(edges) == sum(len(dependencies) for dependencies in expected_dependencies.values())

    # Note: quick visualization for debugging can be done using matplotlib
    #   import matplotlib.pyplot as plt
    #   nx.draw_planar(full_graph, with_labels=True)
