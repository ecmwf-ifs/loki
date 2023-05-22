# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import deque
from pathlib import Path
import re
import networkx as nx
import pytest

from loki import (
    HAVE_FP, HAVE_OFP, REGEX, RegexParserClass, as_tuple, CaseInsensitiveDict, gettempdir,
    FileItem, ModuleItem, ProcedureItem, TypeDefItem, ProcedureBindingItem, GlobalVariableItem,
    SGraph, SchedulerConfig,
    Sourcefile, Section, RawSource, Import, CallStatement, Scalar
)

pytestmark = pytest.mark.skipif(not HAVE_FP and not HAVE_OFP, reason='Fparser and OFP not available')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(name='default_config')
def fixture_default_config():
    """
    Default SchedulerConfig configuration with basic options.
    """
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'disable': ['abort']
        },
        'routines': []
    }


@pytest.fixture(name='comp1_expected_dependencies')
def fixture_comp1_expected_dependencies():
    return {
        '#comp1': ('header_mod', 't_mod#t', '#comp2', 't_mod#t%proc', 't_mod#t%no%way'),
        '#comp2': ('header_mod#k', 't_mod#t', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc'),
        'a_mod#a': ('header_mod#k',),
        'b_mod#b': (),
        't_mod#t': ('tt_mod#tt', 't_mod#t1'),
        't_mod#t1': (),
        't_mod#t%proc': ('t_mod#t_proc',),
        't_mod#t_proc': ('t_mod#t', 'a_mod#a', 't_mod#t%yay%proc'),
        't_mod#t%no%way': ('t_mod#t1%way',),
        't_mod#t%yay%proc': ('tt_mod#tt%proc',),
        't_mod#t1%way': ('t_mod#my_way',),
        't_mod#my_way': ('t_mod#t1', 't_mod#t1%way'),
        'tt_mod#tt': (),
        'tt_mod#tt%proc': ('tt_mod#proc',),
        'tt_mod#proc': ('tt_mod#tt',),
        'header_mod': (),
        'header_mod#k': (),
    }


@pytest.fixture(name='mod_proc_expected_dependencies')
def fixture_mod_proc_expected_dependencies():
    return {
        'other_mod#mod_proc': ('tt_mod#tt', 'tt_mod#tt%proc', 'b_mod#b'),
        'tt_mod#tt': (),
        'tt_mod#tt%proc': ('tt_mod#proc',),
        'tt_mod#proc': ('tt_mod#tt',),
        'b_mod#b': ()
    }


@pytest.fixture(name='expected_dependencies')
def fixture_expected_dependencies(comp1_expected_dependencies, mod_proc_expected_dependencies):
    return comp1_expected_dependencies | mod_proc_expected_dependencies


@pytest.fixture(name='no_expected_dependencies')
def fixture_no_expected_dependencies():
    return {}


class VisGraphWrapper:
    """
    Testing utility to parse the generated callgraph visualisation.
    """

    _re_nodes = re.compile(r'\s*\"?(?P<node>[\w%#./]+)\"? \[colo', re.IGNORECASE)
    _re_edges = re.compile(r'\s*\"?(?P<parent>[\w%#./]+)\"? -> \"?(?P<child>[\w%#./]+)\"?', re.IGNORECASE)

    def __init__(self, path):
        self.text = Path(path).read_text()

    @property
    def nodes(self):
        return list(self._re_nodes.findall(self.text))

    @property
    def edges(self):
        return list(self._re_edges.findall(self.text))


def get_item(cls, path, name, parser_classes, scheduler_config=None):
    source = Sourcefile.from_file(path, frontend=REGEX, parser_classes=parser_classes)
    if scheduler_config:
        config = scheduler_config.create_item_config(name)
    else:
        config = None
    return cls(name, source=source, config=config)


def test_file_item1(here, default_config):
    proj = here/'sources/projBatch'

    # A file with simple module that contains a single subroutine
    item = get_item(FileItem, proj/'module/a_mod.F90', 'module/a_mod.F90', RegexParserClass.EmptyClass)
    assert item.name == 'module/a_mod.F90'
    assert item.local_name == item.name
    assert item.scope_name is None
    assert not item.scope
    assert item.ir is item.source
    assert str(item) == 'loki.bulk.FileItem<module/a_mod.F90>'

    # A few checks on the item comparison
    assert item == 'module/a_mod.F90'
    assert item != FileItem('some_name', source=item.source)
    assert item == FileItem(item.name, source=item.source)

    # The file is not parsed at all
    assert not item.source.definitions
    assert isinstance(item.source.ir, Section)
    assert len(item.source.ir.body) == 1
    assert isinstance(item.source.ir.body[0], RawSource)

    # Querying definitions triggers a round of parsing
    assert item.definitions == (item.source['a_mod'],)
    assert len(item.source.definitions) == 1

    with pytest.raises(RuntimeError):
        # Without the FileItem in the item_cache, we can't create the modules
        item.create_definition_items(item_cache={}, config=SchedulerConfig.from_dict(default_config))

    # However, without strict parsing it will simply return an empty list
    assert not item.create_definition_items(item_cache={})

    items = item.create_definition_items(item_cache={item.name: item})
    assert len(items) == 1
    assert items[0] != None  # pylint: disable=singleton-comparison  # (intentionally trigger __eq__ here)
    assert items[0].name == 'a_mod'
    assert items[0].definitions == (item.source['a'],)

    # The default behavior would be to have the ProgramUnits parsed already
    item = get_item(FileItem, proj/'module/a_mod.F90', 'module/a_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 'module/a_mod.F90'
    assert item.definitions == (item.source['a_mod'],)
    assert item.ir is item.source
    items = item.create_definition_items(item_cache={item.name: item})
    assert len(items) == 1
    assert items[0].name == 'a_mod'
    assert items[0].definitions == (item.source['a'],)


def test_file_item2(here):
    proj = here/'sources/projBatch'

    # A file with a simple module that contains a single typedef
    item = get_item(FileItem, proj/'module/t_mod.F90', 'module/t_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 'module/t_mod.F90'
    assert item.definitions == (item.source['t_mod'],)

    items = item.create_definition_items(item_cache={item.name: item})
    assert len(items) == 1
    assert items[0].name == 't_mod'
    assert items[0].ir is item.source['t_mod']
    # No typedefs because not selected in parser classes
    assert not items[0].ir.typedefs
    # Calling definitions automatically further completes the source
    assert items[0].definitions == (
        items[0].ir['t_proc'],
        items[0].ir['my_way'],
        items[0].ir.typedef_map['t1'],
        items[0].ir.typedef_map['t']
    )

    # Files don't have dependencies (direct dependencies, anyway)
    assert item.dependencies is ()


def test_file_item3(here):
    proj = here/'sources/projBatch'

    # The same file but with typedefs parsed from the get-go
    item = get_item(
        FileItem, proj/'module/t_mod.F90', 'module/t_mod.F90',
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass
    )
    assert item.name == 'module/t_mod.F90'
    assert item.definitions == (item.source['t_mod'],)

    items = item.create_definition_items(item_cache={item.name: item})
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
    assert not item.create_definition_items(only=ProcedureItem, item_cache={item.name: item})
    items = item.create_definition_items(only=ModuleItem, item_cache={item.name: item})
    assert len(items) == 1
    assert isinstance(items[0], ModuleItem)
    assert items[0].ir == item.source['t_mod']


def test_module_item1(here):
    proj = here/'sources/projBatch'

    # A file with simple module that contains a single subroutine and has no dependencies on
    # the module level
    item = get_item(ModuleItem, proj/'module/a_mod.F90', 'a_mod', RegexParserClass.ProgramUnitClass)
    assert item.name == 'a_mod'
    assert item == 'a_mod'
    assert str(item) == 'loki.bulk.ModuleItem<a_mod>'
    assert item.ir is item.source['a_mod']
    assert item.definitions == (item.source['a'],)

    items = item.create_definition_items(item_cache={item.name: item})
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir == item.source['a']

    assert not item.dependencies


def test_module_item2(here):
    proj = here/'sources/projBatch'

    # A different file with a simple module that contains a single subroutine but has an import
    # dependency on the module level
    item = get_item(ModuleItem, proj/'module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    assert item.name == 'b_mod'
    assert item.ir is item.source['b_mod']
    assert item.definitions == (item.source['b'],)

    items = item.create_definition_items(item_cache={item.name: item})
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir == item.source['b']

    dependencies = item.dependencies
    assert len(dependencies) == 1
    assert isinstance(dependencies[0], Import)
    assert dependencies[0].module == 'header_mod'


def test_module_item3(here):
    proj = here/'sources/projBatch'

    # Make sure the dependencies are also found correctly if done without parsing definitions first
    item = get_item(ModuleItem, proj/'module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    dependencies = item.dependencies
    assert len(dependencies) == 1 and dependencies[0].module == 'header_mod'


def test_procedure_item1(here):
    proj = here/'sources/projBatch'

    # A file with a single subroutine definition that calls a routine via interface block
    item = get_item(ProcedureItem, proj/'source/comp1.F90', '#comp1', RegexParserClass.ProgramUnitClass)
    assert item.name == '#comp1'
    assert item == '#comp1'
    assert str(item) == 'loki.bulk.ProcedureItem<#comp1>'
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

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_cache = {item.name: item}
    item_cache = {
        (i := get_item(ModuleItem, proj/path, name, RegexParserClass.ProgramUnitClass)).name: i
        for path, name in [
            ('module/t_mod.F90', 't_mod'), ('source/comp2.F90', '#comp2'), ('headers/header_mod.F90', 'header_mod')
        ]
    }

    # To ensure any existing items from the item_cache are re-used, we instantiate one for
    # the procedure binding
    t_mod_t_proc = get_item(
        ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%proc',
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )
    item_cache[t_mod_t_proc.name] = t_mod_t_proc

    items = item.create_dependency_items(item_cache=item_cache)
    assert items == ('t_mod#t', 'header_mod', 't_mod#t%proc', '#comp2', 't_mod#t%no%way')
    assert item_cache[t_mod_t_proc.name] is t_mod_t_proc
    assert items[2] is t_mod_t_proc


def test_procedure_item2(here):
    proj = here/'sources/projBatch'

    # A file with a single subroutine definition that calls two routines via module imports
    item = get_item(ProcedureItem, proj/'source/comp2.F90', '#comp2', RegexParserClass.ProgramUnitClass)
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

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_cache = {item.name: item}
    item_cache = {
        (i := get_item(ModuleItem, proj/path, name, RegexParserClass.ProgramUnitClass)).name: i
        for path, name in [
            ('module/t_mod.F90', 't_mod'), ('module/a_mod.F90', 'a_mod'),
            ('module/b_mod.F90', 'b_mod'), ('headers/header_mod.F90', 'header_mod')
        ]
    }
    items = item.create_dependency_items(item_cache=item_cache)
    assert items == ('t_mod#t', 'header_mod#k', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc')

    # Does it still work if we call it again?
    assert items == item.create_dependency_items(item_cache=item_cache)


def test_procedure_item3(here):
    proj = here/'sources/projBatch'

    # A file with a single subroutine declared in a module that calls a typebound procedure
    # where the type is imported via an import statement in the module scope
    item = get_item(
        ProcedureItem, proj/'module/other_mod.F90', 'other_mod#mod_proc',
        RegexParserClass.ProgramUnitClass
    )
    dependencies = item.dependencies
    assert len(dependencies) == 3
    assert dependencies[0].module == 'tt_mod'
    assert dependencies[1].name == 'arg%proc'
    assert dependencies[2].name == 'b'

    item_cache = {
        item.name: item,
        'tt_mod': get_item(ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', RegexParserClass.ProgramUnitClass),
        'b_mod': get_item(ModuleItem, proj/'module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    }
    assert item.create_dependency_items(item_cache=item_cache) == ('tt_mod#tt', 'tt_mod#tt%proc', 'b_mod#b')


def test_procedure_item4(here):
    proj = here/'sources/projBatch'

    # A routine with a typebound procedure call where the typedef is in the same module
    item = get_item(
        ProcedureItem, proj/'module/t_mod.F90', 't_mod#my_way', RegexParserClass.ProgramUnitClass
    )
    dependencies = item.dependencies
    assert len(dependencies) == 2
    assert dependencies[0].name == 't1'
    assert dependencies[1].name == 'this%way'

    item_cache = {
        item.name: item,
        't_mod': ModuleItem('t_mod', source=item.source)
    }
    items = item.create_dependency_items(item_cache=item_cache)
    assert items == ('t_mod#t1', 't_mod#t1%way')


@pytest.mark.parametrize('config,expected_dependencies', [
    ({}, ('t_mod#t', 'header_mod#k', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc')),
    ({'default': {'disable': ['a']}}, ('t_mod#t', 'header_mod#k', 'b_mod#b', 't_mod#t%yay%proc')),
    ({'default': {'disable': ['a_mod#a']}}, ('t_mod#t', 'header_mod#k', 'b_mod#b', 't_mod#t%yay%proc')),
    ({'default': {'disable': ['t_mod#t%yay%proc']}}, ('t_mod#t', 'header_mod#k', 'a_mod#a', 'b_mod#b')),
    ({'default': {'disable': ['k']}}, ('t_mod#t', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc')),
    ({'default': {'disable': ['header_mod#k']}}, ('t_mod#t', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc')),
])
def test_procedure_item_with_config(here, config, expected_dependencies):
    proj = here/'sources/projBatch'

    # A file with a single subroutine definition that calls two routines via module imports
    item = get_item(ProcedureItem, proj/'source/comp2.f90', '#comp2', RegexParserClass.ProgramUnitClass)

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_cache = {item.name: item}
    item_cache |= {
        (i := get_item(ModuleItem, proj/path, name, RegexParserClass.ProgramUnitClass)).name: i
        for path, name in [
            ('module/t_mod.F90', 't_mod'), ('module/a_mod.F90', 'a_mod'),
            ('module/b_mod.F90', 'b_mod'), ('headers/header_mod.F90', 'header_mod')
        ]
    }
    scheduler_config = SchedulerConfig.from_dict(config)
    assert item.create_dependency_items(item_cache=item_cache, config=scheduler_config) == expected_dependencies


@pytest.mark.parametrize('disable', ['#comp2', 'comp2'])
def test_procedure_item_with_config2(here, disable):
    proj = here/'sources/projBatch'

    # Similar to the previous test but checking disabling of subroutines without scope
    item = get_item(ProcedureItem, proj/'source/comp1.f90', '#comp1', RegexParserClass.ProgramUnitClass)

    item_cache = {item.name: item}
    item_cache['t_mod'] = get_item(ModuleItem, proj/'module/t_mod.F90', 't_mod', RegexParserClass.ProgramUnitClass)
    item_cache['header_mod'] = get_item(
        ModuleItem, proj/'headers/header_mod.F90', 'header_mod', RegexParserClass.ProgramUnitClass
    )
    scheduler_config = SchedulerConfig.from_dict({'default': {'disable': [disable]}})
    assert item.create_dependency_items(item_cache=item_cache, config=scheduler_config) == (
        't_mod#t', 'header_mod', 't_mod#t%proc', 't_mod#t%no%way'
    )


def test_typedef_item(here):
    proj = here/'sources/projBatch'

    # A file with multiple type definitions, of which we pick one
    item = get_item(
        TypeDefItem, proj/'module/t_mod.F90', 't_mod#t',
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass
    )
    assert item.name == 't_mod#t'
    assert str(item) == 'loki.bulk.TypeDefItem<t_mod#t>'
    assert item.ir is item.source['t']
    assert item.definitions is ()

    assert not item.create_definition_items(item_cache={})
    assert item.dependencies == (item.scope.import_map['tt'], item.ir.parent['t1'])

    item_cache = CaseInsensitiveDict()
    item_cache[item.name] = item
    with pytest.raises(RuntimeError):
        item.create_dependency_items(item_cache=item_cache)

    # Need to add the modules of the dependent types
    item_cache['t_mod'] = ModuleItem('t_mod', source=item.source)
    item_cache['tt_mod'] = get_item(
        ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', RegexParserClass.ProgramUnitClass
    )
    assert 'tt_mod#tt' not in item_cache
    assert 't_mod#t1' not in item_cache
    items = item.create_dependency_items(item_cache=item_cache)
    assert 'tt_mod#tt' in item_cache
    assert 't_mod#t1' in item_cache
    assert items == (item_cache['tt_mod#tt'], item_cache['t_mod#t1'])
    assert all(isinstance(i, TypeDefItem) for i in items)
    assert not items[0].dependencies


def test_interface_item(here):
    pass


def test_global_variable_item(here):
    proj = here/'sources/projBatch'

    # A file with a global parameter definition
    item = get_item(
        GlobalVariableItem, proj/'headers/header_mod.F90', 'header_mod#k',
        RegexParserClass.ProgramUnitClass | RegexParserClass.DeclarationClass
    )
    assert item.name == 'header_mod#k'
    assert str(item) == 'loki.bulk.GlobalVariableItem<header_mod#k>'
    assert item.ir == item.source['header_mod'].declarations[0]
    assert item.definitions is ()
    assert not item.create_definition_items(item_cache={})
    assert item.dependencies is ()
    assert not item.create_dependency_items(item_cache={})


def test_procedure_binding_item1(here):
    proj = here/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    # A typedef with a procedure binding as well as nested types that have in turn procedure bindings

    # 1. A direct procedure binding
    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%proc', parser_classes)
    assert item.name == 't_mod#t%proc'
    assert str(item) == 'loki.bulk.ProcedureBindingItem<t_mod#t%proc>'
    assert item.ir is item.source['t'].variable_map['proc']
    assert item.definitions is ()
    assert not item.create_definition_items(item_cache={})
    assert item.dependencies == as_tuple(item.source['t_proc'])

    item_cache = {'t_mod': ModuleItem('t_mod', source=item.source)}
    items = item.create_dependency_items(item_cache=item_cache)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir is item.source['t_proc']


def test_procedure_binding_item2(here, default_config):
    proj = here/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    # 2. An indirect procedure binding via a nested type member, where the type is declared in the same module
    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%no%way', parser_classes)
    assert item.name == 't_mod#t%no%way'
    assert isinstance(item.ir, Scalar)
    assert item.definitions is ()
    assert not item.create_definition_items(item_cache={})
    assert item.dependencies == ('no%way',)

    item_cache = {item.name: item}
    with pytest.raises(RuntimeError):
        # Fails because item_cache does not contain the relevant module
        item.create_dependency_items(item_cache=item_cache, config=SchedulerConfig.from_dict(default_config))

    item_cache['t_mod'] = ModuleItem('t_mod', source=item.source)
    items = item.create_dependency_items(item_cache=item_cache)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureBindingItem)
    assert items[0].name == 't_mod#t1%way'
    assert 't_mod#t1%way' in item_cache

    assert 't_mod#my_way' not in item_cache
    next_items = items[0].create_dependency_items(item_cache=item_cache)
    assert len(next_items) == 1
    assert isinstance(next_items[0], ProcedureItem)
    assert next_items[0].ir is item.source['my_way']
    assert 't_mod#my_way' in item_cache


def test_procedure_binding_item3(here):
    proj = here/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    # 3. An indirect procedure binding via a nested type member, where the type is declared in a different module
    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%yay%proc', parser_classes)
    assert item.name == 't_mod#t%yay%proc'
    assert isinstance(item.ir, Scalar)
    assert item.definitions is ()
    assert not item.create_definition_items(item_cache={})
    assert item.dependencies == ('yay%proc',)

    item_cache = {item.name: item}
    item_cache['tt_mod'] = get_item(ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', parser_classes)
    items = item.create_dependency_items(item_cache=item_cache)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureBindingItem)
    assert items[0].name == 'tt_mod#tt%proc'
    assert 'tt_mod#tt%proc' in item_cache

    assert 'tt_mod#proc' not in item_cache
    next_items = items[0].create_dependency_items(item_cache=item_cache)
    assert len(next_items) == 1
    assert isinstance(next_items[0], ProcedureItem)
    assert next_items[0].ir is items[0].source['proc']
    assert 'tt_mod#proc' in item_cache


@pytest.mark.parametrize('config,expected_dependencies', [
    ({}, (('tt_mod#tt%proc',), ('tt_mod#proc',))),
    ({'default': {'disable': ['tt_mod#proc']}}, (('tt_mod#tt%proc',), ())),
    ({'default': {'disable': ['proc']}}, (('tt_mod#tt%proc',), ())),
    ({'default': {'disable': ['tt%proc']}}, ((),)),
    ({'default': {'disable': ['tt_mod#tt%proc']}}, ((),)),
])
def test_procedure_binding_with_config(here, config, expected_dependencies):
    proj = here/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%yay%proc', parser_classes)

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_cache = {item.name: item}
    item_cache['tt_mod'] = get_item(ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', RegexParserClass.ProgramUnitClass)
    scheduler_config = SchedulerConfig.from_dict(config)

    for dependencies in expected_dependencies:
        items = item.create_dependency_items(item_cache, config=scheduler_config)
        assert items == dependencies
        if items:
            item = items[0]


def test_item_graph(here, comp1_expected_dependencies):
    """
    Build a :any:`nx.Digraph` from a dummy call hierarchy to check the incremental parsing and
    discovery behaves as expected.
    """
    proj = here/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    # Map item names to items
    item_cache = CaseInsensitiveDict()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(FileItem, path, relative_path, RegexParserClass.ProgramUnitClass)
        item_cache[relative_path] = file_item
        item_cache.update((item.name, item) for item in file_item.create_definition_items(item_cache=item_cache))

    # Populate a graph from a seed routine
    seed = '#comp1'
    queue = deque()
    full_graph = nx.DiGraph()
    full_graph.add_node(item_cache[seed])
    queue.append(item_cache[seed])

    while queue:
        item = queue.popleft()
        dependencies = item.create_dependency_items(item_cache=item_cache)
        new_items = [i for i in dependencies if i not in full_graph]
        if new_items:
            full_graph.add_nodes_from(new_items)
            queue.extend(new_items)
        full_graph.add_edges_from((item, dependency) for dependency in dependencies)

    assert set(full_graph) == set(comp1_expected_dependencies)

    edges = tuple((a.name, b.name) for a, b in full_graph.edges)
    for node, dependencies in comp1_expected_dependencies.items():
        for dependency in dependencies:
            assert (node, dependency) in edges
    assert len(edges) == sum(len(dependencies) for dependencies in comp1_expected_dependencies.values())

    # Note: quick visualization for debugging can be done using matplotlib
    # import matplotlib.pyplot as plt
    # nx.draw_planar(full_graph, with_labels=True)
    # plt.show()
    # # -or-
    # plt.savefig('test_item_graph.png')


@pytest.mark.parametrize('seed,dependencies_fixture', [
    ('#comp1', 'comp1_expected_dependencies'),
    ('other_mod#mod_proc', 'mod_proc_expected_dependencies'),
    (['#comp1', 'other_mod#mod_proc'], 'expected_dependencies'),
    ('foobar', 'no_expected_dependencies')
])
def test_sgraph_from_seed(here, default_config, seed, dependencies_fixture, request):
    expected_dependencies = request.getfixturevalue(dependencies_fixture)
    proj = here/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    scheduler_config = SchedulerConfig.from_dict(default_config)

    # Map item names to items
    item_cache = CaseInsensitiveDict()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(
            FileItem, path, relative_path, RegexParserClass.ProgramUnitClass,
            scheduler_config
        )
        item_cache[relative_path] = file_item
        item_cache.update(
            (item.name, item)
            for item in file_item.create_definition_items(item_cache=item_cache, config=scheduler_config)
        )

    # Create the graph
    sgraph = SGraph(seed, item_cache, scheduler_config)

    # Check the graph
    assert set(sgraph.items) == set(expected_dependencies)
    assert set(sgraph.dependencies) == {
        (node, dependency)
        for node, dependencies in expected_dependencies.items()
        for dependency in dependencies
    }

    # Check the graph visualization
    graph_file = gettempdir()/'sgraph_from_seed.dot'
    sgraph.export_to_file(graph_file)
    assert graph_file.exists()
    assert graph_file.with_suffix('.dot.pdf').exists()

    vgraph = VisGraphWrapper(graph_file)
    assert set(vgraph.nodes) == {item.upper() for item in expected_dependencies}
    assert set(vgraph.edges) == {
        (node.upper(), dependency.upper())
        for node, dependencies in expected_dependencies.items()
        for dependency in dependencies
    }
    graph_file.unlink()
    graph_file.with_suffix('.dot.pdf').unlink()


@pytest.mark.parametrize('seed, disable,active_nodes', [
    ('#comp1', ('comp2', 'a'), (
        '#comp1', 't_mod#t', 'header_mod', 't_mod#t%proc', 't_mod#t%no%way',
        't_mod#t_proc', 't_mod#t%yay%proc', 'tt_mod#tt%proc', 'tt_mod#proc',
        't_mod#t1%way', 't_mod#my_way', 'tt_mod#tt', 't_mod#t1'
    )),
    ('#comp1', ('comp2', 'a', 't_mod#t%no%way'), (
        '#comp1', 't_mod#t', 'header_mod', 't_mod#t%proc',
        't_mod#t_proc', 't_mod#t%yay%proc', 'tt_mod#tt%proc', 'tt_mod#proc',
        'tt_mod#tt', 't_mod#t1'
    )),
    ('#comp1', ('#comp2', 't1%way'), (
        '#comp1', 't_mod#t', 'header_mod', 't_mod#t%proc', 't_mod#t%no%way',
        't_mod#t_proc', 't_mod#t%yay%proc', 'tt_mod#tt%proc', 'tt_mod#proc',
        'tt_mod#tt', 't_mod#t1', 'a_mod#a', 'header_mod#k'
    )),
    ('t_mod#t_proc', ('t_mod#t1', 'proc'), (
        't_mod#t_proc', 't_mod#t', 'tt_mod#tt', 'a_mod#a', 'header_mod#k',
        't_mod#t%yay%proc', 'tt_mod#tt%proc'
    ))
])
def test_sgraph_disable(here, default_config, expected_dependencies, seed, disable, active_nodes):
    proj = here/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    default_config['default']['disable'] = disable
    scheduler_config = SchedulerConfig.from_dict(default_config)

    # Map item names to items
    item_cache = CaseInsensitiveDict()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(
            FileItem, path, relative_path, RegexParserClass.ProgramUnitClass,
            scheduler_config
        )
        item_cache[relative_path] = file_item
        item_cache.update(
            (item.name, item)
            for item in file_item.create_definition_items(item_cache=item_cache, config=scheduler_config)
        )

    # Create the graph
    sgraph = SGraph(seed, item_cache, scheduler_config)

    # Check the graph
    assert set(sgraph.items) == set(active_nodes)
    assert set(sgraph.dependencies) == {
        (node, dependency)
        for node, dependencies in expected_dependencies.items()
        for dependency in dependencies
        if node in active_nodes and dependency in active_nodes
    }


@pytest.mark.parametrize('seed,routines,active_nodes', [
    (
        '#comp1', [
            {'name': '#comp1', 'expand': False}
        ], (
            '#comp1',
        )
    ),
    (
        '#comp2', [
            {'name': '#comp2', 'block': ['a', 'b']}
        ], (
            '#comp2', 't_mod#t', 'header_mod#k', 't_mod#t%yay%proc', 'tt_mod#tt',
            't_mod#t1', 'tt_mod#tt%proc', 'tt_mod#proc'
        )
    ),
    (
        '#comp2', [
            {'name': '#comp2', 'ignore': ['a'], 'block': ['b']}
        ], (
            '#comp2', 't_mod#t', 'header_mod#k', 't_mod#t%yay%proc', 'tt_mod#tt',
            't_mod#t1', 'tt_mod#tt%proc', 'tt_mod#proc'
        )
    ),
])
def test_sgraph_routines(here, default_config, expected_dependencies, seed, routines, active_nodes):
    proj = here/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    default_config['routine'] = routines
    scheduler_config = SchedulerConfig.from_dict(default_config)

    # Map item names to items
    item_cache = CaseInsensitiveDict()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(
            FileItem, path, relative_path, RegexParserClass.ProgramUnitClass,
            scheduler_config
        )
        item_cache[relative_path] = file_item
        item_cache.update(
            (item.name, item)
            for item in file_item.create_definition_items(item_cache=item_cache, config=scheduler_config)
        )

    # Create the graph
    sgraph = SGraph(seed, item_cache, scheduler_config)

    # Check the graph
    assert set(sgraph.items) == set(active_nodes)
    assert set(sgraph.dependencies) == {
        (node, dependency)
        for node, dependencies in expected_dependencies.items()
        for dependency in dependencies
        if node in active_nodes and dependency in active_nodes
    }
