# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

from collections import deque
from pathlib import Path
import re
import networkx as nx
import pytest

from loki import (
    Sourcefile, Subroutine, as_tuple, RawSource, TypeDef,
    Scalar, ProcedureSymbol
)
from loki.batch import (
    FileItem, ModuleItem, ProcedureItem, TypeDefItem,
    ProcedureBindingItem, ExternalItem, InterfaceItem, SGraph,
    SchedulerConfig, ItemFactory
)
from loki.frontend import HAVE_FP, HAVE_OFP, REGEX, RegexParserClass
from loki.ir import nodes as ir


pytestmark = pytest.mark.skipif(not HAVE_FP and not HAVE_OFP, reason='Fparser and OFP not available')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


@pytest.fixture(name='default_config', scope='function')
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
        '#comp1': ('header_mod', 't_mod', 't_mod#t', '#comp2', 't_mod#t%proc', 't_mod#t%no%way'),
        '#comp2': ('header_mod', 't_mod#t', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc'),
        'a_mod#a': ('header_mod',),
        'b_mod#b': (),
        't_mod': ('tt_mod#tt',),
        't_mod#t': ('tt_mod#tt', 't_mod#t1'),
        't_mod#t1': (),
        't_mod#t%proc': ('t_mod#t_proc',),
        't_mod#t_proc': ('t_mod#t', 'a_mod#a', 't_mod#t%yay%proc'),
        't_mod#t%no%way': ('t_mod#t1%way',),
        't_mod#t%yay%proc': ('tt_mod#tt%proc',),
        't_mod#t1%way': ('t_mod#my_way',),
        't_mod#my_way': ('t_mod#t1',),
        'tt_mod#tt': (),
        'tt_mod#tt%proc': ('tt_mod#proc',),
        'tt_mod#proc': ('tt_mod#tt',),
        'header_mod': (),
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
    dependencies = {}
    dependencies.update(comp1_expected_dependencies)
    dependencies.update(mod_proc_expected_dependencies)
    return dependencies


@pytest.fixture(name='no_expected_dependencies')
def fixture_no_expected_dependencies():
    return {}


@pytest.fixture(name='file_dependencies')
def fixture_file_dependencies():
    return {
        'source/comp1.F90': (
            'source/comp2.f90',
            'module/t_mod.F90',
            'headers/header_mod.F90'
        ),
        'source/comp2.f90': (
            'module/t_mod.F90',
            'headers/header_mod.F90',
            'module/a_mod.F90',
            'module/b_mod.F90'
        ),
        'module/t_mod.F90': (
            'module/tt_mod.F90',
            'module/a_mod.F90',
        ),
        'module/tt_mod.F90': (),
        'module/a_mod.F90': (
            'headers/header_mod.F90',
        ),
        'module/b_mod.F90': (),
        'headers/header_mod.F90': ()
    }


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


def test_file_item1(testdir, default_config):
    proj = testdir/'sources/projBatch'

    # A file with simple module that contains a single subroutine
    item = get_item(FileItem, proj/'module/a_mod.F90', 'module/a_mod.F90', RegexParserClass.EmptyClass)
    assert item.name == 'module/a_mod.F90'
    assert item.local_name == item.name
    assert item.scope_name is None
    assert not item.scope
    assert item.ir is item.source
    assert str(item) == 'loki.batch.FileItem<module/a_mod.F90>'

    # A few checks on the item comparison
    assert item == 'module/a_mod.F90'
    assert item != FileItem('some_name', source=item.source)
    assert item == FileItem(item.name, source=item.source)

    # The file is not parsed at all
    assert not item.source.definitions
    assert isinstance(item.source.ir, ir.Section)
    assert len(item.source.ir.body) == 1
    assert isinstance(item.source.ir.body[0], RawSource)

    # Querying definitions triggers a round of parsing
    assert item.definitions == (item.source['a_mod'],)
    assert len(item.source.definitions) == 1

    # Without the FileItem in the item_cache, the modules will be created as ExternalItem
    assert all(
        isinstance(_item, ExternalItem) and _item.origin_cls is ModuleItem
        for _item in item.create_definition_items(
            item_factory=ItemFactory(), config=SchedulerConfig.from_dict(default_config)
        )
    )

    # Check that external item raises an exception whenever we try to access any IR nodes
    external_item = item.create_definition_items(
        item_factory=ItemFactory(), config=SchedulerConfig.from_dict(default_config)
    )[0]

    for attr in ('ir', 'scope', 'path'):
        with pytest.raises(RuntimeError):
            getattr(external_item, attr)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_definition_items(item_factory=item_factory)
    assert len(items) == 1
    assert items[0] != None  # pylint: disable=singleton-comparison  # (intentionally trigger __eq__ here)
    assert items[0].name == 'a_mod'
    assert items[0].definitions == (item.source['a'],)

    # The default behavior would be to have the ProgramUnits parsed already
    item = get_item(FileItem, proj/'module/a_mod.F90', 'module/a_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 'module/a_mod.F90'
    assert item.definitions == (item.source['a_mod'],)
    assert item.ir is item.source
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_definition_items(item_factory=item_factory)
    assert len(items) == 1
    assert items[0].name == 'a_mod'
    assert items[0].definitions == (item.source['a'],)


def test_file_item2(testdir):
    proj = testdir/'sources/projBatch'

    # A file with a simple module that contains a single typedef
    item = get_item(FileItem, proj/'module/t_mod.F90', 'module/t_mod.F90', RegexParserClass.ProgramUnitClass)
    assert item.name == 'module/t_mod.F90'
    assert item.definitions == (item.source['t_mod'],)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_definition_items(item_factory=item_factory)
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
        items[0].ir.typedef_map['t'],
    )

    # Files don't have dependencies (direct dependencies, anyway)
    assert item.dependencies is ()


def test_file_item3(testdir):
    proj = testdir/'sources/projBatch'

    # The same file but with typedefs parsed from the get-go
    item = get_item(
        FileItem, proj/'module/t_mod.F90', 'module/t_mod.F90',
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass
    )
    assert item.name == 'module/t_mod.F90'
    assert item.definitions == (item.source['t_mod'],)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_definition_items(item_factory=item_factory)
    assert len(items) == 1
    assert items[0].name == 't_mod'
    assert len(items[0].ir.typedefs) == 2
    assert items[0].definitions == (
        item.source['t_proc'],
        item.source['my_way'],
        item.source['t1'],
        item.source['t'],
    )

    # Filter items when calling create_definition_items()
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_definition_items(item_factory=item_factory)
    assert not item.create_definition_items(only=ProcedureItem, item_factory=item_factory)
    items = item.create_definition_items(only=ModuleItem, item_factory=item_factory)
    assert len(items) == 1
    assert isinstance(items[0], ModuleItem)
    assert items[0].ir == item.source['t_mod']


def test_module_item1(testdir):
    proj = testdir/'sources/projBatch'

    # A file with simple module that contains a single subroutine and has no dependencies on
    # the module level
    item = get_item(ModuleItem, proj/'module/a_mod.F90', 'a_mod', RegexParserClass.ProgramUnitClass)
    assert item.name == 'a_mod'
    assert item == 'a_mod'
    assert str(item) == 'loki.batch.ModuleItem<a_mod>'
    assert item.ir is item.source['a_mod']
    assert item.definitions == (item.source['a'],)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_definition_items(item_factory=item_factory)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir == item.source['a']

    assert not item.dependencies


def test_module_item2(testdir):
    proj = testdir/'sources/projBatch'

    # A different file with a simple module that contains a single subroutine but has an import
    # dependency on the module level
    item = get_item(ModuleItem, proj/'module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    assert item.name == 'b_mod'
    assert item.ir is item.source['b_mod']
    assert item.definitions == (item.source['b'],)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_definition_items(item_factory=item_factory)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir == item.source['b']

    dependencies = item.dependencies
    assert len(dependencies) == 1
    assert isinstance(dependencies[0], ir.Import)
    assert dependencies[0].module == 'header_mod'


def test_module_item3(testdir):
    proj = testdir/'sources/projBatch'

    # Make sure the dependencies are also found correctly if done without parsing definitions first
    item = get_item(ModuleItem, proj/'module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    dependencies = item.dependencies
    assert len(dependencies) == 1 and dependencies[0].module == 'header_mod'


def test_module_item4(testdir):
    proj = testdir/'sources/projInlineCalls'

    # Make sure interfaces are correctly identified as definitions
    item = get_item(ModuleItem, proj/'some_module.F90', 'some_module', RegexParserClass.ProgramUnitClass)
    definitions = item.definitions
    assert len(definitions) == 8
    assert len(item.ir.interfaces) == 1
    assert item.ir.interfaces[0] in definitions

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item

    items = item.create_definition_items(item_factory=item_factory)
    assert len(items) == 10
    assert len(set(items)) == 6
    assert 'some_module#add_args' in item_factory.item_cache
    assert isinstance(item_factory.item_cache['some_module#add_args'], InterfaceItem)
    assert item_factory.item_cache['some_module#add_args'] in items


def test_procedure_item1(testdir):
    proj = testdir/'sources/projBatch'

    # A file with a single subroutine definition that calls a routine via interface block
    item = get_item(ProcedureItem, proj/'source/comp1.F90', '#comp1', RegexParserClass.ProgramUnitClass)
    assert item.name == '#comp1'
    assert item == '#comp1'
    assert str(item) == 'loki.batch.ProcedureItem<#comp1>'
    assert item.ir is item.source['comp1']
    assert item.definitions is ()

    assert not item.create_definition_items(item_factory=ItemFactory())

    dependencies = item.dependencies
    assert len(dependencies) == 5
    assert isinstance(dependencies[0], ir.Import)
    assert dependencies[0].module == 't_mod'
    assert isinstance(dependencies[1], ir.Import)
    assert dependencies[1].module == 'header_mod'
    assert isinstance(dependencies[2], ir.CallStatement)
    assert dependencies[2].name == 'arg%proc'
    assert isinstance(dependencies[3], ir.CallStatement)
    assert dependencies[3].name == 'comp2'
    assert isinstance(dependencies[4], ir.CallStatement)
    assert dependencies[4].name == 'arg%no%way'

    assert item.targets == ('t_mod', 't', 'nt1', 'header_mod', 'arg%proc', 'comp2', 'arg%no%way')

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache.update({
        (i := get_item(ModuleItem, proj/path, name, RegexParserClass.ProgramUnitClass)).name: i
        for path, name in [
            ('module/t_mod.F90', 't_mod'), ('source/comp2.f90', '#comp2'), ('headers/header_mod.F90', 'header_mod')
        ]
    })

    # To ensure any existing items from the item_cache are re-used, we instantiate one for
    # the procedure binding
    t_mod_t_proc = get_item(
        ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%proc',
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )
    item_factory.item_cache[t_mod_t_proc.name] = t_mod_t_proc

    items = item.create_dependency_items(item_factory=item_factory)
    assert items == ('t_mod', 't_mod#t', 'header_mod', 't_mod#t%proc', '#comp2', 't_mod#t%no%way')
    assert item_factory.item_cache[t_mod_t_proc.name] is t_mod_t_proc
    assert items[3] is t_mod_t_proc


def test_procedure_item2(testdir):
    proj = testdir/'sources/projBatch'

    # A file with a single subroutine definition that calls two routines via module imports
    item = get_item(ProcedureItem, proj/'source/comp2.f90', '#comp2', RegexParserClass.ProgramUnitClass)
    assert item.name == '#comp2'
    assert item.ir is item.source['comp2']
    assert item.definitions is ()

    item_factory = ItemFactory()
    assert not item.create_definition_items(item_factory=item_factory)

    dependencies = item.dependencies
    assert len(dependencies) == 7
    assert isinstance(dependencies[0], ir.Import)
    assert dependencies[0].module == 't_mod'
    assert isinstance(dependencies[1], ir.Import)
    assert dependencies[1].module == 'header_mod'
    assert isinstance(dependencies[2], ir.Import)
    assert dependencies[2].module == 'a_mod'
    assert isinstance(dependencies[3], ir.Import)
    assert dependencies[3].module == 'b_mod'
    assert isinstance(dependencies[4], ir.CallStatement)
    assert dependencies[4].name == 'a'
    assert isinstance(dependencies[5], ir.CallStatement)
    assert dependencies[5].name == 'b'
    assert isinstance(dependencies[6], ir.CallStatement)
    assert dependencies[6].name == 'arg%yay%proc'

    assert item.targets == (
        't_mod', 't', 'header_mod', 'k',
        'a_mod', 'a', 'b_mod', 'b', 'arg%yay%proc'
    )

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_factory.item_cache[item.name] = item
    item_factory.item_cache.update({
        (i := get_item(ModuleItem, proj/path, name, RegexParserClass.ProgramUnitClass)).name: i
        for path, name in [
            ('module/t_mod.F90', 't_mod'), ('module/a_mod.F90', 'a_mod'),
            ('module/b_mod.F90', 'b_mod'), ('headers/header_mod.F90', 'header_mod')
        ]
    })
    items = item.create_dependency_items(item_factory=item_factory)
    assert items == ('t_mod#t', 'header_mod', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc')

    # Does it still work if we call it again?
    assert items == item.create_dependency_items(item_factory=item_factory)


def test_procedure_item3(testdir):
    proj = testdir/'sources/projBatch'

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

    assert item.targets == ('tt_mod', 'tt', 'arg%proc', 'b')

    item_factory = ItemFactory()
    item_factory.item_cache.update({
        item.name: item,
        'tt_mod': get_item(ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', RegexParserClass.ProgramUnitClass),
        'b_mod': get_item(ModuleItem, proj/'module/b_mod.F90', 'b_mod', RegexParserClass.ProgramUnitClass)
    })
    assert item.create_dependency_items(item_factory=item_factory) == ('tt_mod#tt', 'tt_mod#tt%proc', 'b_mod#b')


def test_procedure_item4(testdir):
    proj = testdir/'sources/projBatch'

    # A routine with a typebound procedure call where the typedef is in the same module
    item = get_item(
        ProcedureItem, proj/'module/t_mod.F90', 't_mod#my_way', RegexParserClass.ProgramUnitClass
    )
    dependencies = item.dependencies
    assert len(dependencies) == 2
    assert dependencies[0].name == 't1'
    assert dependencies[1].name == 'this%way'

    assert item.targets == ('t1', 'this%way')

    item_factory = ItemFactory()
    item_factory.item_cache.update({
        item.name: item,
        't_mod': ModuleItem('t_mod', source=item.source)
    })
    items = item.create_dependency_items(item_factory=item_factory)
    assert items == ('t_mod#t1', 't_mod#t1%way')


@pytest.mark.parametrize('config,expected_dependencies,expected_targets', [
    (
        {},
        ('t_mod#t', 'header_mod', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'header_mod', 'k', 'a_mod', 'a', 'b_mod', 'b', 'arg%yay%proc')
    ),
    (
        {'default': {'disable': ['#a']}},
        ('t_mod#t', 'header_mod', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'header_mod', 'k', 'a_mod', 'a', 'b_mod', 'b', 'arg%yay%proc')
    ),
    (
        {'default': {'disable': ['a']}},
        ('t_mod#t', 'header_mod', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'header_mod', 'k', 'a_mod', 'b_mod', 'b', 'arg%yay%proc')
    ),
    (
        {'default': {'disable': ['a', 'a_mod']}},
        ('t_mod#t', 'header_mod', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'header_mod', 'k', 'b_mod', 'b', 'arg%yay%proc'),
    ),
    (
        {'default': {'disable': ['a_mod#a']}},
        ('t_mod#t', 'header_mod', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'header_mod', 'k', 'a_mod', 'b_mod', 'b', 'arg%yay%proc')
    ),
    (
        {'default': {'disable': ['a_mod']}},
        ('t_mod#t', 'header_mod', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'header_mod', 'k', 'b_mod', 'b', 'arg%yay%proc')
    ),
    (
        {'default': {'disable': ['t%yay%proc']}},
        ('t_mod#t', 'header_mod', 'a_mod#a', 'b_mod#b'),
        ('t_mod', 't', 'header_mod', 'k', 'a_mod', 'a', 'b_mod', 'b')
    ),
    (
        {'default': {'disable': ['t_mod#t%yay%proc']}},
        ('t_mod#t', 'header_mod', 'a_mod#a', 'b_mod#b'),
        ('t_mod', 't', 'header_mod', 'k', 'a_mod', 'a', 'b_mod', 'b')
    ),
    (
        {'default': {'disable': ['t_mod#t']}},
        ('header_mod', 'a_mod#a', 'b_mod#b'),
        ('t_mod', 'header_mod', 'k', 'a_mod', 'a', 'b_mod', 'b')
    ),
    (
        {'default': {'disable': ['t_mod']}},
        ('header_mod', 'a_mod#a', 'b_mod#b'),
        ('header_mod', 'k', 'a_mod', 'a', 'b_mod', 'b')
    ),
    (
        {'default': {'disable': ['header_mod']}},
        ('t_mod#t', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'a_mod', 'a', 'b_mod', 'b', 'arg%yay%proc')
    ),
    (
        {'default': {'disable': ['k']}},
        ('t_mod#t', 'a_mod#a', 'b_mod#b', 't_mod#t%yay%proc'),
        ('t_mod', 't', 'header_mod', 'a_mod', 'a', 'b_mod', 'b', 'arg%yay%proc')
    ),
])
def test_procedure_item_with_config(testdir, config, expected_dependencies, expected_targets):
    proj = testdir/'sources/projBatch'
    scheduler_config = SchedulerConfig.from_dict(config)

    # A file with a single subroutine definition that calls two routines via module imports
    item = get_item(
        ProcedureItem, proj/'source/comp2.f90', '#comp2',
        RegexParserClass.ProgramUnitClass, scheduler_config=scheduler_config
    )

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache.update({
        (i := get_item(
            ModuleItem, proj/path, name,
            RegexParserClass.ProgramUnitClass, scheduler_config=scheduler_config
        )).name: i
        for path, name in [
            ('module/t_mod.F90', 't_mod'), ('module/a_mod.F90', 'a_mod'),
            ('module/b_mod.F90', 'b_mod'), ('headers/header_mod.F90', 'header_mod')
        ]
    })
    assert item.create_dependency_items(item_factory=item_factory, config=scheduler_config) == expected_dependencies


    assert as_tuple(item.disable) == as_tuple(config.get('default', {}).get('disable', []))
    assert item.targets == as_tuple(expected_targets)


@pytest.mark.parametrize('disable', ['#comp2', 'comp2'])
def test_procedure_item_with_config2(testdir, disable):
    proj = testdir/'sources/projBatch'
    scheduler_config = SchedulerConfig.from_dict({'default': {'disable': [disable]}})

    # Similar to the previous test but checking disabling of subroutines without scope
    item = get_item(
        ProcedureItem, proj/'source/comp1.F90', '#comp1',
        RegexParserClass.ProgramUnitClass, scheduler_config=scheduler_config)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache['t_mod'] = get_item(
        ModuleItem, proj/'module/t_mod.F90', 't_mod', RegexParserClass.ProgramUnitClass
    )
    item_factory.item_cache['header_mod'] = get_item(
        ModuleItem, proj/'headers/header_mod.F90', 'header_mod',
        RegexParserClass.ProgramUnitClass, scheduler_config=scheduler_config
    )
    assert item.create_dependency_items(item_factory=item_factory, config=scheduler_config) == (
        't_mod', 't_mod#t', 'header_mod', 't_mod#t%proc', 't_mod#t%no%way'
    )

    assert item.targets == ('t_mod', 't', 'nt1', 'header_mod', 'arg%proc', 'arg%no%way')


@pytest.mark.parametrize('enable_imports', [False, True])
def test_procedure_item_external_item(tmp_path, enable_imports, default_config):
    """
    Test that dependencies to external module procedures are marked as external item
    """
    fcode = """
subroutine procedure_item_external_item
    use external_mod, only: external_proc, unused_external_proc, external_type, external_var
    implicit none
    type(external_type) :: my_type

    call external_proc(1)

    my_type%my_val = external_var
end subroutine procedure_item_external_item
    """
    filepath = tmp_path/'procedure_item_external_item.F90'
    filepath.write_text(fcode)

    default_config['default']['enable_imports'] = enable_imports
    scheduler_config = SchedulerConfig.from_dict(default_config)
    item = get_item(
        ProcedureItem, filepath, '#procedure_item_external_item',
        RegexParserClass.ProgramUnitClass, scheduler_config
    )
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    items = item.create_dependency_items(item_factory=item_factory, config=scheduler_config)

    # NB: dependencies to imported symbols are not added as external items because it would be impossible
    #     to determine their type. Instead, the external module is marked as a dependency, regardless if
    #     imports are enabled or not.
    #     However, the external procedure with a call statement is recognized as an external procedure
    #     and therefore included in the dependency tree.
    assert items == ('external_mod', 'external_mod#external_proc')
    assert all(isinstance(it, ExternalItem) for it in items)
    assert [it.origin_cls for it in items] == [ModuleItem, ProcedureItem]


def test_typedef_item(testdir):
    proj = testdir/'sources/projBatch'

    # A file with multiple type definitions, of which we pick one
    item = get_item(
        TypeDefItem, proj/'module/t_mod.F90', 't_mod#t',
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass
    )
    assert item.name == 't_mod#t'
    assert str(item) == 'loki.batch.TypeDefItem<t_mod#t>'
    assert item.ir is item.source['t']
    assert 'proc' in item.ir.variable_map
    assert item.definitions == item.ir.declarations

    # Without module items in the cache, the definition items will be externals
    assert all(
        isinstance(_item, ExternalItem) and _item.origin_cls is ProcedureBindingItem
        for _item in item.create_definition_items(item_factory=ItemFactory())
    )
    assert item.dependencies == (item.scope.import_map['tt'], item.ir.parent['t1'])

    # Without module items in the cache, the dependency items will be externals
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    assert all(
        isinstance(_item, ExternalItem) and _item.origin_cls in (ModuleItem, TypeDefItem)
        for _item in item.create_dependency_items(item_factory=ItemFactory())
    )

    # Need to add the modules of the dependent types
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache['t_mod'] = ModuleItem('t_mod', source=item.source)
    item_factory.item_cache['tt_mod'] = get_item(
        ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', RegexParserClass.ProgramUnitClass
    )
    assert 'tt_mod#tt' not in item_factory.item_cache
    assert 't_mod#t1' not in item_factory.item_cache
    items = item.create_dependency_items(item_factory=item_factory)
    assert 'tt_mod#tt' in item_factory.item_cache
    assert 't_mod#t1' in item_factory.item_cache
    assert items == (item_factory.item_cache['tt_mod#tt'], item_factory.item_cache['t_mod#t1'])
    assert all(isinstance(i, TypeDefItem) for i in items[1:])
    assert not items[1].dependencies


def test_interface_item_in_module(testdir):
    proj = testdir/'sources/projInlineCalls'

    # A file containing a module, with an interface to declare multiple functions
    # with a common name
    item = get_item(
        InterfaceItem, proj/'some_module.F90', 'some_module#add_args',
        RegexParserClass.ProgramUnitClass | RegexParserClass.InterfaceClass
    )

    assert item.name == 'some_module#add_args'
    assert str(item) == 'loki.batch.InterfaceItem<some_module#add_args>'
    assert item.ir is item.source['some_module'].interface_map['add_args']
    assert {'add_args', 'add_two_args', 'add_three_args'} == set(item.ir.symbols)

    # An interface does not define anything by itself
    assert not item.definitions
    assert not item.create_definition_items(item_factory=ItemFactory())

    # An interface depends on the routines it declares
    assert item.dependencies == ('add_two_args', 'add_three_args')

    # Without module item in the cache, the dependencies will be externals
    scheduler_config = SchedulerConfig.from_dict({'default': {'strict': True}})
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    assert all(
        isinstance(_item, ExternalItem) and _item.origin_cls is ProcedureItem
        for _item in item.create_dependency_items(item_factory=item_factory, config=scheduler_config)
    )

    # Let's start again with the module item
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache['some_module'] = ModuleItem('some_module', source=item.source)
    assert 'some_module#add_two_args' not in item_factory.item_cache
    assert 'some_module#add_three_args' not in item_factory.item_cache
    items = item.create_dependency_items(item_factory=item_factory)
    assert 'some_module#add_two_args' in item_factory.item_cache
    assert 'some_module#add_three_args' in item_factory.item_cache
    assert items == (
        item_factory.item_cache['some_module#add_two_args'], item_factory.item_cache['some_module#add_three_args']
    )
    assert all(isinstance(i, ProcedureItem) for i in items)


def test_interface_item_in_subroutine(testdir):
    proj = testdir/'sources/projInlineCalls'

    # A file containing the driver subroutine, which uses an interface to declare an
    # inline call
    item = get_item(
        ProcedureItem, proj/'driver.F90', '#driver',
        RegexParserClass.ProgramUnitClass
    )

    # Make sure the interface is included in the dependencies
    assert len(item.dependencies) == len(item.ir.imports + item.ir.interfaces) + 1 # (+1 for the call)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item

    # Dependency items cannot be created without the corresponding modules present
    with pytest.raises(RuntimeError):
        item.create_dependency_items(item_factory=item_factory)

    # Add the missing dependency modules
    for module_name in ('some_module', 'vars_module'):
        module_item = get_item(ModuleItem, proj/f'{module_name}.F90', module_name, RegexParserClass.ProgramUnitClass)
        item_factory.item_cache[module_item.name] = module_item

    # Dependency items can still not be created because the interface routine is still missing
    with pytest.raises(RuntimeError):
        item.create_dependency_items(item_factory=item_factory)

    # Add the missing dependency
    routine_item = get_item(ProcedureItem, proj/'double_real.F90', '#double_real', RegexParserClass.ProgramUnitClass)
    item_factory.item_cache[routine_item.name] = routine_item

    # Validate dependency items
    items = item.create_dependency_items(item_factory=item_factory)
    assert set(items) == {
        'some_module#return_one', 'some_module', 'some_module#add_args', 'some_module#some_type',
        'vars_module', '#double_real', 'some_module#some_type%do_something'
    }


def test_procedure_binding_item1(testdir):
    proj = testdir/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    # A typedef with a procedure binding as well as nested types that have in turn procedure bindings

    # 1. A direct procedure binding
    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%proc', parser_classes)
    assert item.name == 't_mod#t%proc'
    assert str(item) == 'loki.batch.ProcedureBindingItem<t_mod#t%proc>'
    assert item.ir is item.source['t'].variable_map['proc']
    assert item.definitions is ()
    assert not item.create_definition_items(item_factory=ItemFactory())
    assert item.dependencies == as_tuple(item.source['t_proc'])

    item_factory = ItemFactory()
    item_factory.item_cache.update({'t_mod': ModuleItem('t_mod', source=item.source)})
    items = item.create_dependency_items(item_factory=item_factory)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureItem)
    assert items[0].ir is item.source['t_proc']


def test_procedure_binding_item2(testdir, default_config):
    proj = testdir/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    # 2. An indirect procedure binding via a nested type member, where the type is declared in the same module
    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%no%way', parser_classes)
    assert item.name == 't_mod#t%no%way'
    assert isinstance(item.ir, Scalar)
    assert item.definitions is ()
    assert not item.create_definition_items(item_factory=ItemFactory())
    assert item.dependencies == ('no%way',)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    # ExternalItem, because item_cache does not contain the relevant module
    assert all(
        isinstance(_item, ExternalItem) and _item.origin_cls is ProcedureBindingItem
        for _item in item.create_dependency_items(
            item_factory=item_factory, config=SchedulerConfig.from_dict(default_config)
        )
    )

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache['t_mod'] = ModuleItem('t_mod', source=item.source)
    items = item.create_dependency_items(item_factory=item_factory)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureBindingItem)
    assert items[0].name == 't_mod#t1%way'
    assert 't_mod#t1%way' in item_factory.item_cache

    assert 't_mod#my_way' not in item_factory.item_cache
    next_items = items[0].create_dependency_items(item_factory=item_factory)
    assert len(next_items) == 1
    assert isinstance(next_items[0], ProcedureItem)
    assert next_items[0].ir is item.source['my_way']
    assert 't_mod#my_way' in item_factory.item_cache


def test_procedure_binding_item3(testdir):
    proj = testdir/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    # 3. An indirect procedure binding via a nested type member, where the type is declared in a different module
    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%yay%proc', parser_classes)
    assert item.name == 't_mod#t%yay%proc'
    assert isinstance(item.ir, Scalar)
    assert item.definitions is ()
    assert not item.create_definition_items(item_factory=ItemFactory())
    assert item.dependencies == ('yay%proc',)

    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache['tt_mod'] = get_item(ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', parser_classes)
    items = item.create_dependency_items(item_factory=item_factory)
    assert len(items) == 1
    assert isinstance(items[0], ProcedureBindingItem)
    assert items[0].name == 'tt_mod#tt%proc'
    assert 'tt_mod#tt%proc' in item_factory.item_cache

    assert 'tt_mod#proc' not in item_factory.item_cache
    next_items = items[0].create_dependency_items(item_factory=item_factory)
    assert len(next_items) == 1
    assert isinstance(next_items[0], ProcedureItem)
    assert next_items[0].ir is items[0].source['proc']
    assert 'tt_mod#proc' in item_factory.item_cache


@pytest.mark.parametrize('config,expected_dependencies', [
    ({}, (('tt_mod#tt%proc',), ('tt_mod#proc',))),
    ({'default': {'disable': ['tt_mod#proc']}}, (('tt_mod#tt%proc',), ())),
    ({'default': {'disable': ['proc']}}, (('tt_mod#tt%proc',), ())),
    ({'default': {'disable': ['tt%proc']}}, ((),)),
    ({'default': {'disable': ['tt_mod#tt%proc']}}, ((),)),
])
def test_procedure_binding_with_config(testdir, config, expected_dependencies):
    proj = testdir/'sources/projBatch'
    parser_classes = (
        RegexParserClass.ProgramUnitClass | RegexParserClass.TypeDefClass | RegexParserClass.DeclarationClass
    )

    item = get_item(ProcedureBindingItem, proj/'module/t_mod.F90', 't_mod#t%yay%proc', parser_classes)

    # We need to have suitable dependency modules in the cache to spawn the dependency items
    item_factory = ItemFactory()
    item_factory.item_cache[item.name] = item
    item_factory.item_cache['tt_mod'] = get_item(
        ModuleItem, proj/'module/tt_mod.F90', 'tt_mod', RegexParserClass.ProgramUnitClass
    )
    scheduler_config = SchedulerConfig.from_dict(config)

    for dependencies in expected_dependencies:
        items = item.create_dependency_items(item_factory, config=scheduler_config)
        assert items == dependencies
        if items:
            item = items[0]


def test_item_graph(testdir, comp1_expected_dependencies):
    """
    Build a :any:`nx.Digraph` from a dummy call hierarchy to check the incremental parsing and
    discovery behaves as expected.
    """
    proj = testdir/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    # Map item names to items
    item_factory = ItemFactory()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(FileItem, path, relative_path, RegexParserClass.ProgramUnitClass)
        item_factory.item_cache[relative_path] = file_item
        item_factory.item_cache.update(
            (item.name, item) for item in file_item.create_definition_items(item_factory=item_factory)
        )

    # Populate a graph from a seed routine
    seed = '#comp1'
    queue = deque()
    full_graph = nx.DiGraph()
    full_graph.add_node(item_factory.item_cache[seed])
    queue.append(item_factory.item_cache[seed])

    while queue:
        item = queue.popleft()
        dependencies = item.create_dependency_items(item_factory=item_factory)
        new_items = [i for i in dependencies if i not in full_graph]
        if new_items:
            full_graph.add_nodes_from(new_items)
            queue.extend(new_items)
        full_graph.add_edges_from((item, dependency) for dependency in dependencies)

    # Need to add the cyclic dependency (which isn't included in the fixture)
    comp1_expected_dependencies['t_mod#my_way'] += ('t_mod#t1%way',)

    assert set(full_graph) == set(comp1_expected_dependencies)
    assert {(a.name, b.name) for a, b in full_graph.edges} == {
        (a, b) for a, deps in comp1_expected_dependencies.items() for b in deps
    }

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
    ('#foobar', 'no_expected_dependencies'),
    # Not fully-qualified procedure name for a free subroutine
    ('comp1', 'comp1_expected_dependencies'),
     # Not fully-qualified procedure name for a module procedure
    ('mod_proc', 'mod_proc_expected_dependencies'),
])
def test_sgraph_from_seed(tmp_path, testdir, default_config, seed, dependencies_fixture, request):
    expected_dependencies = request.getfixturevalue(dependencies_fixture)
    proj = testdir/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    scheduler_config = SchedulerConfig.from_dict(default_config)
    item_factory = ItemFactory()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(
            FileItem, path, relative_path, RegexParserClass.ProgramUnitClass,
            scheduler_config
        )
        item_factory.item_cache[relative_path] = file_item
        item_factory.item_cache.update(
            (item.name, item)
            for item in file_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
        )

    # Create the graph
    sgraph = SGraph.from_seed(seed, item_factory, scheduler_config)

    # Check the graph
    assert set(sgraph.items) == set(expected_dependencies)
    assert set(sgraph.dependencies) == {
        (node, dependency)
        for node, dependencies in expected_dependencies.items()
        for dependency in dependencies
    }

    # Check the graph visualization
    graph_file = tmp_path/'sgraph_from_seed.dot'
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


@pytest.mark.parametrize('seed,disable,active_nodes', [
    ('#comp1', ('comp2', 'a'), (
        '#comp1', 't_mod', 't_mod#t', 'header_mod', 't_mod#t%proc', 't_mod#t%no%way',
        't_mod#t_proc', 't_mod#t%yay%proc', 'tt_mod#tt%proc', 'tt_mod#proc',
        't_mod#t1%way', 't_mod#my_way', 'tt_mod#tt', 't_mod#t1'
    )),
    ('#comp1', ('comp2', 'a', 't_mod#t%no%way'), (
        '#comp1', 't_mod', 't_mod#t', 'header_mod', 't_mod#t%proc',
        't_mod#t_proc', 't_mod#t%yay%proc', 'tt_mod#tt%proc', 'tt_mod#proc',
        'tt_mod#tt', 't_mod#t1'
    )),
    ('#comp1', ('#comp2', 't1%way'), (
        '#comp1', 't_mod', 't_mod#t', 'header_mod', 't_mod#t%proc', 't_mod#t%no%way',
        't_mod#t_proc', 't_mod#t%yay%proc', 'tt_mod#tt%proc', 'tt_mod#proc',
        'tt_mod#tt', 't_mod#t1', 'a_mod#a'
    )),
    ('t_mod#t_proc', ('t_mod#t1', 'proc'), (
        't_mod#t_proc', 't_mod#t', 'tt_mod#tt', 'a_mod#a', 'header_mod',
        't_mod#t%yay%proc', 'tt_mod#tt%proc'
    ))
])
def test_sgraph_disable(testdir, default_config, expected_dependencies, seed, disable, active_nodes):
    proj = testdir/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    default_config['default']['disable'] = disable
    scheduler_config = SchedulerConfig.from_dict(default_config)
    item_factory = ItemFactory()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(
            FileItem, path, relative_path, RegexParserClass.ProgramUnitClass,
            scheduler_config
        )
        item_factory.item_cache[relative_path] = file_item
        item_factory.item_cache.update(
            (item.name, item)
            for item in file_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
        )

    # Create the graph
    sgraph = SGraph.from_seed(seed, item_factory, scheduler_config)

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
        '#comp1', {
            'comp1': {'expand': False}
        }, (
            '#comp1',
        )
    ),
    (
        '#comp2', {
            'comp2': {'block': ['a', 'b']},
            't_mod': {'block': ['a']}
        }, (
            '#comp2', 't_mod#t', 'header_mod', 't_mod#t%yay%proc',
            'tt_mod#tt', 't_mod#t1', 'tt_mod#tt%proc', 'tt_mod#proc'
        )
    ),
    (
        '#comp2', {
            'comp2': {'ignore': ['a'], 'block': ['b']},
            't_mod': {'ignore': ['a']}
        }, (
            '#comp2', 't_mod#t', 'header_mod', 't_mod#t%yay%proc',
            'tt_mod#tt', 't_mod#t1', 'tt_mod#tt%proc', 'tt_mod#proc',
            'a_mod#a'
        )
    ),
])
def test_sgraph_routines(testdir, default_config, expected_dependencies, seed, routines, active_nodes):
    proj = testdir/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    default_config['routines'] = routines
    scheduler_config = SchedulerConfig.from_dict(default_config)
    item_factory = ItemFactory()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        relative_path = str(path.relative_to(proj))
        file_item = get_item(
            FileItem, path, relative_path, RegexParserClass.ProgramUnitClass,
            scheduler_config
        )
        item_factory.item_cache[relative_path] = file_item
        item_factory.item_cache.update(
            (item.name, item)
            for item in file_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
        )

    # Create the graph
    sgraph = SGraph.from_seed(seed, item_factory, scheduler_config)

    # Check the graph
    assert set(sgraph.items) == set(active_nodes)
    assert set(sgraph.dependencies) == {
        (node, dependency)
        for node, dependencies in expected_dependencies.items()
        for dependency in dependencies
        if node in active_nodes and dependency in active_nodes
    }

    targets = expected_dependencies[seed]
    targets = [t.replace('t_mod#t%', 'arg%') for t in targets]
    targets = [t.rsplit('#', maxsplit=1)[-1] for t in targets]

    # Without full parse and enriching (as done in the Scheduler before processing),
    # the type of the imported symbol cannot be determined and therefore global
    # variables like `nt1` or parameters like 'k' are listed as targets
    if seed == '#comp1':
        targets += ['nt1']
    if seed == '#comp2':
        targets += ['t_mod', 'b_mod', 'a_mod', 'k']

    if 'block' in routines[seed[1:]]:
        targets = [t for t in targets if t not in routines[seed[1:]]['block']]
    assert set(item_factory.item_cache[seed].targets) == set(targets)

    item_factory.item_cache['t_mod'].source.make_complete()
    item_factory.item_cache['header_mod'].source.make_complete()
    item_factory.item_cache[seed].source.make_complete()
    item_factory.item_cache[seed].ir.enrich([
        item_factory.item_cache['t_mod'].ir, item_factory.item_cache['header_mod'].ir]
    )

    # With fully-parsed and enriched source, we are able to distinguish between
    # the types of imported symbols and consequently global variables and parameters
    # are no longer listed as targets
    if 'nt1' in targets:
        targets.remove('nt1')
    if 'k' in targets:
        targets.remove('k')
    assert set(item_factory.item_cache[seed].targets) == set(targets)


def test_sgraph_filegraph(testdir, default_config, file_dependencies):
    proj = testdir/'sources/projBatch'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 8

    scheduler_config = SchedulerConfig.from_dict(default_config)
    item_factory = ItemFactory()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        file_item = FileItem(
            name=str(path),
            source=Sourcefile.from_file(path, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass),
            config=scheduler_config.create_item_config(str(path))
        )
        item_factory.item_cache[file_item.name] = file_item
        item_factory.item_cache.update(
            (item.name, item)
            for item in file_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
        )

    # Create the graph
    sgraph = SGraph.from_seed('#comp1', item_factory, scheduler_config)

    # Derive the file graph
    file_graph = SGraph.as_filegraph(sgraph, item_factory, scheduler_config)

    assert set(file_graph.items) == {str(proj/name) for name in file_dependencies}
    assert set(file_graph.dependencies) == {
        (str(proj/node), str(proj/dependency))
        for node, dependencies in file_dependencies.items()
        for dependency in dependencies
    }

def discover_proj_typebound_item_factory(testdir, scheduler_config):
    proj = testdir/'sources/projTypeBound'
    suffixes = ['.f90', '.F90']

    path_list = [f for ext in suffixes for f in proj.glob(f'**/*{ext}')]
    assert len(path_list) == 3

    item_factory = ItemFactory()

    # Instantiate the basic list of items (files, modules, subroutines)
    for path in path_list:
        file_item = FileItem(
            name=str(path),
            source=Sourcefile.from_file(path, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass),
        )
        item_factory.item_cache[file_item.name] = file_item

        definitions = {
            item.name: item
            for item in file_item.create_definition_items(item_factory=item_factory, config=scheduler_config)
        }
        item_factory.item_cache.update(definitions)

        module_names = [item.name for item in definitions.values() if isinstance(item, ModuleItem)]
        definitions = {
            item.name: item
            for module in module_names
            for item in item_factory.item_cache[module].create_definition_items(
                item_factory=item_factory, config=scheduler_config
            )
        }
        item_factory.item_cache.update(definitions)

        type_names = [item.name for item in definitions.values() if isinstance(item, TypeDefItem)]
        definitions = {
            item.name: item
            for type_ in type_names
            for item in item_factory.item_cache[type_].create_definition_items(
                item_factory=item_factory, config=scheduler_config
            )
        }
        item_factory.item_cache.update(definitions)

    return item_factory


@pytest.mark.parametrize('name,config_override,item_type,ir_type,attrs_to_check,dependency_items', [
    (
        ##########
        '#driver',
        ##########
        # This depends on  modules (via unqualified imports), typebound procedures,
        # and typebound procedures in nested derived types
        {},
        ProcedureItem,
        Subroutine,
        {
            'calls': (
                'some_type%other_routine',
                'some_type%some_routine',
                'header_type%member_routine',
                'header_type%routine',
                'other%member',
                'other%var%member_routine'
            ),
            'targets': (
                'typebound_item',
                'typebound_header',
                'typebound_other',
                'other',
                'obj%other_routine',
                'obj2%some_routine',
                'header%member_routine',
                'header%routine',
                'other_obj%member',
                'derived%var%member_routine'
            )
        },
        (
            'typebound_item',
            'typebound_header',
            'typebound_other#other_type',
            'typebound_item#some_type%other_routine',
            'typebound_item#some_type%some_routine',
            'typebound_header#header_type%member_routine',
            'typebound_header#header_type%routine',
            'typebound_other#other_type%member',
            'typebound_other#other_type%var%member_routine'
        )
    ),
    (
        ###############################
        'typebound_item#other_routine',
        ###############################
        # This is a module routine that depends on a subroutine from an unqualified import,
        # and typebound procedures in the same module
        {},
        ProcedureItem,
        Subroutine,
        {
            'calls': ('abor1', 'some_type%routine1', 'some_type%routine2'),
            'targets': ('some_type', 'abor1', 'self%routine1', 'self%routine2'),
        },
        (
            'typebound_item#some_type',
            'typebound_header#abor1',
            'typebound_item#some_type%routine1',
            'typebound_item#some_type%routine2'
        ),
    ),
    (
        #########################
        'typebound_item#routine',
        #########################
        # This is a module routine that depends on a type bound procedure,
        # which is listed as disabled in the scheduler config with fully qualified name
        {
            'disable': ['typebound_item#some_type%some_routine'],
        },
        ProcedureItem,
        Subroutine,
        {
            'calls': ('some_type%some_routine',),
            'targets': ('some_type',)
        },
        ('typebound_item#some_type',),
    ),
    (
        #########################
        'typebound_item#routine',
        #########################
        # This is a module routine that depends on a type bound procedure,
        # which is listed as disabled in the scheduler config without providing scope
        {
            'disable': ['some_type%some_routine'],
        },
        ProcedureItem,
        Subroutine,
        {
            'calls': ('some_type%some_routine',),
            'targets': ('some_type',)
        },
        ('typebound_item#some_type',),
    ),
    (
        #########################
        'typebound_item#routine1',
        #########################
        # This is a module routine that depends on a module procedure,
        # which is listed as disabled in the scheduler config without providing scope
        {
            'disable': ['module_routine'],
        },
        ProcedureItem,
        Subroutine,
        {
            'calls': ('module_routine',),
            'targets': ('some_type',)
        },
        ('typebound_item#some_type',),
    ),
    (
        #######################################
        'typebound_item#some_type%some_routine',
        #######################################
        # This is a procedure binding, where the bound procedure is listed as disabled
        # without providing fully qualified name - This means that the dependency item
        # is not created
        {
            'disable': ['some_routine'],
        },
        ProcedureBindingItem,
        ProcedureSymbol,
        {
            'calls': (),
            'targets': (),
        },
        (),
    ),
    (
        #######################################
        'typebound_item#some_type%some_routine',
        #######################################
        # This is a procedure binding, where the bound procedure is listed as disabled
        # with fully qualified name provided - this means that the dependency item
        # is not created
        {
            'disable': ['typebound_item#some_routine'],
        },
        ProcedureBindingItem,
        ProcedureSymbol,
        {
            'calls': (),
            'targets': (),
        },
        (),
    ),
    (
        #######################################
        'typebound_item#some_type%some_routine',
        #######################################
        # This is a procedure binding, where the bound procedure is listed as ignored,
        # which still includes it in the targets list
        {
            'ignore': ['some_routine'],
        },
        ProcedureBindingItem,
        ProcedureSymbol,
        {
            'calls': (),
            'targets': ('some_routine',)
        },
        ('typebound_item#some_routine',),
    ),
    (
        #######################################
        'typebound_item#some_type%some_routine',
        #######################################
        # This is a procedure binding, where the bound procedure is listed as blocked,
        # which excludes it from the targets list
        {
            'block': ['some_routine'],
        },
        ProcedureBindingItem,
        ProcedureSymbol,
        {
            'calls': (),
            'targets': ()
        },
        (),
    ),
    (
        ###################################
        'typebound_item#some_type%routine',
        ###################################
        # This is a procedure binding with renaming
        {},
        ProcedureBindingItem,
        ProcedureSymbol,
        {
            'calls': (),
            'targets': ('module_routine',),
        },
        ('typebound_item#module_routine',),
    ),
    (
        #############################
        'typebound_other#other_type',
        #############################
        # This is a derived type definition that has a dependency on another
        # type that is imported from another module, and renamed upon import
        {},
        TypeDefItem,
        TypeDef,
        {
            'calls': (),
            'targets': ('typebound_header', 'header')
        },
        ('typebound_header#header_type',),
    ),
])
def test_batch_typebound_item(
    testdir, default_config,
    name, config_override, item_type, ir_type, attrs_to_check, dependency_items
):
    """
    Test the basic regex frontend nodes in :any:`Item` objects for fast dependency detection
    for type-bound procedures.
    """
    default_config['default'].update(config_override)
    scheduler_config = SchedulerConfig.from_dict(default_config)
    item_factory = discover_proj_typebound_item_factory(testdir, scheduler_config)

    item = item_factory.item_cache[name]
    assert isinstance(item, item_type)
    assert isinstance(item.ir, ir_type)

    for key, value in attrs_to_check.items():
        assert getattr(item, key) == value

    assert item.create_dependency_items(item_factory, scheduler_config) == dependency_items


def test_batch_typebound_nested_item(testdir, default_config):
    """
    Test the basic regex frontend nodes in :any:`Item` objects for fast dependency detection
    for type-bound procedures for calls to nested derived type bindings
    """
    scheduler_config = SchedulerConfig.from_dict(default_config)
    item_factory = discover_proj_typebound_item_factory(testdir, scheduler_config)

    item = item_factory.item_cache['typebound_other#other_member']
    assert isinstance(item, ProcedureItem)
    assert isinstance(item.ir, Subroutine)
    assert len(item.dependencies) == 4

    assert isinstance(item.dependencies[0], ir.Import)
    assert item.dependencies[0].module == 'typebound_header'

    # Verify that the call to the nested type's routine is added when creating
    # dependency items
    assert 'typebound_other#other_type%var%member_routine' not in item_factory
    assert item.create_dependency_items(item_factory, scheduler_config) == (
        'typebound_other#other_type',
        'typebound_header#header_member_routine',
        'typebound_other#other_type%var%member_routine',
    )
    assert 'typebound_other#other_type%var%member_routine' in item_factory

    # Verify that the nested binding item can correctly resolve this to the binding
    # in the type
    proc_bind_item = item_factory.item_cache['typebound_other#other_type%var%member_routine']
    assert isinstance(proc_bind_item, ProcedureBindingItem)
    assert isinstance(proc_bind_item.ir, Scalar)
    assert proc_bind_item.ir == 'var'
    assert proc_bind_item.dependencies == ('var%member_routine',)
    assert proc_bind_item.create_dependency_items(item_factory, scheduler_config) == (
        'typebound_header#header_type%member_routine',
    )

    # Verify that the binding in the type correctly resolves to the module routine
    nested_bind_item = item_factory.item_cache['typebound_header#header_type%member_routine']
    assert isinstance(nested_bind_item, ProcedureBindingItem)
    assert isinstance(nested_bind_item.ir, ProcedureSymbol)
    assert nested_bind_item.ir == 'member_routine'
    assert nested_bind_item.create_dependency_items(item_factory, scheduler_config) == (
        'typebound_header#header_member_routine',
    )

    # Verify that we're now at the module routine
    routine_item = item_factory.item_cache['typebound_header#header_member_routine']
    assert isinstance(routine_item, ProcedureItem)
    assert isinstance(routine_item.ir, Subroutine)

    # Lastly, look at the deeply nested call...
    nested_call_item = item_factory.item_cache['typebound_other#nested_call']
    assert isinstance(nested_call_item, ProcedureItem)
    assert nested_call_item.create_dependency_items(item_factory, scheduler_config) == (
        'typebound_other#outer_type', 'typebound_other#outer_type%other%var%member_routine'
    )

    # ...and see if we can chase the deeply nested dependencies correctly
    other_var_member_item = item_factory.item_cache['typebound_other#outer_type%other%var%member_routine']
    assert isinstance(other_var_member_item, ProcedureBindingItem)
    assert isinstance(other_var_member_item.ir, Scalar)
    assert other_var_member_item.dependencies == ('other%var%member_routine',)
    assert other_var_member_item.create_dependency_items(item_factory, scheduler_config) == (
        'typebound_other#other_type%var%member_routine',
    )


def test_batch_typebound_item_targets(default_config):
    default_config['default']['disable'] += ['timer_mod']

    fcode = """
MODULE TYPEBOUND_ITEM_TARGETS_MOD
    USE TIMER_MOD, ONLY: PERFORMANCE_TIMER
    IMPLICIT NONE
CONTAINS
    SUBROUTINE DRIVER
        IMPLICIT NONE
        TYPE(PERFORMANCE_TIMER) :: TIMER

        CALL TIMER%START()

        ! DO SOMETHING

        CALL TIMER%END()
    END SUBROUTINE DRIVER
END MODULE TYPEBOUND_ITEM_TARGETS_MOD
    """.strip()

    source = Sourcefile.from_source(fcode, parser_classes=RegexParserClass.ProgramUnitClass, frontend=REGEX)
    source.path = 'None'
    item_factory = ItemFactory()
    scheduler_config = SchedulerConfig.from_dict(default_config)

    file_item = item_factory.get_or_create_file_item_from_source(source, scheduler_config)
    assert file_item.targets == ()
    assert file_item.definitions == (source['typebound_item_targets_mod'],)
    assert file_item.dependencies == ()

    file_definitions = file_item.create_definition_items(item_factory, scheduler_config)
    assert file_definitions == ('typebound_item_targets_mod',)

    module_item = file_definitions[0]
    assert module_item.targets == ()
    assert module_item.definitions == (source['driver'],)
    assert module_item.dependencies == module_item.ir.imports

    module_definitions = module_item.create_definition_items(item_factory, scheduler_config)
    assert module_definitions == ('typebound_item_targets_mod#driver',)
    assert module_item.create_dependency_items(item_factory, scheduler_config) == ()

    driver_item = module_definitions[0]
    assert driver_item.targets == ()
    assert driver_item.definitions == ()
    assert ('timer_mod', 'timer%start', 'timer%end') == (
        (driver_item.dependencies[0].module.lower(),) +
        tuple(dep.name for dep in driver_item.dependencies[1:])
    )

    assert driver_item.create_definition_items(item_factory, scheduler_config) == ()
    assert driver_item.create_dependency_items(item_factory, scheduler_config) == ()
