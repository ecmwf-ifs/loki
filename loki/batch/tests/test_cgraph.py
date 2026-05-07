# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki.batch import (
    CodeGraph, FileItem, Item, ItemFactory, ModuleItem, ProcedureBindingItem,
    ProcedureItem, SchedulerConfig, TypeDefItem, discover_file_codegraph
)
from loki.frontend import HAVE_FP, REGEX, RegexParserClass
from loki.sourcefile import Sourcefile


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


def test_cgraph_snapshot_preserves_item_names():
    """
    Test that CodeGraph snapshots preserve item names after original items are renamed.
    """
    parent = Item('parent', source=None)
    child = Item('parent#child', source=None)

    cgraph = CodeGraph()
    cgraph.add_edge((parent, child))
    snapshot = cgraph.snapshot()

    parent.name = 'renamed_parent'
    child.name = 'renamed_parent#child'

    assert snapshot.items == ('parent', 'parent#child')
    assert snapshot.definitions == (('parent', 'parent#child'),)


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
def test_cgraph_from_file_item(tmp_path):
    """
    Test file-local CodeGraph discovery from a FileItem.
    """
    fcode = """
module cgraph_mod
    implicit none
    type cgraph_type
    contains
        procedure :: proc => cgraph_type_proc
    end type cgraph_type
contains
    subroutine cgraph_kernel
    end subroutine cgraph_kernel

    subroutine cgraph_type_proc(this)
        class(cgraph_type), intent(inout) :: this
    end subroutine cgraph_type_proc
end module cgraph_mod

subroutine cgraph_free
end subroutine cgraph_free
    """.strip()
    filepath = tmp_path/'cgraph_file.F90'
    filepath.write_text(fcode)

    source = Sourcefile.from_file(
        filepath, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass
    )
    file_item = FileItem(str(filepath).lower(), source=source)

    cgraph, items = CodeGraph.from_file_item(file_item)

    assert set(items) == {
        str(filepath).lower(), 'cgraph_mod', '#cgraph_free',
        'cgraph_mod#cgraph_kernel', 'cgraph_mod#cgraph_type_proc',
        'cgraph_mod#cgraph_type', 'cgraph_mod#cgraph_type%proc'
    }
    assert set(cgraph.definitions) == {
        (file_item, 'cgraph_mod'),
        (file_item, '#cgraph_free'),
        ('cgraph_mod', 'cgraph_mod#cgraph_kernel'),
        ('cgraph_mod', 'cgraph_mod#cgraph_type_proc'),
        ('cgraph_mod', 'cgraph_mod#cgraph_type'),
        ('cgraph_mod#cgraph_type', 'cgraph_mod#cgraph_type%proc'),
    }


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
def test_cgraph_from_file_items_imports_canonical_items(tmp_path):
    """
    Test merging file-local CodeGraphs into a canonical ItemFactory.
    """
    fcode_a = """
module cgraph_mod_a
contains
    subroutine routine_a
    end subroutine routine_a
end module cgraph_mod_a
    """.strip()
    fcode_b = """
module cgraph_mod_b
contains
    subroutine routine_b
    end subroutine routine_b
end module cgraph_mod_b
    """.strip()

    path_a = tmp_path/'cgraph_a.F90'
    path_b = tmp_path/'cgraph_b.F90'
    path_a.write_text(fcode_a)
    path_b.write_text(fcode_b)

    source_a = Sourcefile.from_file(path_a, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)
    source_b = Sourcefile.from_file(path_b, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)
    file_item_a = FileItem(str(path_a).lower(), source=source_a)
    file_item_b = FileItem(str(path_b).lower(), source=source_b)

    item_factory = ItemFactory()
    item_factory.import_items((file_item_a, file_item_b))
    cgraph = CodeGraph.from_file_items((file_item_a, file_item_b), item_factory=item_factory)

    assert set(cgraph.items) == {
        str(path_a).lower(), str(path_b).lower(),
        'cgraph_mod_a', 'cgraph_mod_a#routine_a',
        'cgraph_mod_b', 'cgraph_mod_b#routine_b'
    }
    assert set(cgraph.definitions) == {
        (file_item_a, 'cgraph_mod_a'),
        ('cgraph_mod_a', 'cgraph_mod_a#routine_a'),
        (file_item_b, 'cgraph_mod_b'),
        ('cgraph_mod_b', 'cgraph_mod_b#routine_b'),
    }

    assert isinstance(item_factory.item_cache[str(path_a).lower()], FileItem)
    assert isinstance(item_factory.item_cache['cgraph_mod_a'], ModuleItem)
    assert isinstance(item_factory.item_cache['cgraph_mod_a#routine_a'], ProcedureItem)

    for parent, child in cgraph.definitions:
        assert parent is item_factory.item_cache[parent.name]
        assert child is item_factory.item_cache[child.name]


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
def test_cgraph_from_file_items_imports_typedef_bindings(tmp_path):
    """
    Test canonical import of TypeDefItem and ProcedureBindingItem nodes.
    """
    fcode = """
module cgraph_binding_mod
    type cgraph_binding_type
    contains
        procedure :: proc => cgraph_binding_proc
    end type cgraph_binding_type
contains
    subroutine cgraph_binding_proc(this)
        class(cgraph_binding_type), intent(inout) :: this
    end subroutine cgraph_binding_proc
end module cgraph_binding_mod
    """.strip()
    filepath = tmp_path/'cgraph_binding.F90'
    filepath.write_text(fcode)

    source = Sourcefile.from_file(filepath, frontend=REGEX, parser_classes=RegexParserClass.ProgramUnitClass)
    file_item = FileItem(str(filepath).lower(), source=source)
    item_factory = ItemFactory()
    cgraph = CodeGraph.from_file_items((file_item,), item_factory=item_factory)

    assert isinstance(item_factory.item_cache['cgraph_binding_mod#cgraph_binding_type'], TypeDefItem)
    assert isinstance(
        item_factory.item_cache['cgraph_binding_mod#cgraph_binding_type%proc'],
        ProcedureBindingItem
    )
    assert (
        'cgraph_binding_mod#cgraph_binding_type',
        'cgraph_binding_mod#cgraph_binding_type%proc'
    ) in cgraph.definitions


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
def test_discover_file_codegraph(tmp_path):
    """
    Test process-ready file-local CodeGraph discovery from a source path.
    """
    fcode = """
module cgraph_worker_mod
contains
    subroutine cgraph_worker_kernel
    end subroutine cgraph_worker_kernel
end module cgraph_worker_mod
    """.strip()
    filepath = tmp_path/'cgraph_worker.F90'
    filepath.write_text(fcode)

    config = SchedulerConfig.from_dict({'default': {'role': 'kernel'}, 'routines': {}})
    frontend_args = {'frontend': REGEX, 'parser_classes': RegexParserClass.ProgramUnitClass}
    cgraph, items = discover_file_codegraph(filepath, config=config, frontend_args=frontend_args)

    assert set(items) == {
        str(filepath).lower(), 'cgraph_worker_mod', 'cgraph_worker_mod#cgraph_worker_kernel'
    }
    assert set(cgraph.definitions) == {
        (str(filepath).lower(), 'cgraph_worker_mod'),
        ('cgraph_worker_mod', 'cgraph_worker_mod#cgraph_worker_kernel')
    }
    assert next(item for item in items if item == 'cgraph_worker_mod').role == 'kernel'


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
def test_discover_file_codegraph_from_dict_config(tmp_path):
    """
    Test process-ready file-local discovery with a dict config.
    """
    fcode = """
subroutine cgraph_worker_free
end subroutine cgraph_worker_free
    """.strip()
    filepath = tmp_path/'cgraph_worker_free.F90'
    filepath.write_text(fcode)

    frontend_args = {'frontend': REGEX, 'parser_classes': RegexParserClass.ProgramUnitClass}
    cgraph, items = discover_file_codegraph(
        filepath, config={'default': {'role': 'driver'}, 'routines': {}},
        frontend_args=frontend_args
    )

    assert set(items) == {str(filepath).lower(), '#cgraph_worker_free'}
    assert cgraph.definitions == ((str(filepath).lower(), '#cgraph_worker_free'),)
    assert next(item for item in items if item == '#cgraph_worker_free').role == 'driver'


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
@pytest.mark.parametrize('workers', [1, 2])
def test_cgraph_from_paths(tmp_path, workers):
    """
    Test CodeGraph discovery from source paths in serial and parallel modes.
    """
    fcode_a = """
module cgraph_paths_mod_a
contains
    subroutine cgraph_paths_kernel_a
    end subroutine cgraph_paths_kernel_a
end module cgraph_paths_mod_a
    """.strip()
    fcode_b = """
subroutine cgraph_paths_free_b
end subroutine cgraph_paths_free_b
    """.strip()

    path_a = tmp_path/'cgraph_paths_a.F90'
    path_b = tmp_path/'cgraph_paths_b.F90'
    path_a.write_text(fcode_a)
    path_b.write_text(fcode_b)

    item_factory = ItemFactory()
    cgraph = CodeGraph.from_paths(
        (path_b, path_a), item_factory=item_factory,
        config={'default': {'role': 'kernel'}, 'routines': {}},
        frontend_args={'frontend': REGEX, 'parser_classes': RegexParserClass.ProgramUnitClass},
        workers=workers
    )

    assert set(cgraph.items) == {
        str(path_a).lower(), str(path_b).lower(),
        'cgraph_paths_mod_a', 'cgraph_paths_mod_a#cgraph_paths_kernel_a',
        '#cgraph_paths_free_b'
    }
    assert set(cgraph.definitions) == {
        (str(path_a).lower(), 'cgraph_paths_mod_a'),
        ('cgraph_paths_mod_a', 'cgraph_paths_mod_a#cgraph_paths_kernel_a'),
        (str(path_b).lower(), '#cgraph_paths_free_b')
    }

    for parent, child in cgraph.definitions:
        assert parent is item_factory.item_cache[parent.name]
        assert child is item_factory.item_cache[child.name]


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
def test_cgraph_from_paths_parallel_matches_serial(tmp_path):
    """
    Test that parallel path discovery matches serial discovery.
    """
    paths = []
    for idx in range(3):
        fcode = f"""
module cgraph_parallel_mod_{idx}
contains
    subroutine cgraph_parallel_kernel_{idx}
    end subroutine cgraph_parallel_kernel_{idx}
end module cgraph_parallel_mod_{idx}
        """.strip()
        path = tmp_path/f'cgraph_parallel_{idx}.F90'
        path.write_text(fcode)
        paths.append(path)

    frontend_args = {'frontend': REGEX, 'parser_classes': RegexParserClass.ProgramUnitClass}
    serial_graph = CodeGraph.from_paths(paths, ItemFactory(), frontend_args=frontend_args, workers=1)
    parallel_graph = CodeGraph.from_paths(paths, ItemFactory(), frontend_args=frontend_args, workers=2)

    assert set(serial_graph.items) == set(parallel_graph.items)
    assert set(serial_graph.definitions) == set(parallel_graph.definitions)
