# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from itertools import chain

import pytest

from loki import Subroutine, fexprgen, graphviz_present
from loki.batch import Scheduler, TypeDefItem
from loki.frontend import HAVE_FP
from loki.ir import FindNodes, nodes as ir

from .conftest import VisGraphWrapper

pytestmark = pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')


@pytest.fixture(name='driverA_dependencies')
def fixture_drivera_dependencies():
    return {
        'driverA_mod#driverA': ('kernelA_mod#kernelA', 'header_mod', 'header_mod#header_type'),
        'kernelA_mod#kernelA': ('compute_l1_mod#compute_l1', '#another_l1'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        '#another_l1': ('#another_l2', 'header_mod'),
        '#another_l2': ('header_mod',),
        'header_mod': (),
        'header_mod#header_type': (),
    }


@pytest.fixture(name='driverB_dependencies')
def fixture_driverb_dependencies():
    return {
        'driverB_mod#driverB': (
            'kernelB_mod#kernelB',
            'header_mod#header_type',
            'header_mod'
        ),
        'kernelB_mod#kernelB': ('compute_l1_mod#compute_l1', 'ext_driver_mod#ext_driver'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'ext_driver_mod#ext_driver': ('ext_kernel_mod', 'ext_kernel_mod#ext_kernel',),
        'ext_kernel_mod': (),
        'ext_kernel_mod#ext_kernel': (),
        'header_mod#header_type': (),
        'header_mod': (),
    }


def test_scheduler_cgraph_original_code_state(testdir, config, frontend, tmp_path):
    """
    Create a code graph for all visible definitions while keeping the dependency graph seed-based.
    """
    projA = testdir/'sources/projA'
    paths = [projA/'module', projA/'source/another_l1.F90', projA/'source/another_l2.F90']

    scheduler = Scheduler(
        paths=paths, includes=projA/'include', config=config,
        seed_routines='driverA', frontend=frontend, xmods=[tmp_path]
    )

    expected_definition_edges = {
        (str(projA/'module/driverA_mod.f90').lower(), 'drivera_mod'),
        ('drivera_mod', 'drivera_mod#drivera'),
        (str(projA/'module/kernelA_mod.F90').lower(), 'kernela_mod'),
        ('kernela_mod', 'kernela_mod#kernela'),
        (str(projA/'module/header_mod.f90').lower(), 'header_mod'),
        ('header_mod', 'header_mod#header_type'),
        (str(projA/'source/another_l1.F90').lower(), '#another_l1'),
    }

    assert expected_definition_edges <= set(scheduler.cgraph.definitions)
    assert 'drivera_mod#drivera' in scheduler.sgraph.items
    assert str(projA/'module/driverA_mod.f90').lower() not in scheduler.sgraph.items
    assert str(projA/'module/driverA_mod.f90').lower() in scheduler.cgraph.items
    assert isinstance(scheduler.item_factory.item_cache['header_mod#header_type'], TypeDefItem)


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
@pytest.mark.parametrize('with_file_graph', [True, False, 'filegraph_simple'])
@pytest.mark.parametrize('with_legend', [True, False])
@pytest.mark.parametrize('seed', ['driverA', 'driverA_mod#driverA'])
def test_scheduler_graph_simple(
        tmp_path, testdir, config, frontend, driverA_dependencies,
        with_file_graph, with_legend, seed
):
    """
    Create a simple task graph from a single sub-project:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """

    projA = testdir/'sources/projA'
    # Combine directory globbing and explicit file paths for lookup
    paths = [projA/'module', projA/'source/another_l1.F90', projA/'source/another_l2.F90']

    scheduler = Scheduler(
        paths=paths, includes=projA/'include', config=config,
        seed_routines=seed, frontend=frontend, xmods=[tmp_path]
    )

    assert set(scheduler.items) == {item.lower() for item in driverA_dependencies}
    assert set(scheduler.dependencies) == {
        (item.lower(), child.lower())
        for item, children in driverA_dependencies.items()
        for child in children
    }

    if with_file_graph:
        file_graph = scheduler.file_graph
        expected_file_dependencies = {
            'module/driverA_mod.f90': ('module/kernelA_mod.F90', 'module/header_mod.f90'),
            'module/kernelA_mod.F90': ('module/compute_l1_mod.f90', 'source/another_l1.F90'),
            'module/compute_l1_mod.f90': ('module/compute_l2_mod.f90',),
            'module/compute_l2_mod.f90': (),
            'source/another_l1.F90': ('source/another_l2.F90', 'module/header_mod.f90'),
            'source/another_l2.F90': ('module/header_mod.f90',),
            'module/header_mod.f90': (),
        }
        assert set(file_graph.items) == {str(projA/name).lower() for name in expected_file_dependencies}
        assert set(file_graph.dependencies) == {
            (str(projA/a).lower(), str(projA/b).lower())
            for a, deps in expected_file_dependencies.items() for b in deps
        }

    cg_path = tmp_path/'callgraph_simple'
    if not isinstance(with_file_graph, bool):
        with_file_graph = tmp_path/with_file_graph
    scheduler.callgraph(cg_path, with_file_graph=with_file_graph, with_legend=with_legend)

    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)
    if with_legend:
        assert set(vgraph.nodes) == {item.upper() for item in driverA_dependencies} | {
            'FileItem', 'ModuleItem', 'ProcedureItem', 'TypeDefItem',
            'ProcedureBindingItem', 'InterfaceItem', 'ExternalItem'
        }
    else:
        assert set(vgraph.nodes) == {item.upper() for item in driverA_dependencies}
    assert set(vgraph.edges) == {
        (item.upper(), child.upper())
        for item, children in driverA_dependencies.items()
        for child in children
    }

    if with_file_graph:
        if isinstance(with_file_graph, bool):
            fg_path = cg_path.with_name(f'{cg_path.stem}_file_graph{cg_path.suffix}')
        else:
            fg_path = tmp_path/with_file_graph
        fgraph = VisGraphWrapper(fg_path)
        assert set(fgraph.nodes) == {name.lower() for name in expected_file_dependencies}
        assert set(fgraph.edges) == {
            (a.lower(), b.lower())
            for a, deps in expected_file_dependencies.items() for b in deps
        }

        fg_path.unlink()
        fg_path.with_suffix('.pdf').unlink(missing_ok=True)

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink(missing_ok=True)


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
@pytest.mark.parametrize('seed', ['compute_l1', 'compute_l1_mod#compute_l1'])
def test_scheduler_graph_partial(tmp_path, testdir, config, frontend, seed):
    """
    Create a sub-graph from a select set of branches in  single project:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = testdir/'sources/projA'

    config['routines'] = {
        seed: {
            'role': 'driver',
            'expand': True,
        },
        'another_l1': {
            'role': 'driver',
            'expand': True,
        },
    }

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, frontend=frontend, xmods=[tmp_path])

    expected_items = [
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2', '#another_l1', '#another_l2'
    ]
    expected_dependencies = [
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
        ('#another_l1', '#another_l2')
    ]

    # Check the correct sub-graph is generated
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)
    assert 'driverA' not in scheduler.items
    assert 'kernelA' not in scheduler.items

    cg_path = tmp_path/'callgraph_partial'
    scheduler.callgraph(cg_path)

    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)
    assert 'DRIVERA' not in vgraph.nodes
    assert 'KERNELA' not in vgraph.nodes

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_graph_config_file(tmp_path, testdir, frontend):
    """
    Create a sub-graph from a branches using a config file:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = testdir/'sources/projA'
    # We're blocking `compute_l2` in config file
    config = projA/'scheduler_partial.config'

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, frontend=frontend, xmods=[tmp_path])

    expected_dependencies = {
        'compute_l1_mod#compute_l1': (),
        '#another_l1': ('#another_l2', 'header_mod'),
        '#another_l2': ('header_mod',),
        'header_mod': (),
    }

    # Check the correct sub-graph is generated
    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }
    assert 'compute_l2' not in scheduler.items

    cg_path = tmp_path/'callgraph_config_file'
    scheduler.callgraph(cg_path)
    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)

    # We're blocking compute_l2 but it's still in the VGraph
    assert set(vgraph.nodes) == {name.upper() for name in expected_dependencies} | {'COMPUTE_L2'}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in expected_dependencies.items() for b in deps
    } | {('COMPUTE_L1_MOD#COMPUTE_L1', 'COMPUTE_L2')}

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
@pytest.mark.parametrize('seed', ['driverA', 'driverA_mod#driverA'])
def test_scheduler_graph_blocked(tmp_path, testdir, config, frontend, seed):
    """
    Create a simple task graph with a single branch blocked:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           X --> <blocked>
    """
    projA = testdir/'sources/projA'

    config['default']['block'] = ['another_l1']

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=[seed], frontend=frontend, xmods=[tmp_path]
    )

    expected_dependencies = {
        'drivera_mod#drivera': ('kernela_mod#kernela', 'header_mod', 'header_mod#header_type'),
        'kernela_mod#kernela': ('compute_l1_mod#compute_l1',),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'header_mod#header_type': (),
        'header_mod': ()
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    assert '#another_l1' not in scheduler.items
    assert '#another_l2' not in scheduler.items
    assert ('kernelA', 'another_l1') not in scheduler.dependencies
    assert ('another_l1', 'another_l2') not in scheduler.dependencies

    cg_path = tmp_path/'callgraph_block'
    fg_path = cg_path.with_name(cg_path.name + '_file_graph')
    scheduler.callgraph(cg_path, with_file_graph=True)
    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)

    # We're blocking another_l1, but it's still in the VGraph
    assert set(vgraph.nodes) == {n.upper() for n in expected_dependencies} | {'ANOTHER_L1'}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in expected_dependencies.items() for b in deps
    } | {('KERNELA_MOD#KERNELA', 'ANOTHER_L1')}

    file_dependencies = {
        'drivera_mod.f90': ('kernela_mod.f90', 'header_mod.f90'),
        'kernela_mod.f90': ('compute_l1_mod.f90',),
        'compute_l1_mod.f90': ('compute_l2_mod.f90',),
        'compute_l2_mod.f90': (),
        'header_mod.f90': ()
    }

    vgraph = VisGraphWrapper(fg_path)
    assert set(vgraph.nodes) == set(file_dependencies)
    assert set(vgraph.edges) == {(a, b) for a, deps in file_dependencies.items() for b in deps}

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()
    fg_path.unlink()
    if fg_path.with_suffix('.pdf').exists():
        fg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_graph_multiple_combined(tmp_path, testdir, config, driverB_dependencies, frontend):
    """
    Create a single task graph spanning two projects

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """
    projA = testdir/'sources/projA'
    projB = testdir/'sources/projB'

    scheduler = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=config,
        seed_routines=['driverB_mod#driverB'], frontend=frontend, xmods=[tmp_path]
    )

    assert set(scheduler.items) == {item.lower() for item in driverB_dependencies}
    assert set(scheduler.dependencies) == {
        (item.lower(), child.lower())
        for item, children in driverB_dependencies.items()
        for child in children
    }

    cg_path = tmp_path/'callgraph_multiple_combined'
    scheduler.callgraph(cg_path)

    # Testing of callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {item.upper() for item in driverB_dependencies}
    assert set(vgraph.edges) == {
        (item.upper(), child.upper())
        for item, children in driverB_dependencies.items()
        for child in children
    }

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_graph_multiple_separate(tmp_path, testdir, config, frontend):
    """
    Tests combining two scheduler graphs, where that an individual
    sub-branch is pruned in the driver schedule, while IPA meta-info
    is still injected to create a seamless jump between two distinct
    schedules for projA and projB

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
                     <ext_driver>

    projB:            ext_driver -> ext_kernel
    """
    projA = testdir/'sources/projA'
    projB = testdir/'sources/projB'

    configA = config.copy()
    configA['routines'] = {
        'kernelB': {
            'role': 'kernel',
            'ignore': ['ext_driver'],
            'enrich': ['ext_driver'],
        },
    }

    schedulerA = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=configA,
        seed_routines=['driverB'], frontend=frontend, xmods=[tmp_path]
    )

    expected_dependenciesA = {
        'driverb_mod#driverb': (
            'kernelb_mod#kernelb',
            'header_mod#header_type',
            'header_mod',
        ),
        'kernelb_mod#kernelb': (
            'compute_l1_mod#compute_l1',
            'ext_driver_mod#ext_driver',
        ),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'header_mod#header_type': (),
        'header_mod': (),
    }

    ignored_dependenciesA = {
        'ext_driver_mod#ext_driver': ('ext_kernel_mod', 'ext_kernel_mod#ext_kernel',),
        'ext_kernel_mod': (),
        'ext_kernel_mod#ext_kernel': (),
    }

    assert set(schedulerA.items) == set(chain(expected_dependenciesA, ignored_dependenciesA))
    assert set(schedulerA.dependencies) == {
        (a, b)
        for a, deps in chain(expected_dependenciesA.items(), ignored_dependenciesA.items())
        for b in deps
    }
    assert all(schedulerA[name].is_ignored for name in ignored_dependenciesA)
    assert all(not schedulerA[name].is_ignored for name in expected_dependenciesA)

    cg_path = tmp_path/'callgraph_multiple_separate_A'
    schedulerA.callgraph(cg_path)

    # Test callgraph visualisation
    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {n.upper() for n in chain(expected_dependenciesA, ignored_dependenciesA)}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper())
        for a, deps in chain(expected_dependenciesA.items(), ignored_dependenciesA.items())
        for b in deps
    }

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()

    # Test second scheduler instance that holds the receiver items
    configB = config.copy()
    configB['routines'] = {
        'ext_driver': {'role': 'kernel'},
    }

    schedulerB = Scheduler(
        paths=projB, config=configB, seed_routines=['ext_driver'],
        frontend=frontend, xmods=[tmp_path]
    )

    assert 'ext_driver_mod#ext_driver' in schedulerB.items
    assert 'ext_kernel_mod#ext_kernel' in schedulerB.items
    assert ('ext_driver_mod#ext_driver', 'ext_kernel_mod#ext_kernel') in schedulerB.dependencies

    call = FindNodes(ir.CallStatement).visit(schedulerA['kernelb_mod#kernelb'].ir.body)[1]
    assert isinstance(call.routine, Subroutine)
    # Check that the call from kernelB to ext_driver has been enriched with IPA meta-info
    assert fexprgen(call.routine.arguments) == '(vector(:), matrix(:, :))'

    cg_path = tmp_path/'callgraph_multiple_separate_B'
    schedulerB.callgraph(cg_path)

    # Test callgraph visualisation
    vgraphB = VisGraphWrapper(cg_path)
    assert 'EXT_DRIVER_MOD#EXT_DRIVER' in vgraphB.nodes
    assert 'EXT_KERNEL_MOD#EXT_KERNEL' in vgraphB.nodes
    assert ('EXT_DRIVER_MOD#EXT_DRIVER', 'EXT_KERNEL_MOD#EXT_KERNEL') in vgraphB.edges

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


def test_scheduler_depths(testdir, config, frontend, tmp_path):
    projA = testdir/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=['driverA'], frontend=frontend, xmods=[tmp_path]
    )

    expected_depths = {
        'drivera_mod#drivera': 0,
        'header_mod#header_type': 1,
        'kernela_mod#kernela': 1,
        'compute_l1_mod#compute_l1': 2,
        '#another_l1': 2,
        'compute_l2_mod#compute_l2': 3,
        '#another_l2': 3,
        'header_mod': 4,
    }
    assert scheduler.sgraph.depths == expected_depths
