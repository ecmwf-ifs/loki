"""
Specialised test that exercises the bulk-processing capabilities and
source-injection mechanism provided by the `loki.scheduler` and
`loki.task` sub-modules.

Test directory structure

 - projA:
   - include
     - another_l1.intfb.h
     - another_l2.intfb.h
   - source
     - another_l1
     - another_l2
   - module
     - header_mod
       * header_type
     - driverA_mod
     - kernelA_mod
     - compute_l1_mod
     - compute_l2_mod
     - driverB_mod
     - kernelB_mod
     - driverC_mod
     - kernelC_mod
     - driverD_mod
     - kernelD_mod

 - projB:
   - external
     - ext_driver_mod
   - module
     - ext_kernel_mod

 - projC:
   - util
     - proj_c_util_mod
       * routine_one
       * routine_two
"""

import pytest
import shutil
from pathlib import Path
from collections import OrderedDict

from loki import (
    Scheduler, FP, SourceFile, FindNodes, CallStatement, fexprgen, Transformation
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builddir')
def fixture_builddir(here):
    builddir = here/'build'
    builddir.mkdir(parents=True, exist_ok=True)
    yield builddir

    # Clean up after us
    if builddir.exists():
        shutil.rmtree(builddir)


def test_scheduler_taskgraph_simple(here, builddir):
    """
    Create a simple task graph from a single sub-project:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routines': []
    }

    scheduler = Scheduler(paths=projA, includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.append('driverA')
    scheduler.populate()

    expected_nodes = ['driverA', 'kernelA', 'compute_l1', 'compute_l2', 'another_l1', 'another_l2']
    expected_edges = [
        'driverA -> kernelA',
        'kernelA -> compute_l1',
        'compute_l1 -> compute_l2',
        'kernelA -> another_l1',
        'another_l1 -> another_l2'
    ]

    nodes = [n.name for n in scheduler.taskgraph.nodes]
    edges = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in scheduler.taskgraph.edges]
    assert all(n in nodes for n in expected_nodes)
    assert all(e in edges for e in expected_edges)


def test_scheduler_taskgraph_partial(here, builddir):
    """
    Create a simple task graph from a single sub-project:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routine': [
            {
                'name': 'compute_l1',
                'role': 'driver',
                'expand': True,
            }, {
                'name': 'another_l1',
                'role': 'driver',
                'expand': True,
            },
        ]
    }
    # TODO: Note this is too convoluted, but mimicking the toml file config
    # reads we do in the current all-physics demonstrator. Needs internalising!
    config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, builddir=builddir)
    scheduler.append('driverA')
    scheduler.populate()

    # Check the correct sub-graph is generated
    nodes = [n.name for n in scheduler.taskgraph.nodes]
    edges = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in scheduler.taskgraph.edges]
    assert all(n in nodes for n in ['compute_l1', 'compute_l2', 'another_l1', 'another_l2'])
    assert all(e in edges for e in ['compute_l1 -> compute_l2', 'another_l1 -> another_l2'])


def test_scheduler_taskgraph_blocked(here, builddir):
    """
    Create a simple task graph with a single branch blocked:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           X --> <blocked>
    """
    projA = here/'sources/projA'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': ['another_l1']
        },
        'routines': []
    }

    scheduler = Scheduler(paths=projA, includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.append('driverA')
    scheduler.populate()

    expected_nodes = ['driverA', 'kernelA', 'compute_l1', 'compute_l2']
    expected_edges = [
        'driverA -> kernelA',
        'kernelA -> compute_l1',
        'compute_l1 -> compute_l2',
    ]

    nodes = [n.name for n in scheduler.taskgraph.nodes]
    edges = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in scheduler.taskgraph.edges]
    assert all(n in nodes for n in expected_nodes)
    assert all(e in edges for e in expected_edges)

    assert 'another_l1' not in nodes
    assert 'another_l2' not in nodes
    assert 'kernelA -> another_l1' not in edges
    assert 'another_l1 -> another_l2' not in edges


def test_scheduler_typedefs(here, builddir):
    """
    Create a simple task graph with and inject type info via `typedef`s.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                     <header_type>
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routines': []
    }

    header = SourceFile.from_file(projA/'module/header_mod.f90',
                                  builddir=builddir, frontend=FP)

    scheduler = Scheduler(paths=projA, typedefs=header['header_mod'].typedefs,
                          includes=projA/'include', config=config, builddir=builddir)
    scheduler.append('driverA')
    scheduler.populate()

    driver = scheduler.item_map['driverA'].routine
    call = FindNodes(CallStatement).visit(driver.body)[0]
    assert call.arguments[0].parent.type.variables
    assert fexprgen(call.arguments[0].shape) == '(:,)'
    assert call.arguments[1].parent.type.variables
    assert fexprgen(call.arguments[1].shape) == '(3, 3)'


def test_scheduler_process(here, builddir):
    """
    Create a simple task graph from a single sub-project
    and apply a simple transformation to it.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |      <driver>      <kernel>
                           |
                           | --> another_l1 -> another_l2
                                  <driver>      <kernel>
    """
    projA = here/'sources/projA'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': [],
            'whitelist': [],
        },
        'routine': [
            {
                'name': 'compute_l1',
                'role': 'driver',
                'expand': True,
            }, {
                'name': 'another_l1',
                'role': 'driver',
                'expand': True,
            },
        ]
    }
    config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, builddir=builddir)
    scheduler.append(config['routines'].keys())
    scheduler.populate()

    class AppendRole(Transformation):
        """
        Simply append role to subroutine names.
        """
        def transform_subroutine(self, routine, **kwargs):
            # TODO: This needs internalising!
            # Determine role in bulk-processing use case
            task = kwargs.get('task', None)
            role = kwargs.get('role') if task is None else task.config['role']
            routine.name += '_{}'.format(role)

    # Apply re-naming transformation and check result
    scheduler.process(transformation=AppendRole())
    assert scheduler.item_map['compute_l1'].routine.name == 'compute_l1_driver'
    assert scheduler.item_map['compute_l2'].routine.name == 'compute_l2_kernel'
    assert scheduler.item_map['another_l1'].routine.name == 'another_l1_driver'
    assert scheduler.item_map['another_l2'].routine.name == 'another_l2_kernel'


def test_scheduler_taskgraph_multiple_combined(here, builddir):
    """
    Create a single task graph spanning two projects

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """
    projA = here/'sources/projA'
    projB = here/'sources/projB'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routines': [],
    }

    scheduler = Scheduler(paths=[projA, projB], includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.append('driverB')
    scheduler.populate()

    expected_nodes = ['driverB', 'kernelB', 'compute_l1', 'compute_l2', 'ext_driver', 'ext_kernel']
    expected_edges = [
        'driverB -> kernelB',
        'kernelB -> compute_l1',
        'compute_l1 -> compute_l2',
        'kernelB -> ext_driver',
        'ext_driver -> ext_kernel'
    ]
    nodes = [n.name for n in scheduler.taskgraph.nodes]
    edges = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in scheduler.taskgraph.edges]
    assert all(n in nodes for n in expected_nodes)
    assert all(e in edges for e in expected_edges)


def test_scheduler_taskgraph_multiple_separate(here, builddir):
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
    projA = here/'sources/projA'
    projB = here/'sources/projB'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routine': [
            {
                'name': 'kernelB',
                'role': 'kernel',
                'ignore': ['ext_driver'],
                'enrich': ['ext_driver'],
            },
        ]
    }
    # TODO: Note this is too convoluted, but mimicking the toml file config
    # reads we do in the current all-physics demonstrator. Needs internalising!
    config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))

    schedulerA = Scheduler(paths=[projA, projB], includes=projA/'include',
                           config=config, builddir=builddir)
    schedulerA.append('driverB')
    schedulerA.populate()

    expected_nodesA = ['driverB', 'kernelB', 'compute_l1', 'compute_l2']
    expected_edgesA = ['driverB -> kernelB', 'kernelB -> compute_l1', 'compute_l1 -> compute_l2']

    nodesA = [n.name for n in schedulerA.taskgraph.nodes]
    edgesA = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in schedulerA.taskgraph.edges]
    assert all(n in nodesA for n in expected_nodesA)
    assert all(e in edgesA for e in expected_edgesA)
    assert 'ext_driver' not in nodesA
    assert 'ext_kernel' not in nodesA

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routine': [
            {
                'name': 'ext_driver',
                'role': 'kernel',
            },
        ]
    }
    # TODO: Note this is too convoluted, but mimicking the toml file config
    # reads we do in the current all-physics demonstrator. Needs internalising!
    config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))

    schedulerB = Scheduler(paths=projB, config=config, builddir=builddir)
    schedulerB.append('ext_driver')
    schedulerB.populate()

    # TODO: Technically we should check that the role=kernel has been honoured in B
    nodesB = [n.name for n in schedulerB.taskgraph.nodes]
    edgesB = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in schedulerB.taskgraph.edges]
    assert 'ext_driver' in nodesB
    assert 'ext_kernel' in nodesB
    assert 'ext_driver -> ext_kernel' in edgesB

    # Check that the call from kernelB to ext_driver has been enriched with IPA meta-info
    call = FindNodes(CallStatement).visit(schedulerA.item_map['kernelB'].routine.body)[1]
    assert call.context is not None
    assert fexprgen(call.context.routine.arguments) == '(vector(:), matrix(:, :))'


def test_scheduler_module_dependency(here, builddir):
    """
    Ensure dependency chasing is done correctly, even with surboutines
    that do not match module names.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 | --> routine_one -> routine_two
    """
    projA = here/'sources/projA'
    projC = here/'sources/projC'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routines': [],
    }

    scheduler = Scheduler(paths=[projA, projC], includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.append('driverC')
    scheduler.populate()

    expected_nodes = ['driverC', 'kernelC', 'compute_l1', 'compute_l2', 'routine_one', 'routine_two']
    expected_edges = [
        'driverC -> kernelC',
        'kernelC -> compute_l1',
        'compute_l1 -> compute_l2',
        'kernelC -> routine_one',
        'routine_one -> routine_two',
    ]
    nodes = [n.name for n in scheduler.taskgraph.nodes]
    edges = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in scheduler.taskgraph.edges]
    assert all(n in nodes for n in expected_nodes)
    assert all(e in edges for e in expected_edges)



def test_scheduler_module_dependencies_unqualified(here, builddir):
    """
    Ensure dependency chasing is done correctly for unqualified module imports.

    projA: driverD -> kernelD -> compute_l1<replicated> -> compute_l2
                           |
                    < proj_c_util_mod>
                           |
    projC:                 | --> routine_one -> routine_two
    """
    projA = here/'sources/projA'
    projC = here/'sources/projC'

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'blacklist': []
        },
        'routines': [],
    }

    scheduler = Scheduler(paths=[projA, projC], includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.append('driverD')
    scheduler.populate()

    expected_nodes = ['driverD', 'kernelD', 'compute_l1', 'compute_l2', 'routine_one', 'routine_two']
    expected_edges = [
        'driverD -> kernelD',
        'kernelD -> compute_l1',
        'compute_l1 -> compute_l2',
        'kernelD -> routine_one',
        'routine_one -> routine_two',
    ]
    nodes = [n.name for n in scheduler.taskgraph.nodes]
    edges = ['{} -> {}'.format(e1.name, e2.name) for e1, e2 in scheduler.taskgraph.edges]
    assert all(n in nodes for n in expected_nodes)
    assert all(e in edges for e in expected_edges)
