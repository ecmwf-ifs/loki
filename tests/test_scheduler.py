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

import shutil
from pathlib import Path
import pytest

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


@pytest.fixture(name='config')
def fixture_config():
    """
    Default configuration dict with basic options.
    """
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
        },
        'routines': []
    }


def test_scheduler_graph_simple(here, builddir, config):
    """
    Create a simple task graph from a single sub-project:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    scheduler = Scheduler(paths=projA, includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.populate('driverA')

    expected_items = [
        'driverA', 'kernelA', 'compute_l1', 'compute_l2', 'another_l1', 'another_l2'
    ]
    expected_dependencies = [
        ('driverA', 'kernelA'),
        ('kernelA', 'compute_l1'),
        ('compute_l1', 'compute_l2'),
        ('kernelA', 'another_l1'),
        ('another_l1', 'another_l2'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    # On-the-fly testing of callgraph visualisation
    cg_path = here/'callgraph_simple'
    scheduler.callgraph(cg_path)

    with cg_path.open('r') as f:
        cg_graph = f.read()
    graph_nodes = ['{} [color=black'.format(n.upper()) for n in expected_items]
    graph_edges = ['{} -> {}'.format(e1.upper(), e2.upper()) for e1, e2 in expected_dependencies]
    assert all(n in cg_graph for n in graph_nodes)
    assert all(e in cg_graph for e in graph_edges)


def test_scheduler_graph_partial(here, builddir, config):
    """
    Create a sub-graph from a select set of branches in  single project:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    config['routine'] = [
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

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, builddir=builddir)
    scheduler.populate('driverA')

    # Check the correct sub-graph is generated
    assert all(n in scheduler.items for n in ['compute_l1', 'compute_l2', 'another_l1', 'another_l2'])
    assert all(e in scheduler.dependencies for e in [('compute_l1', 'compute_l2'), ('another_l1', 'another_l2')])


def test_scheduler_graph_config_file(here, builddir):
    """
    Create a sub-graph from a branches using a config file:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = here/'sources/projA'
    config = projA/'scheduler_partial.config'

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, builddir=builddir)
    scheduler.populate('driverA')

    # Check the correct sub-graph is generated
    assert all(n in scheduler.items for n in ['compute_l1', 'another_l1', 'another_l2'])
    assert all(e in scheduler.dependencies for e in [('another_l1', 'another_l2')])
    assert 'compute_l2' not in scheduler.items  # We're blocking `compute_l2` in config file


def test_scheduler_graph_blocked(here, builddir, config):
    """
    Create a simple task graph with a single branch blocked:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           X --> <blocked>
    """
    projA = here/'sources/projA'

    config['default']['block'] = ['another_l1']

    scheduler = Scheduler(paths=projA, includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.populate('driverA')

    expected_items = [
        'driverA', 'kernelA', 'compute_l1', 'compute_l2'
    ]
    expected_dependencies = [
        ('driverA', 'kernelA'),
        ('kernelA', 'compute_l1'),
        ('compute_l1', 'compute_l2'),
    ]

    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    assert 'another_l1' not in scheduler.items
    assert 'another_l2' not in scheduler.items
    assert ('kernelA', 'another_l1') not in scheduler.dependencies
    assert ('another_l1', 'another_l2') not in scheduler.dependencies


def test_scheduler_typedefs(here, builddir, config):
    """
    Create a simple task graph with and inject type info via `typedef`s.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                     <header_type>
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    header = SourceFile.from_file(projA/'module/header_mod.f90',
                                  builddir=builddir, frontend=FP)

    scheduler = Scheduler(paths=projA, typedefs=header['header_mod'].typedefs,
                          includes=projA/'include', config=config, builddir=builddir)
    scheduler.populate('driverA')

    driver = scheduler.item_map['driverA'].routine
    call = FindNodes(CallStatement).visit(driver.body)[0]
    assert call.arguments[0].parent.type.variables
    assert fexprgen(call.arguments[0].shape) == '(:,)'
    assert call.arguments[1].parent.type.variables
    assert fexprgen(call.arguments[1].shape) == '(3, 3)'


def test_scheduler_process(here, builddir, config):
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

    config['routine'] = [
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

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, builddir=builddir)
    scheduler.populate([r['name'] for r in config['routine']])

    class AppendRole(Transformation):
        """
        Simply append role to subroutine names.
        """
        def transform_subroutine(self, routine, **kwargs):
            role = kwargs.get('role', None)
            routine.name += '_{}'.format(role)

    # Apply re-naming transformation and check result
    scheduler.process(transformation=AppendRole())
    assert scheduler.item_map['compute_l1'].routine.name == 'compute_l1_driver'
    assert scheduler.item_map['compute_l2'].routine.name == 'compute_l2_kernel'
    assert scheduler.item_map['another_l1'].routine.name == 'another_l1_driver'
    assert scheduler.item_map['another_l2'].routine.name == 'another_l2_kernel'


def test_scheduler_graph_multiple_combined(here, builddir, config):
    """
    Create a single task graph spanning two projects

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """
    projA = here/'sources/projA'
    projB = here/'sources/projB'

    scheduler = Scheduler(paths=[projA, projB], includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.populate('driverB')

    expected_items = [
        'driverB', 'kernelB', 'compute_l1', 'compute_l2', 'ext_driver', 'ext_kernel'
    ]
    expected_dependencies = [
        ('driverB', 'kernelB'),
        ('kernelB', 'compute_l1'),
        ('compute_l1', 'compute_l2'),
        ('kernelB', 'ext_driver'),
        ('ext_driver', 'ext_kernel'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)


def test_scheduler_graph_multiple_separate(here, builddir, config):
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

    configA = config.copy()
    configA['routine'] = [
        {
            'name': 'kernelB',
            'role': 'kernel',
            'ignore': ['ext_driver'],
            'enrich': ['ext_driver'],
        },
    ]

    schedulerA = Scheduler(paths=[projA, projB], includes=projA/'include',
                           config=configA, builddir=builddir)
    schedulerA.populate('driverB')

    expected_itemsA = [
        'driverB', 'kernelB', 'compute_l1', 'compute_l2'
    ]
    expected_dependenciesA = [
        ('driverB', 'kernelB'),
        ('kernelB', 'compute_l1'),
        ('compute_l1', 'compute_l2'),
     ]

    assert all(n in schedulerA.items for n in expected_itemsA)
    assert all(e in schedulerA.dependencies for e in expected_dependenciesA)
    assert 'ext_driver' not in schedulerA.items
    assert 'ext_kernel' not in schedulerA.items

    configB = config.copy()
    configB['routine'] = [
        {
            'name': 'ext_driver',
            'role': 'kernel',
        },
    ]

    schedulerB = Scheduler(paths=projB, config=configB, builddir=builddir)
    schedulerB.populate('ext_driver')

    # TODO: Technically we should check that the role=kernel has been honoured in B
    assert 'ext_driver' in schedulerB.items
    assert 'ext_kernel' in schedulerB.items
    assert ('ext_driver', 'ext_kernel') in schedulerB.dependencies

    # Check that the call from kernelB to ext_driver has been enriched with IPA meta-info
    call = FindNodes(CallStatement).visit(schedulerA.item_map['kernelB'].routine.body)[1]
    assert call.context is not None
    assert fexprgen(call.context.routine.arguments) == '(vector(:), matrix(:, :))'


def test_scheduler_module_dependency(here, builddir, config):
    """
    Ensure dependency chasing is done correctly, even with surboutines
    that do not match module names.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 | --> routine_one -> routine_two
    """
    projA = here/'sources/projA'
    projC = here/'sources/projC'

    scheduler = Scheduler(paths=[projA, projC], includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.populate('driverC')

    expected_items = [
        'driverC', 'kernelC', 'compute_l1', 'compute_l2', 'routine_one', 'routine_two'
    ]
    expected_dependencies = [
        ('driverC', 'kernelC'),
        ('kernelC', 'compute_l1'),
        ('compute_l1', 'compute_l2'),
        ('kernelC', 'routine_one'),
        ('routine_one', 'routine_two'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    # Ensure that we got the right routines from the module
    assert scheduler.item_map['routine_one'].routine.name == 'routine_one'
    assert scheduler.item_map['routine_two'].routine.name == 'routine_two'


def test_scheduler_module_dependencies_unqualified(here, builddir, config):
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

    scheduler = Scheduler(paths=[projA, projC], includes=projA/'include',
                          config=config, builddir=builddir)
    scheduler.populate('driverD')

    expected_items = [
        'driverD', 'kernelD', 'compute_l1', 'compute_l2', 'routine_one', 'routine_two'
    ]
    expected_dependencies = [
        ('driverD', 'kernelD'),
        ('kernelD', 'compute_l1'),
        ('compute_l1', 'compute_l2'),
        ('kernelD', 'routine_one'),
        ('routine_one', 'routine_two'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    # Ensure that we got the right routines from the module
    assert scheduler.item_map['routine_one'].routine.name == 'routine_one'
    assert scheduler.item_map['routine_two'].routine.name == 'routine_two'
