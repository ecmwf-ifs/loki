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
import re
from pathlib import Path
import pytest

from loki import (
    Scheduler, FP, Sourcefile, FindNodes, CallStatement, fexprgen, Transformation, BasicType
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


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


class VisGraphWrapper:
    """
    Testing utility to parse the generated callgraph visualisation.
    """

    _re_nodes = re.compile(r'\s*(?P<node>\w+) \[color', re.IGNORECASE)
    _re_edges = re.compile(r'\s*(?P<parent>\w+) -> (?P<child>\w+)', re.IGNORECASE)

    def __init__(self, path):
        with Path(path).open('r') as f:
            self.text = f.read()

    @property
    def nodes(self):
        return [m for m in self._re_nodes.findall(self.text)]

    @property
    def edges(self):
        return [m for m in self._re_edges.findall(self.text)]


def test_scheduler_graph_simple(here, config):
    """
    Create a simple task graph from a single sub-project:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config)
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

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_simple'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)

    cg_path.unlink()


def test_scheduler_graph_partial(here, config):
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

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config)
    scheduler.populate(scheduler.config.routines)

    expected_items = [
        'compute_l1', 'compute_l2', 'another_l1', 'another_l2'
    ]
    expected_dependencies = [
        ('compute_l1', 'compute_l2'),
        ('another_l1', 'another_l2')
    ]

    # Check the correct sub-graph is generated
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)
    assert 'driverA' not in scheduler.items
    assert 'kernelA' not in scheduler.items

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_partial'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)
    assert 'DRIVERA' not in vgraph.nodes
    assert 'KERNELA' not in vgraph.nodes

    cg_path.unlink()


def test_scheduler_graph_config_file(here):
    """
    Create a sub-graph from a branches using a config file:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = here/'sources/projA'
    config = projA/'scheduler_partial.config'

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config)
    scheduler.populate(scheduler.config.routines)

    expected_items = ['compute_l1', 'another_l1', 'another_l2']
    expected_dependencies = [('another_l1', 'another_l2')]

    # Check the correct sub-graph is generated
    assert all(n in scheduler.items for n in [])
    assert all(e in scheduler.dependencies for e in [])
    assert 'compute_l2' not in scheduler.items  # We're blocking `compute_l2` in config file

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_config_file'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)
    assert 'COMPUTE_L2' in vgraph.nodes  # We're blocking this, but it's still in the VGraph
    assert ('COMPUTE_L1', 'COMPUTE_L2') in vgraph.edges
    assert len(vgraph.nodes) == 4
    assert len(vgraph.edges) == 2

    cg_path.unlink()


def test_scheduler_graph_blocked(here, config):
    """
    Create a simple task graph with a single branch blocked:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           X --> <blocked>
    """
    projA = here/'sources/projA'

    config['default']['block'] = ['another_l1']

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config)
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

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_block'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)
    assert 'ANOTHER_L1' in vgraph.nodes  # We're blocking this, but it's still in the VGraph
    assert 'ANOTHER_L2' not in vgraph.nodes
    assert ('KERNELA', 'ANOTHER_L1') in vgraph.edges
    assert len(vgraph.nodes) == 5
    assert len(vgraph.edges) == 4

    cg_path.unlink()


def test_scheduler_definitions(here, config):
    """
    Create a simple task graph with and inject type info via `definitions`.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                     <header_type>
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    header = Sourcefile.from_file(projA/'module/header_mod.f90', frontend=FP)

    scheduler = Scheduler(paths=projA, definitions=header['header_mod'],
                          includes=projA/'include', config=config)
    scheduler.populate('driverA')

    driver = scheduler.item_map['driverA'].routine
    call = FindNodes(CallStatement).visit(driver.body)[0]
    assert call.arguments[0].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[0].shape) == '(:,)'
    assert call.arguments[1].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[1].shape) == '(3, 3)'


def test_scheduler_process(here, config):
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

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config)
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


def test_scheduler_graph_multiple_combined(here, config):
    """
    Create a single task graph spanning two projects

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """
    projA = here/'sources/projA'
    projB = here/'sources/projB'

    scheduler = Scheduler(paths=[projA, projB], includes=projA/'include', config=config)
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

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_multiple_combined'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)

    cg_path.unlink()


def test_scheduler_graph_multiple_separate(here, config):
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

    schedulerA = Scheduler(paths=[projA, projB], includes=projA/'include', config=configA)
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
    # assert 'ext_driver' not in schedulerA.items
    # assert 'ext_kernel' not in schedulerA.items

    # Test callgraph visualisation
    cg_path = here/'callgraph_multiple_separate_A'
    schedulerA.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_itemsA)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependenciesA)

    cg_path.unlink()

    # Test second scheduler instance that holds the receiver items
    configB = config.copy()
    configB['routine'] = [
        {
            'name': 'ext_driver',
            'role': 'kernel',
        },
    ]

    schedulerB = Scheduler(paths=projB, config=configB)
    schedulerB.populate('ext_driver')

    # TODO: Technically we should check that the role=kernel has been honoured in B
    assert 'ext_driver' in schedulerB.items
    assert 'ext_kernel' in schedulerB.items
    assert ('ext_driver', 'ext_kernel') in schedulerB.dependencies

    # Check that the call from kernelB to ext_driver has been enriched with IPA meta-info
    call = FindNodes(CallStatement).visit(schedulerA.item_map['kernelB'].routine.body)[1]
    assert call.context is not None
    assert fexprgen(call.context.routine.arguments) == '(vector(:), matrix(:, :))'

    # Test callgraph visualisation
    cg_path = here/'callgraph_multiple_separate_B'
    schedulerB.callgraph(cg_path)

    vgraphB = VisGraphWrapper(cg_path)
    assert 'EXT_DRIVER' in vgraphB.nodes
    assert 'EXT_KERNEL' in vgraphB.nodes
    assert ('EXT_DRIVER', 'EXT_KERNEL') in vgraphB.edges

    cg_path.unlink()


def test_scheduler_module_dependency(here, config):
    """
    Ensure dependency chasing is done correctly, even with surboutines
    that do not match module names.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 | --> routine_one -> routine_two
    """
    projA = here/'sources/projA'
    projC = here/'sources/projC'

    scheduler = Scheduler(paths=[projA, projC], includes=projA/'include', config=config)
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


def test_scheduler_module_dependencies_unqualified(here, config):
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

    scheduler = Scheduler(paths=[projA, projC], includes=projA/'include', config=config)
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
