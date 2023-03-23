# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

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

import importlib
import re
from pathlib import Path
from shutil import rmtree
import pytest

from conftest import available_frontends
from loki import (
    Scheduler, SchedulerConfig, DependencyTransformation, FP, OFP,
    HAVE_FP, HAVE_OFP, REGEX, Sourcefile, FindNodes, CallStatement,
    fexprgen, Transformation, BasicType, CMakePlanner, Subroutine,
    SubroutineItem, ProcedureBindingItem, gettempdir, ProcedureSymbol,
    ProcedureType, DerivedType, TypeDef, Scalar, Array, FindInlineCalls,
    Import, Variable
)


pytestmark = pytest.mark.skipif(not HAVE_FP and not HAVE_OFP, reason='Fparser and OFP not available')


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
            'disable': ['abort']
        },
        'routines': []
    }


@pytest.fixture(name='frontend')
def fixture_frontend():
    """
    Frontend to use.

    Not parametrizing the tests as the scheduler functionality should be
    independent from the specific frontend used. Cannot use OMNI for this
    as not all tests have dependencies fully resolved.
    """
    return FP if HAVE_FP else OFP


class VisGraphWrapper:
    """
    Testing utility to parse the generated callgraph visualisation.
    """

    _re_nodes = re.compile(r'\s*\"?(?P<node>[\w%#./]+)\"? \[colo', re.IGNORECASE)
    _re_edges = re.compile(r'\s*\"?(?P<parent>[\w%#./]+)\"? -> \"?(?P<child>[\w%#./]+)\"?', re.IGNORECASE)

    def __init__(self, path):
        with Path(path).open('r') as f:
            self.text = f.read()

    @property
    def nodes(self):
        return list(self._re_nodes.findall(self.text))

    @property
    def edges(self):
        return list(self._re_edges.findall(self.text))


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
@pytest.mark.parametrize('with_file_graph', [True, False, 'filegraph_simple'])
def test_scheduler_graph_simple(here, config, frontend, with_file_graph):
    """
    Create a simple task graph from a single sub-project:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=['driverA'], frontend=frontend
    )

    expected_items = [
        'driverA_mod#driverA', 'kernelA_mod#kernelA',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2',
        '#another_l1', '#another_l2'
    ]
    expected_dependencies = [
        ('driverA_mod#driverA', 'kernelA_mod#kernelA'),
        ('kernelA_mod#kernelA', 'compute_l1_mod#compute_l1'),
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
        ('kernelA_mod#kernelA', '#another_l1'),
        ('#another_l1', '#another_l2'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    if with_file_graph:
        file_graph = scheduler.file_graph
        expected_files = [
            'module/driverA_mod.f90', 'module/kernelA_mod.F90',
            'module/compute_l1_mod.f90', 'module/compute_l2_mod.f90',
            'source/another_l1.F90', 'source/another_l2.F90',
        ]
        expected_file_dependencies = [
            ('module/driverA_mod.f90', 'module/kernelA_mod.F90'),
            ('module/kernelA_mod.F90', 'module/compute_l1_mod.f90'),
            ('module/compute_l1_mod.f90', 'module/compute_l2_mod.f90'),
            ('module/kernelA_mod.F90', 'source/another_l1.F90'),
            ('source/another_l1.F90', 'source/another_l2.F90'),
        ]
        assert all(Path(n) in file_graph for n in expected_files)
        assert all((Path(a), Path(b)) in file_graph.edges for a, b in expected_file_dependencies)

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_simple'
    if not isinstance(with_file_graph, bool):
        with_file_graph = here/with_file_graph
    scheduler.callgraph(cg_path, with_file_graph=with_file_graph)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)

    if with_file_graph:
        if isinstance(with_file_graph, bool):
            fg_path = cg_path.with_name(f'{cg_path.stem}_file_graph{cg_path.suffix}')
        else:
            fg_path = here/with_file_graph
        fgraph = VisGraphWrapper(fg_path)
        assert all(n in fgraph.nodes for n in expected_files)
        assert all((e[0], e[1]) in fgraph.edges for e in expected_file_dependencies)

        fg_path.unlink()
        fg_path.with_suffix('.pdf').unlink(missing_ok=True)

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink(missing_ok=True)


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
def test_scheduler_graph_partial(here, config, frontend):
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

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, frontend=frontend)

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

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_partial'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)
    assert 'DRIVERA' not in vgraph.nodes
    assert 'KERNELA' not in vgraph.nodes

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
def test_scheduler_graph_config_file(here, frontend):
    """
    Create a sub-graph from a branches using a config file:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = here/'sources/projA'
    config = projA/'scheduler_partial.config'

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, frontend=frontend)

    expected_items = ['compute_l1_mod#compute_l1', '#another_l1', '#another_l2']
    expected_dependencies = [('#another_l1', '#another_l2')]

    # Check the correct sub-graph is generated
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)
    assert 'compute_l2' not in scheduler.items  # We're blocking `compute_l2` in config file

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_config_file'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)
    assert 'COMPUTE_L2_MOD#COMPUTE_L2' in vgraph.nodes  # We're blocking this, but it's still in the VGraph
    assert ('COMPUTE_L1_MOD#COMPUTE_L1', 'COMPUTE_L2_MOD#COMPUTE_L2') in vgraph.edges
    assert len(vgraph.nodes) == 4
    assert len(vgraph.edges) == 2

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
def test_scheduler_graph_blocked(here, config, frontend):
    """
    Create a simple task graph with a single branch blocked:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           X --> <blocked>
    """
    projA = here/'sources/projA'

    config['default']['block'] = ['another_l1']

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=['driverA'], frontend=frontend
    )

    expected_items = [
        'driverA_mod#driverA', 'kernelA_mod#kernelA', 'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'
    ]
    expected_dependencies = [
        ('driverA_mod#driverA', 'kernelA_mod#kernelA'),
        ('kernelA_mod#kernelA', 'compute_l1_mod#compute_l1'),
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
    ]

    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    assert '#another_l1' not in scheduler.items
    assert '#another_l2' not in scheduler.items
    assert ('kernelA', 'another_l1') not in scheduler.dependencies
    assert ('another_l1', 'another_l2') not in scheduler.dependencies

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_block'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)
    assert '#ANOTHER_L1' in vgraph.nodes  # We're blocking this, but it's still in the VGraph
    assert '#ANOTHER_L2' not in vgraph.nodes
    assert ('KERNELA_MOD#KERNELA', '#ANOTHER_L1') in vgraph.edges
    assert len(vgraph.nodes) == 5
    assert len(vgraph.edges) == 4

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


def test_scheduler_definitions(here, config, frontend):
    """
    Create a simple task graph with and inject type info via `definitions`.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                     <header_type>
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    header = Sourcefile.from_file(projA/'module/header_mod.f90', frontend=frontend)

    scheduler = Scheduler(
        paths=projA, definitions=header['header_mod'], includes=projA/'include',
        config=config, seed_routines=['driverA'], frontend=frontend
    )

    driver = scheduler.item_map['drivera_mod#drivera'].routine
    call = FindNodes(CallStatement).visit(driver.body)[0]
    assert call.arguments[0].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[0].shape) == '(:,)'
    assert call.arguments[1].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[1].shape) == '(3, 3)'


def test_scheduler_process(here, config, frontend):
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

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, frontend=frontend)

    class AppendRole(Transformation):
        """
        Simply append role to subroutine names.
        """
        def transform_subroutine(self, routine, **kwargs):
            role = kwargs.get('role', None)
            routine.name += f'_{role}'

    # Apply re-naming transformation and check result
    scheduler.process(transformation=AppendRole())
    assert scheduler.item_map['compute_l1_mod#compute_l1'].routine.name == 'compute_l1_driver'
    assert scheduler.item_map['compute_l2_mod#compute_l2'].routine.name == 'compute_l2_kernel'
    assert scheduler.item_map['#another_l1'].routine.name == 'another_l1_driver'
    assert scheduler.item_map['#another_l2'].routine.name == 'another_l2_kernel'


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
def test_scheduler_graph_multiple_combined(here, config, frontend):
    """
    Create a single task graph spanning two projects

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """
    projA = here/'sources/projA'
    projB = here/'sources/projB'

    scheduler = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=config,
        seed_routines=['driverB'], frontend=frontend
    )

    expected_items = [
        'driverB_mod#driverB', 'kernelB_mod#kernelB',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2',
        'ext_driver_mod#ext_driver', 'ext_kernel_mod#ext_kernel'
    ]
    expected_dependencies = [
        ('driverB_mod#driverB', 'kernelB_mod#kernelB'),
        ('kernelB_mod#kernelB', 'compute_l1_mod#compute_l1'),
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
        ('kernelB_mod#kernelB', 'ext_driver_mod#ext_driver'),
        ('ext_driver_mod#ext_driver', 'ext_kernel_mod#ext_kernel'),
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
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
def test_scheduler_graph_multiple_separate(here, config, frontend):
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

    schedulerA = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=configA,
        seed_routines=['driverB'], frontend=frontend
    )

    expected_itemsA = [
        'driverB_mod#driverB', 'kernelB_mod#kernelB',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2',
    ]
    expected_dependenciesA = [
        ('driverB_mod#driverB', 'kernelB_mod#kernelB'),
        ('kernelB_mod#kernelB', 'compute_l1_mod#compute_l1'),
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
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
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()

    # Test second scheduler instance that holds the receiver items
    configB = config.copy()
    configB['routine'] = [
        {
            'name': 'ext_driver',
            'role': 'kernel',
        },
    ]

    schedulerB = Scheduler(
        paths=projB, config=configB, seed_routines=['ext_driver'],
        frontend=frontend
    )

    # TODO: Technically we should check that the role=kernel has been honoured in B
    assert 'ext_driver_mod#ext_driver' in schedulerB.items
    assert 'ext_kernel_mod#ext_kernel' in schedulerB.items
    assert ('ext_driver_mod#ext_driver', 'ext_kernel_mod#ext_kernel') in schedulerB.dependencies

    # Check that the call from kernelB to ext_driver has been enriched with IPA meta-info
    call = FindNodes(CallStatement).visit(schedulerA.item_map['kernelb_mod#kernelb'].routine.body)[1]
    assert isinstance(call.routine, Subroutine)
    assert fexprgen(call.routine.arguments) == '(vector(:), matrix(:, :))'

    # Test callgraph visualisation
    cg_path = here/'callgraph_multiple_separate_B'
    schedulerB.callgraph(cg_path)

    vgraphB = VisGraphWrapper(cg_path)
    assert 'EXT_DRIVER_MOD#EXT_DRIVER' in vgraphB.nodes
    assert 'EXT_KERNEL_MOD#EXT_KERNEL' in vgraphB.nodes
    assert ('EXT_DRIVER_MOD#EXT_DRIVER', 'EXT_KERNEL_MOD#EXT_KERNEL') in vgraphB.edges

    cg_path.unlink()
    if cg_path.with_suffix('.pdf').exists():
        cg_path.with_suffix('.pdf').unlink()


def test_scheduler_module_dependency(here, config, frontend):
    """
    Ensure dependency chasing is done correctly, even with surboutines
    that do not match module names.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 | --> routine_one -> routine_two
    """
    projA = here/'sources/projA'
    projC = here/'sources/projC'

    scheduler = Scheduler(
        paths=[projA, projC], includes=projA/'include', config=config,
        seed_routines=['driverC'], frontend=frontend
    )

    expected_items = [
        'driverC_mod#driverC', 'kernelC_mod#kernelC',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2',
        'proj_c_util_mod#routine_one', 'proj_c_util_mod#routine_two'
    ]
    expected_dependencies = [
        ('driverC_mod#driverC', 'kernelC_mod#kernelC'),
        ('kernelC_mod#kernelC', 'compute_l1_mod#compute_l1'),
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
        ('kernelC_mod#kernelC', 'proj_c_util_mod#routine_one'),
        ('proj_c_util_mod#routine_one', 'proj_c_util_mod#routine_two'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    # Ensure that we got the right routines from the module
    assert scheduler.item_map['proj_c_util_mod#routine_one'].routine.name == 'routine_one'
    assert scheduler.item_map['proj_c_util_mod#routine_two'].routine.name == 'routine_two'


def test_scheduler_module_dependencies_unqualified(here, config, frontend):
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

    scheduler = Scheduler(
        paths=[projA, projC], includes=projA/'include', config=config,
        seed_routines=['driverD'], frontend=frontend
    )

    expected_items = [
        'driverD_mod#driverD', 'kernelD_mod#kernelD',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2',
        'proj_c_util_mod#routine_one', 'proj_c_util_mod#routine_two'
    ]
    expected_dependencies = [
        ('driverD_mod#driverD', 'kernelD_mod#kernelD'),
        ('kernelD_mod#kernelD', 'compute_l1_mod#compute_l1'),
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
        ('kernelD_mod#kernelD', 'proj_c_util_mod#routine_one'),
        ('proj_c_util_mod#routine_one', 'proj_c_util_mod#routine_two'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    # Ensure that we got the right routines from the module
    assert scheduler.item_map['proj_c_util_mod#routine_one'].routine.name == 'routine_one'
    assert scheduler.item_map['proj_c_util_mod#routine_two'].routine.name == 'routine_two'


def test_scheduler_missing_files(here, config, frontend):
    """
    Ensure that ``strict=True`` triggers failure if source paths are
    missing and that ``strict=Files`` goes through gracefully.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 < cannot find path >
    """
    projA = here/'sources/projA'

    config['default']['strict'] = True
    with pytest.raises(FileNotFoundError):
        scheduler = Scheduler(
            paths=[projA], includes=projA/'include', config=config,
            seed_routines=['driverC'], frontend=frontend
        )

    config['default']['strict'] = False
    scheduler = Scheduler(
        paths=[projA], includes=projA/'include', config=config,
        seed_routines=['driverC'], frontend=frontend
    )

    expected_items = [
        'driverC_mod#driverC', 'kernelC_mod#kernelC',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'
    ]
    expected_dependencies = [
        ('driverC_mod#driverC', 'kernelC_mod#kernelC'),
        ('kernelC_mod#kernelC', 'compute_l1_mod#compute_l1'),
        ('compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'),
    ]
    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    # Ensure that the missing items are not in the graph
    assert 'proj_c_util_mod#routine_one' not in scheduler.items
    assert 'proj_c_util_mod#routine_two' not in scheduler.items


def test_scheduler_dependencies_ignore(here, frontend):
    """
    Test multi-lib transformation by applying the :any:`DependencyTransformation`
    over two distinct projects with two distinct invocations.

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """
    projA = here/'sources/projA'
    projB = here/'sources/projB'

    configA = SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routine': [
            {'name': 'driverB', 'role': 'driver'},
            {'name': 'kernelB', 'ignore': ['ext_driver']},
        ]
    })

    configB = SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routine': [
            {'name': 'ext_driver', 'role': 'kernel'}
        ]
    })

    schedulerA = Scheduler(paths=[projA, projB], includes=projA/'include', config=configA, frontend=frontend)

    schedulerB = Scheduler(paths=projB, includes=projB/'include', config=configB, frontend=frontend)

    assert all(n in schedulerA.items for n in [
        'driverB_mod#driverB', 'kernelB_mod#kernelB',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2'
    ])
    assert 'ext_driver_mod#ext_driver' not in schedulerA.items
    assert 'ext_kernel_mod#ext_kernel' not in schedulerA.items

    assert all(n in schedulerB.items for n in ['ext_driver_mod#ext_driver', 'ext_kernel_mod#ext_kernel'])

    # Apply dependency injection transformation and ensure only the root driver is not transformed
    dependency = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    schedulerA.process(transformation=dependency)

    assert schedulerA.items[0].source.all_subroutines[0].name == 'driverB'
    assert schedulerA.items[1].source.all_subroutines[0].name == 'kernelB_test'
    assert schedulerA.items[2].source.all_subroutines[0].name == 'compute_l1_test'
    assert schedulerA.items[3].source.all_subroutines[0].name == 'compute_l2_test'

    # For the second target lib, we want the driver to be converted
    schedulerB.process(transformation=dependency)

    # Repeat processing to ensure DependencyTransform is idempotent
    schedulerB.process(transformation=dependency)

    assert schedulerB.items[0].source.all_subroutines[0].name == 'ext_driver_test'
    assert schedulerB.items[1].source.all_subroutines[0].name == 'ext_kernel_test'


def test_scheduler_cmake_planner(here, frontend):
    """
    Test the plan generation feature over a call hierarchy spanning two
    distinctive projects.

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """

    sourcedir = here/'sources'
    proj_a = sourcedir/'projA'
    proj_b = sourcedir/'projB'

    config = SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routine': [
            {'name': 'driverB', 'role': 'driver'},
            {'name': 'kernelB', 'ignore': ['ext_driver']},
        ]
    })

    # Populate the scheduler
    # (this is the same as SchedulerA in test_scheduler_dependencies_ignore, so no need to
    # check scheduler set-up itself)
    scheduler = Scheduler(
        paths=[proj_a, proj_b], includes=proj_a/'include',
        config=config, frontend=frontend
    )

    # Apply the transformation
    builddir = here/'scheduler_cmake_planner_dummy_dir'
    builddir.mkdir(exist_ok=True)
    planfile = builddir/'loki_plan.cmake'

    planner = CMakePlanner(rootpath=sourcedir, mode='foobar', build=builddir)
    scheduler.process(transformation=planner)

    # Validate the generated lists
    expected_files = {
        proj_a/'module/driverB_mod.f90', proj_a/'module/kernelB_mod.F90',
        proj_a/'module/compute_l1_mod.f90', proj_a/'module/compute_l2_mod.f90'
    }

    assert set(planner.sources_to_remove) == {f.relative_to(sourcedir) for f in expected_files}
    assert set(planner.sources_to_append) == {
        (builddir/f.stem).with_suffix('.foobar.F90') for f in expected_files
    }
    assert set(planner.sources_to_transform) == {f.relative_to(sourcedir) for f in expected_files}

    # Write the plan file
    planner.write_planfile(planfile)

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)

    loki_plan = planfile.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    expected_files = {
        'driverB_mod', 'kernelB_mod',
        'compute_l1_mod', 'compute_l2_mod'
    }

    assert 'LOKI_SOURCES_TO_TRANSFORM' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == expected_files

    assert 'LOKI_SOURCES_TO_REMOVE' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == expected_files

    assert 'LOKI_SOURCES_TO_APPEND' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {
        f'{name}.foobar' for name in expected_files
    }

    planfile.unlink()
    builddir.rmdir()


def test_scheduler_item(here):
    """
    Test the basic regex frontend nodes in :any:`Item` objects for fast dependency detection.
    """
    filepath = here/'sources/sourcefile_item.f90'
    sourcefile = Sourcefile.from_file(filepath, frontend=REGEX)

    available_names = [f'#{r.name}' for r in sourcefile.subroutines]
    available_names += [f'{m.name}#{r.name}' for m in sourcefile.modules for r in m.subroutines]

    item_a = SubroutineItem(name='#routine_a', source=sourcefile)
    assert item_a.calls == ('routine_b',)
    assert not item_a.members
    assert item_a.children == ('routine_b',)
    assert item_a.qualify_names(item_a.children, available_names) == ('#routine_b',)
    assert item_a.targets == item_a.children

    item_module = SubroutineItem(name='some_module#module_routine', source=sourcefile)
    assert item_module.calls == ('routine_b',)
    assert not item_module.members
    assert item_module.children == ('routine_b',)
    assert item_module.qualify_names(item_module.children, available_names) == ('#routine_b',)
    assert item_module.targets == item_module.children

    item_b = SubroutineItem(name='#routine_b', source=sourcefile)
    assert item_b.calls == ('contained_c', 'routine_a')
    assert 'contained_c' in item_b.members
    assert 'contained_d' in item_b.members
    assert item_b.children == ('routine_a',)
    assert item_b.qualify_names(item_b.children, available_names) == ('#routine_a',)
    assert item_b.targets == ('contained_c', 'routine_a')

    item_b = SubroutineItem(name='#routine_b', source=sourcefile, config={'disable': ['routine_a']})
    assert item_b.calls == ('contained_c', 'routine_a')
    assert 'contained_c' in item_b.members
    assert 'contained_d' in item_b.members
    assert not item_b.children
    assert not item_b.qualify_names(item_b.children, available_names)
    assert item_b.targets == ('contained_c',)

    item_b = SubroutineItem(name='#routine_b', source=sourcefile, config={'ignore': ['routine_a']})
    assert item_b.calls == ('contained_c', 'routine_a')
    assert 'contained_c' in item_b.members
    assert 'contained_d' in item_b.members
    assert item_b.children == ('routine_a',)
    assert item_b.qualify_names(item_b.children, available_names) == ('#routine_a',)
    assert item_b.targets == ('contained_c',)

    item_b = SubroutineItem(name='#routine_b', source=sourcefile, config={'block': ['routine_a']})
    assert item_b.calls == ('contained_c', 'routine_a')
    assert 'contained_c' in item_b.members
    assert 'contained_d' in item_b.members
    assert item_b.children == ('routine_a',)
    assert item_b.qualify_names(item_b.children, available_names) == ('#routine_a',)
    assert item_b.targets == ('contained_c',)


def test_scheduler_item_children(here):
    """
    Make sure children are correct and unique for items
    """
    config = SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routine': [
            {'name': 'driver', 'role': 'driver'},
            {'name': 'another_driver', 'role': 'driver'}
        ]
    })

    proj_hoist = here/'sources/projHoist'

    scheduler = Scheduler(paths=proj_hoist, config=config)

    assert scheduler.item_map['transformation_module_hoist#driver'].children == (
        'kernel1', 'kernel2'
    )
    assert scheduler.item_map['transformation_module_hoist#another_driver'].children == (
        'kernel1',
    )
    assert not scheduler.item_map['subroutines_mod#kernel1'].children
    assert scheduler.item_map['subroutines_mod#kernel2'].children == (
        'device1', 'device2'
    )
    assert scheduler.item_map['subroutines_mod#device1'].children == (
        'device2',
    )
    assert not scheduler.item_map['subroutines_mod#device2'].children


@pytest.fixture(name='loki_69_dir')
def fixture_loki_69_dir(here):
    """
    Fixture to write test file for LOKI-69 test.
    """
    fcode = """
subroutine random_call_0(v_out,v_in,v_inout)
implicit none

    real(kind=jprb),intent(in)  :: v_in
    real(kind=jprb),intent(out)  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout


end subroutine random_call_0

!subroutine random_call_1(v_out,v_in,v_inout)
!implicit none
!
!  real(kind=jprb),intent(in)  :: v_in
!  real(kind=jprb),intent(out)  :: v_out
!  real(kind=jprb),intent(inout)  :: v_inout
!
!
!end subroutine random_call_1

subroutine random_call_2(v_out,v_in,v_inout)
implicit none

    real(kind=jprb),intent(in)  :: v_in
    real(kind=jprb),intent(out)  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout


end subroutine random_call_2

subroutine test(v_out,v_in,v_inout,some_logical)
implicit none

    real(kind=jprb),intent(in   )  :: v_in
    real(kind=jprb),intent(out  )  :: v_out
    real(kind=jprb),intent(inout)  :: v_inout

    logical,intent(in)             :: some_logical

    v_inout = 0._jprb
    if(some_logical)then
        call random_call_0(v_out,v_in,v_inout)
    endif

    if(some_logical) call random_call_2

end subroutine test
    """.strip()

    dirname = here/'loki69'
    dirname.mkdir(exist_ok=True)
    filename = dirname/'test.F90'
    filename.write_text(fcode)
    yield dirname
    try:
        filename.unlink()
        dirname.rmdir()
    except FileNotFoundError:
        pass


def test_scheduler_loki_69(loki_69_dir):
    """
    Test compliance of scheduler with edge cases reported in LOKI-69.
    """
    config = {
        'default': {
            'expand': True,
            'strict': True,
        },
    }

    scheduler = Scheduler(paths=loki_69_dir, seed_routines=['test'], config=config)
    assert sorted(scheduler.obj_map.keys()) == ['#random_call_0', '#random_call_2', '#test']
    assert '#random_call_1' not in scheduler.obj_map

    children_map = {
        '#test': ('random_call_0', 'random_call_2'),
        '#random_call_0': (),
        '#random_call_2': ()
    }
    assert len(scheduler.items) == len(children_map)
    assert all(item.children == children_map[item.name] for item in scheduler.items)


def test_scheduler_scopes(here, config, frontend):
    """
    Test discovery with import renames and duplicate names in separate scopes

      driver ----> kernel1_mod#kernel ----> kernel1_impl#kernel_impl
        |
        +--------> kernel2_mod#kernel ----> kernel2_impl#kernel_impl
    """
    proj = here/'sources/projScopes'

    scheduler = Scheduler(paths=proj, seed_routines=['driver'], config=config, frontend=frontend)

    expected_items = {
        '#driver', 'kernel1_mod#kernel', 'kernel1_impl#kernel_impl',
        'kernel2_mod#kernel', 'kernel2_impl#kernel_impl'
    }
    expected_dependencies = {
        ('#driver', 'kernel1_mod#kernel'), ('#driver', 'kernel2_mod#kernel'),
        ('kernel1_mod#kernel', 'kernel1_impl#kernel_impl'),
        ('kernel2_mod#kernel', 'kernel2_impl#kernel_impl'),
    }

    assert expected_items == {n.name for n in scheduler.items}
    assert expected_dependencies == {(e[0].name, e[1].name) for e in scheduler.dependencies}

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_scopes'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


def test_scheduler_typebound_item(here):
    """
    Test the basic regex frontend nodes in :any:`Item` objects for fast dependency detection
    for type-bound procedures.
    """
    filepath = here/'sources/projTypeBound/typebound_item.F90'
    headerpath = here/'sources/projTypeBound/typebound_header.F90'
    otherpath = here/'sources/projTypeBound/typebound_other.F90'
    source = Sourcefile.from_file(filepath, frontend=REGEX)
    header = Sourcefile.from_file(headerpath, frontend=REGEX)
    other = Sourcefile.from_file(otherpath, frontend=REGEX)

    available_names = []
    for s in [source, header, other]:
        available_names += [f'#{r.name.lower()}' for r in s.subroutines]
        available_names += [f'{m.name.lower()}#{r.name.lower()}' for m in s.modules for r in m.subroutines]
        available_names += [f'{m.name.lower()}#{t.lower()}' for m in s.modules for t in m.typedefs]

    driver = SubroutineItem(name='#driver', source=source)

    # Check that calls (= dependencies) are correctly identified
    assert driver.calls == (
        'some_type%other_routine', 'some_type%some_routine',
        'header_type%member_routine', 'header_type%routine',
        'header_type%routine', 'other%member', 'other%var%member_routine'
    )

    # Check that imports are correctly identified
    assert [i.module for i in driver.imports] == ['typebound_item', 'typebound_header', 'typebound_other']
    assert driver.unqualified_imports == ('typebound_item', 'typebound_header')
    assert driver.qualified_imports == {'other': 'typebound_other#other_type'}

    # Check that calls are propagated as children
    assert set(driver.children) == set(driver.calls)

    # Check that fully-qualified names are correctly picked from the available names
    assert driver.qualify_names(driver.children, available_names) == (
        'typebound_item#some_type%other_routine', 'typebound_item#some_type%some_routine',
        'typebound_header#header_type%member_routine', 'typebound_header#header_type%routine',
        'typebound_other#other_type%member', 'typebound_other#other_type%var%member_routine'
    )

    other_routine = SubroutineItem(name='typebound_item#other_routine', source=source)
    assert isinstance(other_routine, SubroutineItem)
    assert isinstance(other_routine.routine, Subroutine)
    assert other_routine.calls == ('abor1', 'some_type%routine1', 'some_type%routine2')
    assert other_routine.children == other_routine.calls
    assert other_routine.qualify_names(other_routine.children, available_names) == (
        'typebound_header#abor1', 'typebound_item#some_type%routine1', 'typebound_item#some_type%routine2'
    )

    # Local names (i.e. within the same scope) can be qualified in any case, while non-local names
    # can potentially exist globally or come from one of the unqualified imports, for which we return
    # a tuple of candidates
    assert other_routine.qualify_names(other_routine.children, available_names=[]) == (
        ('#abor1', 'typebound_header#abor1'), 'typebound_item#some_type%routine1', 'typebound_item#some_type%routine2'
    )

    routine = SubroutineItem(
        name='typebound_item#routine', source=source, config={'disable': ['some_type%some_routine']}
    )
    assert isinstance(routine, SubroutineItem)
    assert isinstance(routine.routine, Subroutine)
    assert routine.calls == ('some_type%some_routine',)
    # No children due to `disable` config
    assert not routine.children
    assert not routine.qualify_names(routine.children, available_names)

    routine1 = SubroutineItem(name='typebound_item#routine1', source=source, config={'disable': ['module_routine']})
    assert isinstance(routine1, SubroutineItem)
    assert isinstance(routine1.routine, Subroutine)
    assert routine1.calls == ('module_routine',)
    assert not routine1.children
    assert not routine1.qualify_names(routine1.children, available_names)

    some_type_some_routine = ProcedureBindingItem(
        name='typebound_item#some_type%some_routine', source=source,
        config={'ignore': ['some_routine']}
    )
    assert isinstance(some_type_some_routine, ProcedureBindingItem)
    assert some_type_some_routine.routine is None
    assert some_type_some_routine.calls == ('some_routine',)
    # Ignored routines still show up as children
    assert some_type_some_routine.children == some_type_some_routine.calls
    assert some_type_some_routine.qualify_names(some_type_some_routine.children, available_names) == (
        'typebound_item#some_routine',
    )

    some_type_routine = ProcedureBindingItem(
        name='typebound_item#some_type%routine', source=source,
        config={'block': ['module_routine']}
    )
    assert isinstance(some_type_routine, ProcedureBindingItem)
    assert some_type_routine.routine is None
    assert some_type_routine.calls == ('module_routine',)
    # Blocked routines still show up as children
    assert some_type_routine.children == some_type_routine.calls
    assert some_type_routine.qualify_names(some_type_routine.children, available_names) == (
        'typebound_item#module_routine',
    )
    assert some_type_routine.qualify_names(some_type_routine.children) == (
        'typebound_item#module_routine',
    )

    other_type_member = ProcedureBindingItem(name='typebound_other#other_type%member', source=other)
    assert isinstance(other_type_member, ProcedureBindingItem)
    assert other_type_member.routine is None
    assert other_type_member.qualified_imports == {'header': 'typebound_header#header_type'}
    assert other_type_member.calls == ('other_member',)
    assert other_type_member.children == other_type_member.calls
    assert other_type_member.qualify_names(other_type_member.children, available_names) == (
        'typebound_other#other_member',
    )

    other_type_var_member_routine = ProcedureBindingItem(
        name='typebound_other#other_type%var%member_routine', source=other
    )
    assert isinstance(other_type_var_member_routine, ProcedureBindingItem)
    assert other_type_var_member_routine.routine is None
    assert other_type_var_member_routine.qualified_imports == {'header': 'typebound_header#header_type'}
    assert other_type_var_member_routine.calls == ('header%member_routine',)
    assert other_type_var_member_routine.children == other_type_var_member_routine.calls
    # typebound names can also be fully qualified if they are declared in the same scope
    assert other_type_var_member_routine.qualify_names(other_type_var_member_routine.calls, available_names) == (
        'typebound_header#header_type%member_routine',
    )
    assert other_type_var_member_routine.qualify_names(other_type_var_member_routine.calls) == (
        'typebound_header#header_type%member_routine',
    )

    header_type_member_routine = ProcedureBindingItem(
        name='typebound_header#header_type%member_routine', source=header
    )
    assert isinstance(header_type_member_routine, ProcedureBindingItem)
    assert header_type_member_routine.routine is None
    assert header_type_member_routine.calls == ('header_member_routine',)
    assert header_type_member_routine.children == header_type_member_routine.calls
    assert header_type_member_routine.qualify_names(header_type_member_routine.children, available_names) == (
        'typebound_header#header_member_routine',
    )


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
def test_scheduler_typebound(here, config, frontend):
    """
    Test correct dependency chasing for typebound procedure calls.

    projTypeBound: driver -> some_type%other_routine -> other_routine -> some_type%routine1 -> routine1
                 | | | | | |                                          |                                |
                 | | | | | |       +- routine <- some_type%routine2 <-+                                +---------+
                 | | | | | |       |                                                                             |
                 | | | | | +--> some_type%some_routine -> some_routine -> some_type%routine -> module_routine  <-+
                 | | | +------> header_type%member_routine -> header_member_routine
                 | | +--------> header_type%routine -> header_type%routine_real -> header_routine_real
                 | |                           |
                 | |                           +---> header_type%routine_integer -> routine_integer
                 | +---------->other_type%member -> other_member -> header_member_routine   <--+
                 |                                                                             |
                 +------------>other_type%var%%member_routine -> header_type%member_routine  --+
    """
    proj = here/'sources/projTypeBound'

    scheduler = Scheduler(
        paths=proj, seed_routines=['driver'], config=config,
        full_parse=False, frontend=frontend
    )

    expected_items = {
        '#driver',
        'typebound_item#some_type%some_routine', 'typebound_item#some_type%other_routine',
        'typebound_item#other_routine', 'typebound_item#some_type%routine1',
        'typebound_item#routine1', 'typebound_item#some_type%routine2', 'typebound_item#routine',
        'typebound_header#header_type%member_routine',
        'typebound_header#header_member_routine',
        'typebound_item#some_type%routine', 'typebound_item#module_routine',
        'typebound_item#some_routine', 'typebound_header#header_type%routine',
        'typebound_header#header_type%routine_real', 'typebound_header#header_routine_real',
        'typebound_header#header_type%routine_integer', 'typebound_header#routine_integer',
        'typebound_header#abor1', 'typebound_other#other_type%member',
        'typebound_other#other_member', 'typebound_other#other_type%var%member_routine'
    }
    expected_dependencies = {
        ('#driver', 'typebound_item#some_type%other_routine'),
        ('typebound_item#some_type%other_routine', 'typebound_item#other_routine'),
        ('typebound_item#other_routine', 'typebound_item#some_type%routine1'),
        ('typebound_item#some_type%routine1', 'typebound_item#routine1'),
        ('typebound_item#routine1', 'typebound_item#module_routine'),
        ('typebound_item#other_routine', 'typebound_item#some_type%routine2'),
        ('typebound_item#some_type%routine2', 'typebound_item#routine'),
        ('typebound_item#routine', 'typebound_item#some_type%some_routine'),
        ('#driver', 'typebound_item#some_type%some_routine'),
        ('typebound_item#some_type%some_routine', 'typebound_item#some_routine'),
        ('typebound_item#some_routine', 'typebound_item#some_type%routine'),
        ('typebound_item#some_type%routine', 'typebound_item#module_routine'),
        ('#driver', 'typebound_header#header_type%member_routine'),
        ('typebound_header#header_type%member_routine', 'typebound_header#header_member_routine'),
        ('#driver', 'typebound_other#other_type%member'),
        ('typebound_other#other_type%member', 'typebound_other#other_member'),
        ('typebound_other#other_member', 'typebound_header#header_member_routine'),
        ('typebound_other#other_member', 'typebound_other#other_type%var%member_routine'),
        ('#driver', 'typebound_other#other_type%var%member_routine'),
        ('typebound_other#other_type%var%member_routine', 'typebound_header#header_type%member_routine'),
        ('typebound_item#other_routine', 'typebound_header#abor1'),
        ('#driver', 'typebound_header#header_type%routine'),
        ('typebound_header#header_type%routine', 'typebound_header#header_type%routine_real'),
        ('typebound_header#header_type%routine_real', 'typebound_header#header_routine_real'),
        ('typebound_header#header_type%routine', 'typebound_header#header_type%routine_integer'),
        ('typebound_header#header_type%routine_integer', 'typebound_header#routine_integer')
    }
    assert expected_items == {n.name for n in scheduler.items}
    assert expected_dependencies == {(e[0].name, e[1].name) for e in scheduler.dependencies}

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_typebound'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(importlib.util.find_spec('graphviz') is None, reason='Graphviz is not installed')
def test_scheduler_typebound_ignore(here, config, frontend):
    """
    Test correct dependency chasing for typebound procedure calls with ignore working for
    typebound procedures correctly.

    projTypeBound: driver -> some_type%other_routine -> other_routine -> some_type%routine1 -> routine1
                   | | | | |                                          |                                |
                   | | | | |       +- routine <- some_type%routine2 <-+                                +---------+
                   | | | | |       |                                                                             |
                   | | | | +--> some_type%some_routine -> some_routine -> some_type%routine -> module_routine  <-+
                   | | +------> header_type%member_routine -> header_member_routine
                   | +--------> header_type%routine -> header_type%routine_real -> header_routine_real
                   |                           |
                   |                           +---> header_type%routine_integer -> routine_integer
                   +---------->other_type%member -> other_member -> header_member_routine
    """
    proj = here/'sources/projTypeBound'

    my_config = config.copy()
    my_config['default']['disable'] += ['some_type%some_routine', 'header_member_routine']
    my_config['routine'] = [
        {
            'name': 'other_member',
            'disable': my_config['default']['disable'] + ['member_routine']
        }
    ]

    scheduler = Scheduler(
        paths=proj, seed_routines=['driver'], config=my_config,
        full_parse=False, frontend=frontend
    )

    expected_items = {
        '#driver',
        'typebound_item#some_type%other_routine', 'typebound_item#other_routine',
        'typebound_item#some_type%routine1', 'typebound_item#routine1',
        'typebound_item#some_type%routine2', 'typebound_item#routine',
        'typebound_header#header_type%member_routine',
        'typebound_item#module_routine',
        'typebound_header#header_type%routine',
        'typebound_header#header_type%routine_real', 'typebound_header#header_routine_real',
        'typebound_header#header_type%routine_integer', 'typebound_header#routine_integer',
        'typebound_header#abor1', 'typebound_other#other_type%member',
        'typebound_other#other_member', 'typebound_other#other_type%var%member_routine'
    }
    expected_dependencies = {
        ('#driver', 'typebound_item#some_type%other_routine'),
        ('typebound_item#some_type%other_routine', 'typebound_item#other_routine'),
        ('typebound_item#other_routine', 'typebound_item#some_type%routine1'),
        ('typebound_item#some_type%routine1', 'typebound_item#routine1'),
        ('typebound_item#routine1', 'typebound_item#module_routine'),
        ('typebound_item#other_routine', 'typebound_item#some_type%routine2'),
        ('typebound_item#some_type%routine2', 'typebound_item#routine'),
        ('#driver', 'typebound_header#header_type%member_routine'),
        ('#driver', 'typebound_other#other_type%member'),
        ('typebound_other#other_type%member', 'typebound_other#other_member'),
        ('typebound_other#other_member', 'typebound_other#other_type%var%member_routine'),
        ('#driver', 'typebound_other#other_type%var%member_routine'),
        ('typebound_other#other_type%var%member_routine', 'typebound_header#header_type%member_routine'),
        ('typebound_item#other_routine', 'typebound_header#abor1'),
        ('#driver', 'typebound_header#header_type%routine'),
        ('typebound_header#header_type%routine', 'typebound_header#header_type%routine_real'),
        ('typebound_header#header_type%routine_real', 'typebound_header#header_routine_real'),
        ('typebound_header#header_type%routine', 'typebound_header#header_type%routine_integer'),
        ('typebound_header#header_type%routine_integer', 'typebound_header#routine_integer')
    }
    assert expected_items == {n.name for n in scheduler.items}
    assert expected_dependencies == {(e[0].name, e[1].name) for e in scheduler.dependencies}

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_typebound'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert all(n.upper() in vgraph.nodes for n in expected_items)
    assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.parametrize('use_file_graph,reverse', [
    (False, False), (False, True), (True, False), (True, True)
])
def test_scheduler_traversal_order(here, config, frontend, use_file_graph, reverse):
    """
    Test correct traversal order for scheduler processing

    """
    proj = here/'sources/projHoist'

    scheduler = Scheduler(
        paths=proj, seed_routines=['driver'], config=config,
        full_parse=True, frontend=frontend
    )

    if use_file_graph:
        expected = [
            'transformation_module_hoist#driver', 'subroutines_mod#kernel1'
        ]
    else:
        expected = [
            'transformation_module_hoist#driver', 'subroutines_mod#kernel1', 'subroutines_mod#kernel2',
            'subroutines_mod#device1', 'subroutines_mod#device2'
        ]

    class LoggingTransformation(Transformation):

        def __init__(self):
            self.record = []

        def transform_file(self, sourcefile, **kwargs):
            item = kwargs['item']
            self.record += [item.name]

    transformation = LoggingTransformation()
    scheduler.process(transformation=transformation, reverse=reverse, use_file_graph=use_file_graph)

    if reverse:
        assert transformation.record == expected[::-1]
    else:
        assert transformation.record == expected


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_nested_type_enrichment(frontend, config):
    """
    Make sure that enrichment works correctly for nested types across
    multiple source files
    """
    fcode1 = """
module typebound_procedure_calls_mod
    implicit none

    type my_type
        integer :: val
    contains
        procedure :: reset
        procedure :: add => add_my_type
    end type my_type

    type other_type
        type(my_type) :: arr(3)
    contains
        procedure :: add => add_other_type
        procedure :: total_sum
    end type other_type

contains

    subroutine reset(this)
        class(my_type), intent(inout) :: this
        this%val = 0
    end subroutine reset

    subroutine add_my_type(this, val)
        class(my_type), intent(inout) :: this
        integer, intent(in) :: val
        this%val = this%val + val
    end subroutine add_my_type

    subroutine add_other_type(this, other)
        class(other_type) :: this
        type(other_type) :: other
        integer :: i
        do i=1,3
            call this%arr(i)%add(other%arr(i)%val)
        end do
    end subroutine add_other_type

    function total_sum(this) result(result)
        class(other_type), intent(in) :: this
        integer :: result
        integer :: i
        result = 0
        do i=1,3
            result = result + this%arr(i)%val
        end do
    end function total_sum

end module typebound_procedure_calls_mod
    """.strip()

    fcode2 = """
module other_typebound_procedure_calls_mod
    use typebound_procedure_calls_mod, only: other_type
    implicit none

    type third_type
        type(other_type) :: stuff(2)
    contains
        procedure :: init
        procedure :: print => print_content
    end type third_type

contains

    subroutine init(this)
        class(third_type), intent(inout) :: this
        integer :: i, j
        do i=1,2
            do j=1,3
                call this%stuff(i)%arr(j)%reset()
                call this%stuff(i)%arr(j)%add(i+j)
            end do
        end do
    end subroutine init

    subroutine print_content(this)
        class(third_type), intent(inout) :: this
        call this%stuff(1)%add(this%stuff(2))
        print *, this%stuff(1)%total_sum()
    end subroutine print_content
end module other_typebound_procedure_calls_mod
    """.strip()

    fcode3 = """
subroutine driver
    use other_typebound_procedure_calls_mod, only: third_type
    implicit none
    type(third_type) :: data
    integer :: mysum

    call data%init()
    call data%stuff(1)%arr(1)%add(1)
    mysum = data%stuff(1)%total_sum() + data%stuff(2)%total_sum()
    call data%print
end subroutine driver
    """.strip()

    workdir = gettempdir()/'test_scheduler_nested_type_enrichment'
    workdir.mkdir(exist_ok=True)
    (workdir/'typebound_procedure_calls_mod.F90').write_text(fcode1)
    (workdir/'other_typebound_procedure_calls_mod.F90').write_text(fcode2)
    (workdir/'driver.F90').write_text(fcode3)

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['driver'], frontend=frontend)

    driver = scheduler['#driver'].source['driver']
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 3
    for call in calls:
        assert isinstance(call.name, ProcedureSymbol)
        assert isinstance(call.name.type.dtype, ProcedureType)
        assert call.name.parent
        assert isinstance(call.name.parent.type.dtype, DerivedType)

    assert isinstance(calls[0].name.parent, Scalar)
    assert calls[0].name.parent.type.dtype.name == 'third_type'
    assert isinstance(calls[0].name.parent.type.dtype.typedef, TypeDef)

    assert isinstance(calls[1].name.parent, Array)
    assert calls[1].name.parent.type.dtype.name == 'my_type'
    assert isinstance(calls[1].name.parent.type.dtype.typedef, TypeDef)

    assert isinstance(calls[1].name.parent.parent, Array)
    assert isinstance(calls[1].name.parent.parent.type.dtype, DerivedType)
    assert calls[1].name.parent.parent.type.dtype.name == 'other_type'
    assert isinstance(calls[1].name.parent.parent.type.dtype.typedef, TypeDef)

    assert isinstance(calls[1].name.parent.parent.parent, Scalar)
    assert isinstance(calls[1].name.parent.parent.parent.type.dtype, DerivedType)
    assert calls[1].name.parent.parent.parent.type.dtype.name == 'third_type'
    assert isinstance(calls[1].name.parent.parent.parent.type.dtype.typedef, TypeDef)

    inline_calls = FindInlineCalls().visit(driver.body)
    assert len(inline_calls) == 2
    for call in inline_calls:
        assert isinstance(call.function, ProcedureSymbol)
        assert isinstance(call.function.type.dtype, ProcedureType)

        assert call.function.parent
        assert isinstance(call.function.parent, Array)
        assert isinstance(call.function.parent.type.dtype, DerivedType)
        assert call.function.parent.type.dtype.name == 'other_type'
        assert isinstance(call.function.parent.type.dtype.typedef, TypeDef)

        assert call.function.parent.parent
        assert isinstance(call.function.parent.parent, Scalar)
        assert isinstance(call.function.parent.parent.type.dtype, DerivedType)
        assert call.function.parent.parent.type.dtype.name == 'third_type'
        assert isinstance(call.function.parent.parent.type.dtype.typedef, TypeDef)

    rmtree(workdir)


def test_scheduler_qualify_names():
    """
    Make sure qualified names are all lower case
    """
    fcode = """
module some_mod
    use other_mod
    use MORE_MOD
    implicit none

    interface override
      module procedure some_random_function
    end interface ovveride

contains

    function some_random_function
    end function

    subroutine DRIVER
        use YET_another_mod
        call routine
    end subroutine DRIVER
end module some_mod
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=REGEX)
    item = SubroutineItem(name='some_mod#driver', source=source)
    assert item.qualify_names(item.children) == (
        ('#routine', 'yet_another_mod#routine', 'other_mod#routine', 'more_mod#routine'),
    )


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_inline_call(here, config, frontend):
    """
    Test that inline function calls declared via an explicit interface are added as dependencies.
    """

    my_config = config.copy()
    my_config['routine'] = [
        {
            'name': 'driver',
            'role': 'driver'
        }
    ]

    scheduler = Scheduler(paths=here/'sources/projInlineCalls', config=my_config, frontend=frontend)

    expected_items = {'#driver', '#double_real'}
    expected_dependencies = {('#driver', '#double_real')}

    assert expected_items == {i.name for i in scheduler.items}
    assert expected_dependencies == {(d[0].name, d[1].name) for d in scheduler.dependencies}

    for i in scheduler.items:
        if i.name == '#double_real':
            assert isinstance(i, SubroutineItem)


def test_scheduler_successors(config):
    fcode_mod = """
module some_mod
    implicit none
    type some_type
        real :: a
    contains
        procedure :: procedure => some_procedure
        procedure :: routine
        procedure :: other
        generic :: do => procedure, routine
    end type some_type
contains
    subroutine some_procedure(t, i)
        class(some_type), intent(inout) :: t
        integer, intent(in) :: i
        t%a = t%a + real(i)
    end subroutine some_procedure

    subroutine routine(t, v)
        class(some_type), intent(inout) :: t
        real, intent(in) :: v
        t%a = t%a + v
        call t%other
    end subroutine routine

    subroutine other(t)
        class(some_type), intent(in) :: t
        print *,t%a
    end subroutine other
end module some_mod
    """.strip()

    fcode = """
subroutine caller(val)
    use some_mod, only: some_type
    implicit none
    real, intent(inout) :: val
    type(some_type) :: t
    t%a = val
    call t%routine(1)
    call t%routine(2.0)
    call t%do(10)
    call t%do(20.0)
    val = t%a
end subroutine caller
    """.strip()

    class SuccessorTransformation(Transformation):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.counter = {}

        def transform_subroutine(self, routine, **kwargs):
            item = kwargs.get('item')
            if item and item.local_name != routine.name.lower():
                return

            assert item.local_name in ('caller', 'routine', 'some_procedure', 'other')
            self.counter[item.local_name] = self.counter.get(item.local_name, 0) + 1

            successors = kwargs.get('successors')
            assert isinstance(successors, list)
            if item.local_name == 'caller':
                expected_successors = {
                    'some_mod#some_type%routine', 'some_mod#some_type%do',
                    'some_mod#some_type%procedure', 'some_mod#some_procedure', 'some_mod#routine'
                }
            elif item.local_name == 'routine':
                expected_successors = {'some_mod#some_type%other', 'some_mod#other'}
            else:
                expected_successors = set()
            assert expected_successors == set(successors)

    workdir = gettempdir()/'test_scheduler_successors'
    workdir.mkdir(exist_ok=True)
    (workdir/'some_mod.F90').write_text(fcode_mod)
    (workdir/'caller.F90').write_text(fcode)

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['caller'])

    transformation = SuccessorTransformation()
    scheduler.process(transformation=transformation)

    assert transformation.counter == {
        'caller': 1,
        'routine': 1,
        'some_procedure': 1,
        'other': 1,
    }

    rmtree(workdir)


@pytest.mark.parametrize('full_parse', [True, False])
def test_scheduler_add_dependencies(config, full_parse):
    fcode_mod = """
module some_mod
    implicit none
    type some_type
        integer :: a
    contains
        procedure :: some_routine
        procedure :: some_function
    end type some_type
contains
    subroutine some_routine(t)
        class(some_type), intent(inout) :: t
        t%a = 5
    end subroutine some_routine

    integer function some_function(t)
        class(some_type), intent(in) :: t
        some_function = t%a
    end function some_function
end module some_mod
    """.strip()

    fcode_caller = """
subroutine caller(b)
    use some_mod, only: some_type
    implicit none
    integer, intent(inout) :: b
    type(some_type) :: t
    t%a = b
    call t%some_routine()
    b = t%some_function()
end subroutine caller
    """.strip()

    workdir = gettempdir()/'test_scheduler_add_dependencies'
    workdir.mkdir(exist_ok=True)
    (workdir/'some_mod.F90').write_text(fcode_mod)
    (workdir/'caller.F90').write_text(fcode_caller)

    def verify_graph(scheduler, expected_items, expected_dependencies):
        assert len(scheduler.items) == len(expected_items)
        assert all(n in scheduler.items for n in expected_items)
        assert len(scheduler.dependencies) == len(expected_dependencies)
        assert all(e in scheduler.dependencies for e in expected_dependencies)

        assert all(item.source._incomplete is not full_parse for item in scheduler.items)

        # Testing of callgraph visualisation
        cg_path = workdir/'callgraph'
        scheduler.callgraph(cg_path)

        vgraph = VisGraphWrapper(cg_path)
        assert all(n.upper() in vgraph.nodes for n in expected_items)
        assert all((e[0].upper(), e[1].upper()) in vgraph.edges for e in expected_dependencies)

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['caller'], full_parse=full_parse)

    expected_items = [
        '#caller', 'some_mod#some_type%some_routine', 'some_mod#some_routine'
    ]
    expected_dependencies = [
        ('#caller', 'some_mod#some_type%some_routine'),
        ('some_mod#some_type%some_routine', 'some_mod#some_routine')
    ]
    verify_graph(scheduler, expected_items, expected_dependencies)

    # Function should be in the obj map already
    assert 'some_mod#some_function' in scheduler.obj_map

    # Add inline call dependency
    scheduler.add_dependencies(
        {'#caller': ['some_mod#some_type%some_function']}
    )

    # Scheduler should have automatically added further relevant dependencies
    expected_items += [
        'some_mod#some_type%some_function', 'some_mod#some_function'
    ]
    expected_dependencies += [
        ('#caller', 'some_mod#some_type%some_function'),
        ('some_mod#some_type%some_function', 'some_mod#some_function')
    ]
    verify_graph(scheduler, expected_items, expected_dependencies)

    rmtree(workdir)


def test_scheduler_cached_properties():
    fcode = """
subroutine some_routine
    implicit none
    integer i
    i = 1
end subroutine some_routine
    """.strip()

    source = Sourcefile.from_source(fcode)
    item = SubroutineItem('#some_routine', source=source)

    assert not item.imports
    assert not item.qualified_imports

    # Add a random import...
    item.routine.spec.prepend(
        Import(module='some_mod', symbols=(Variable(name='some_var', scope=item.routine),))
    )

    # ...and see the caching in effect
    assert not item.imports
    assert not item.qualified_imports

    # Clear the cache to update the property
    item.clear_cached_property('imports')
    assert item.imports and item.imports[0].module == 'some_mod'
    assert 'some_var' in item.qualified_imports


@pytest.mark.parametrize('full_parse', [False, True])
def test_scheduler_cycle(config, full_parse):
    fcode_mod = """
module some_mod
    implicit none
    type some_type
        integer :: a
    contains
        procedure :: proc => some_proc
        procedure :: other => some_other
    end type some_type
contains
    recursive subroutine some_proc(this, val, recurse, fallback)
        class(some_type), intent(inout) :: this
        integer, intent(in) :: val
        logical, intent(in), optional :: recurse

        if (present(recurse)) then
            if (present(fallback)) then
                call this%other(val)
            else
                call some_proc(this, val, .true., .true.)
            end if
        else
            call this%proc(val, .true.)
        end if
    end subroutine some_proc

    subroutine some_other(this, val)
        class(some_type), intent(inout) :: this
        integer, intent(in) :: val
        this%a = val
    end subroutine some_other
end module some_mod
    """.strip()

    fcode_caller = """
subroutine caller
    use some_mod, only: some_type
    implicit none
    type(some_type) :: t

    call t%proc(1)
end subroutine caller
    """.strip()

    workdir = gettempdir()/'test_scheduler_cycle'
    workdir.mkdir(exist_ok=True)
    (workdir/'some_mod.F90').write_text(fcode_mod)
    (workdir/'caller.F90').write_text(fcode_caller)

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['caller'], full_parse=full_parse)

    # Make sure we the outgoing edges from the recursive routine to the procedure binding
    # and itself are removed but the other edge still exists
    assert (scheduler['#caller'], scheduler['some_mod#some_type%proc']) in scheduler.dependencies
    assert (scheduler['some_mod#some_type%proc'], scheduler['some_mod#some_proc']) in scheduler.dependencies
    assert (scheduler['some_mod#some_proc'], scheduler['some_mod#some_type%proc']) not in scheduler.dependencies
    assert (scheduler['some_mod#some_proc'], scheduler['some_mod#some_proc']) not in scheduler.dependencies
    assert (scheduler['some_mod#some_proc'], scheduler['some_mod#some_type%other']) in scheduler.dependencies
    assert (scheduler['some_mod#some_type%other'], scheduler['some_mod#some_other']) in scheduler.dependencies

    rmtree(workdir)
