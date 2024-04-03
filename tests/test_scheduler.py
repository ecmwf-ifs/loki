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

from collections import deque
from itertools import chain
from functools import partial
from pathlib import Path
import re
from shutil import rmtree
from subprocess import CalledProcessError
import pytest

from conftest import available_frontends, graphviz_present
from loki import (
    Scheduler, SchedulerConfig, DependencyTransformation, FP, OFP, OMNI,
    HAVE_FP, HAVE_OFP, HAVE_OMNI, REGEX, Sourcefile, FindNodes, CallStatement,
    fexprgen, Transformation, BasicType, Subroutine,
    gettempdir, ProcedureSymbol, Item, ProcedureItem, ProcedureBindingItem, InterfaceItem,
    ProcedureType, DerivedType, TypeDef, Scalar, Array, FindInlineCalls,
    Import, flatten, as_tuple, TypeDefItem, SFilter, CaseInsensitiveDict, Comment,
    ModuleWrapTransformation, Dimension, PreprocessorDirective, ExternalItem,
    Pipeline, Assignment, Literal
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
            'disable': ['abort'],
            'enable_imports': True,
        },
        'routines': {}
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


@pytest.fixture(name='driverA_dependencies')
def fixture_driverA_dependencies():
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
def fixture_driverB_dependencies():
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


@pytest.fixture(name='proj_typebound_dependencies')
def fixture_proj_typebound_dependencies():
    return {
        '#driver': (
            'typebound_item',
            # 'typebound_item#some_type',
            'typebound_item#some_type%other_routine',
            'typebound_item#some_type%some_routine',
            'typebound_header',
            # 'typebound_header#header_type',
            'typebound_header#header_type%member_routine',
            'typebound_header#header_type%routine',
            'typebound_other#other_type',
            'typebound_other#other_type%member',
            'typebound_other#other_type%var%member_routine',
        ),
        'typebound_item': ('typebound_header',),
        'typebound_item#some_type': (),
        'typebound_item#some_type%some_routine': ('typebound_item#some_routine',),
        'typebound_item#some_type%other_routine': (
            'typebound_item#other_routine',
        ),
        'typebound_item#some_type%routine': ('typebound_item#module_routine',),
        'typebound_item#some_type%routine1': ('typebound_item#routine1',),
        'typebound_item#some_type%routine2': ('typebound_item#routine',),
        'typebound_item#routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%some_routine',
        ),
        'typebound_item#routine1': (
            'typebound_item#module_routine',
            'typebound_item#some_type')
        ,
        'typebound_item#some_routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%routine',
        ),
        'typebound_item#other_routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%routine1',
            'typebound_item#some_type%routine2',
            'typebound_header#abor1'
        ),
        'typebound_item#module_routine': (),
        'typebound_header': (),
        'typebound_header#header_type': (),
        'typebound_header#header_type%member_routine': ('typebound_header#header_member_routine',),
        'typebound_header#header_member_routine': ('typebound_header#header_type',),
        'typebound_header#header_type%routine': (
            'typebound_header#header_type%routine_real',
            'typebound_header#header_type%routine_integer'
        ),
        'typebound_header#header_type%routine_real': (
            'typebound_header#header_routine_real',
        ),
        'typebound_header#header_routine_real': (
            'typebound_header#header_type',
        ),
        'typebound_header#header_type%routine_integer': (
            'typebound_header#routine_integer',
        ),
        'typebound_header#routine_integer': (
            'typebound_header#header_type',
        ),
        'typebound_header#abor1': (),
        'typebound_other#other_type': ('typebound_header#header_type',),
        'typebound_other#other_type%member': ('typebound_other#other_member',),
        'typebound_other#other_member': (
            'typebound_header#header_member_routine',
            'typebound_other#other_type',
            'typebound_other#other_type%var%member_routine'
        ),
        'typebound_other#other_type%var%member_routine': ('typebound_header#header_type%member_routine',)
    }


class VisGraphWrapper:
    """
    Testing utility to parse the generated callgraph visualisation.
    """

    _re_nodes = re.compile(r'\s*\"?(?P<node>[\w%#./-]+)\"? \[colo', re.IGNORECASE)
    _re_edges = re.compile(r'\s*\"?(?P<parent>[\w%#./-]+)\"? -> \"?(?P<child>[\w%#./-]+)\"?', re.IGNORECASE)

    def __init__(self, path):
        with Path(path).open('r') as f:
            self.text = f.read()

    @property
    def nodes(self):
        return list(self._re_nodes.findall(self.text))

    @property
    def edges(self):
        return list(self._re_edges.findall(self.text))


def test_scheduler_enrichment(here, config, frontend):
    projA = here/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=['driverA'], frontend=frontend
    )

    for item in SFilter(scheduler.sgraph, item_filter=ProcedureItem):
        dependency_map = CaseInsensitiveDict(
            (item_.local_name, item_) for item_ in scheduler.sgraph.successors(item)
        )
        for call in FindNodes(CallStatement).visit(item.ir.body):
            if call_item := dependency_map.get(str(call.name)):
                assert call.routine is call_item.ir


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
@pytest.mark.parametrize('with_file_graph', [True, False, 'filegraph_simple'])
@pytest.mark.parametrize('with_legend', [True, False])
@pytest.mark.parametrize('seed', ['driverA', 'driverA_mod#driverA'])
def test_scheduler_graph_simple(here, config, frontend, driverA_dependencies, with_file_graph, with_legend, seed):
    """
    Create a simple task graph from a single sub-project:

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=seed, frontend=frontend
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

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_simple'
    if not isinstance(with_file_graph, bool):
        with_file_graph = here/with_file_graph
    scheduler.callgraph(cg_path, with_file_graph=with_file_graph, with_legend=with_legend)

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
            fg_path = here/with_file_graph
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
def test_scheduler_graph_partial(here, config, frontend, seed):
    """
    Create a sub-graph from a select set of branches in  single project:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = here/'sources/projA'

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


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_graph_config_file(here, frontend):
    """
    Create a sub-graph from a branches using a config file:

    projA: compute_l1 -> compute_l2

           another_l1 -> another_l2
    """
    projA = here/'sources/projA'
    config = projA/'scheduler_partial.config'

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, frontend=frontend)

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
    assert 'compute_l2' not in scheduler.items  # We're blocking `compute_l2` in config file

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_config_file'
    scheduler.callgraph(cg_path)
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
def test_scheduler_graph_blocked(here, config, frontend, seed):
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
        seed_routines=[seed], frontend=frontend
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

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_block'
    fg_path = cg_path.with_name(cg_path.name + '_file_graph')
    scheduler.callgraph(cg_path, with_file_graph=True)
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



@pytest.mark.parametrize('seed', ['driverA', 'driverA_mod#driverA'])
def test_scheduler_definitions(here, config, frontend, seed):
    """
    Create a simple task graph and inject type info via `definitions`.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                     <header_type>
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    header = Sourcefile.from_file(projA/'module/header_mod.f90', frontend=frontend)

    scheduler = Scheduler(
        paths=projA, definitions=header['header_mod'], includes=projA/'include',
        config=config, seed_routines=[seed], frontend=frontend
    )

    driver = scheduler.item_factory.item_cache['drivera_mod#drivera'].ir
    call = FindNodes(CallStatement).visit(driver.body)[0]
    assert call.arguments[0].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[0].shape) == '(:,)'
    assert call.arguments[1].parent.type.dtype.typedef is not BasicType.DEFERRED
    assert fexprgen(call.arguments[1].shape) == '(3, 3)'


@pytest.mark.parametrize('seed', ['compute_l1', 'compute_l1_mod#compute_l1'])
def test_scheduler_process(here, config, frontend, seed):
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

    scheduler = Scheduler(paths=projA, includes=projA/'include', config=config, frontend=frontend)

    class RoleComment(Transformation):
        """
        Simply add role as a comment in the subroutine body.
        """
        def transform_subroutine(self, routine, **kwargs):
            role = kwargs.get('role', None)
            routine.body.prepend(Comment(f'! {role}'))

    # Apply re-naming transformation and check result
    scheduler.process(transformation=RoleComment())

    key_role_map = {
        'compute_l1_mod#compute_l1': 'driver',
        'compute_l2_mod#compute_l2': 'kernel',
        '#another_l1': 'driver',
        '#another_l2': 'kernel',
    }
    for key, role in key_role_map.items():
        comment = scheduler[key].ir.body.body[0]
        assert isinstance(comment, Comment)
        assert comment.text == f'! {role}'


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
@pytest.mark.parametrize('seed', ['driverE_single', 'driverE_mod#driverE_single'])
def test_scheduler_process_filter(here, config, frontend, seed):
    """
    Applies simple kernels over complex callgraphs to check that we
    only apply to the entities requested and only once!

    projA: driverE_single -> kernelE -> compute_l1 -> compute_l2
                                  |
                                  | --> ghost_busters
    """
    projA = here/'sources/projA'
    projB = here/'sources/projB'

    config['routines'] = {
        seed: {'role': 'driver', 'expand': True,},
    }

    scheduler = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=config, frontend=frontend
    )

    class XMarksTheSpot(Transformation):
        """
        Prepend an 'X' comment to a given :any:`Subroutine`
        """
        def transform_subroutine(self, routine, **kwargs):
            routine.body.prepend(Comment('! X'))

    # Apply transformation and check result
    scheduler.process(transformation=XMarksTheSpot())

    key_x_map = {
        'drivere_mod#drivere_single': True,
        'drivere_mod#drivere_multiple': False,
        'kernele_mod#kernele': True,
        'kernele_mod#kernelet': False,
        'compute_l1_mod#compute_l1': True,
        'compute_l2_mod#compute_l2': True,
    }

    # Internal member procedure is not included
    assert not any(
        item.name.endswith('#ghost_busters')
        for item in scheduler.item_factory.item_cache.values()
    )

    for key, is_transformed in key_x_map.items():
        item = scheduler[key]
        if is_transformed:
            ir = item.ir
        else:
            # key should not be found in the callgraph but scope should still exist in the
            # item_cache because the file has been indexed
            assert item is None
            scope_name, local_name = key.split('#')
            assert scope_name in scheduler.item_factory.item_cache
            ir = scheduler.item_factory.item_cache[scope_name].ir[local_name]
        first_node = ir.body.body[0]
        first_node_is_x = isinstance(first_node, Comment) and first_node.text == '! X'
        assert first_node_is_x == is_transformed


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_graph_multiple_combined(here, config, driverB_dependencies, frontend):
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
        seed_routines=['driverB_mod#driverB'], frontend=frontend
    )

    assert set(scheduler.items) == {item.lower() for item in driverB_dependencies}
    assert set(scheduler.dependencies) == {
        (item.lower(), child.lower())
        for item, children in driverB_dependencies.items()
        for child in children
    }

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_multiple_combined'
    scheduler.callgraph(cg_path)

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
    configA['routines'] = {
        'kernelB': {
            'role': 'kernel',
            'ignore': ['ext_driver'],
            'enrich': ['ext_driver'],
        },
    }

    schedulerA = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=configA,
        seed_routines=['driverB'], frontend=frontend
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
        'compute_l1_mod#compute_l1': (
            'compute_l2_mod#compute_l2',
        ),
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

    # Test callgraph visualisation
    cg_path = here/'callgraph_multiple_separate_A'
    schedulerA.callgraph(cg_path)

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
        'ext_driver': { 'role': 'kernel' },
    }

    schedulerB = Scheduler(
        paths=projB, config=configB, seed_routines=['ext_driver'],
        frontend=frontend
    )

    # TODO: Technically we should check that the role=kernel has been honoured in B
    assert 'ext_driver_mod#ext_driver' in schedulerB.items
    assert 'ext_kernel_mod#ext_kernel' in schedulerB.items
    assert ('ext_driver_mod#ext_driver', 'ext_kernel_mod#ext_kernel') in schedulerB.dependencies

    # Check that the call from kernelB to ext_driver has been enriched with IPA meta-info
    call = FindNodes(CallStatement).visit(schedulerA['kernelb_mod#kernelb'].ir.body)[1]
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


@pytest.mark.parametrize('strict', [True, False])
def test_scheduler_graph_multiple_separate_enrich_fail(here, config, frontend, strict):
    """
    Tests that explicit enrichment in "strict" mode will fail because it can't
    find ext_driver

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
                     <ext_driver>

    projB:            ext_driver -> ext_kernelfail
    """
    projA = here/'sources/projA'

    configA = config.copy()
    configA['default']['strict'] = strict
    configA['routine'] = [
        {
            'name': 'kernelB',
            'role': 'kernel',
            'ignore': ['ext_driver'],
            'enrich': ['ext_driver'],
        },
    ]

    schedulerA = Scheduler(
        paths=[projA], includes=projA/'include', config=configA,
        seed_routines=['driverB'], frontend=frontend
    )

    expected_dependenciesA = {
        'driverB_mod#driverB': ('kernelB_mod#kernelB', 'header_mod', 'header_mod#header_type'),
        'kernelB_mod#kernelB': ('compute_l1_mod#compute_l1', 'ext_driver_mod#ext_driver'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'header_mod': (),
        'header_mod#header_type': (),
        'ext_driver_mod#ext_driver': (),
    }

    assert set(schedulerA.items) == {node.lower() for node in expected_dependenciesA}
    assert set(schedulerA.dependencies) == {
        (a.lower(), b.lower()) for a, deps in expected_dependenciesA.items() for b in deps
    }

    class DummyTrafo(Transformation):
        pass

    if strict:
        with pytest.raises(RuntimeError):
            schedulerA.process(transformation=DummyTrafo())
    else:
        schedulerA.process(transformation=DummyTrafo())


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
        seed_routines=['driverC_mod#driverC'], frontend=frontend
    )

    expected_dependencies = {
        'driverc_mod#driverc': ('header_mod', 'header_mod#header_type', 'kernelc_mod#kernelc',),
        'kernelc_mod#kernelc': ('compute_l1_mod#compute_l1', 'proj_c_util_mod#routine_one',),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'proj_c_util_mod#routine_one': ('proj_c_util_mod#routine_two',),
        'proj_c_util_mod#routine_two': (),
        'header_mod#header_type': (),
        'header_mod': (),
    }
    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # Ensure that we got the right routines from the module
    assert scheduler['proj_c_util_mod#routine_one'].ir.name == 'routine_one'
    assert scheduler['proj_c_util_mod#routine_two'].ir.name == 'routine_two'


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
        seed_routines=['driverD_mod#driverD'], frontend=frontend
    )

    expected_dependencies = {
        'driverd_mod#driverd': ('kerneld_mod#kerneld', 'header_mod', 'header_mod#header_type'),
        'kerneld_mod#kerneld': ('compute_l1_mod#compute_l1', 'proj_c_util_mod#routine_one'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'proj_c_util_mod#routine_one': ('proj_c_util_mod#routine_two',),
        'proj_c_util_mod#routine_two': (),
        'header_mod#header_type': (),
        'header_mod': (),
    }
    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # Ensure that we got the right routines from the module
    assert scheduler['proj_c_util_mod#routine_one'].ir.name == 'routine_one'
    assert scheduler['proj_c_util_mod#routine_two'].ir.name == 'routine_two'


@pytest.mark.parametrize('strict', [True, False])
def test_scheduler_missing_files(here, config, frontend, strict):
    """
    Ensure that ``strict=True`` triggers failure if source paths are
    missing and that ``strict=False`` goes through gracefully.

    projA: driverC -> kernelC -> compute_l1<replicated> -> compute_l2
                           |
    projC:                 < cannot find path >
    """
    projA = here/'sources/projA'

    config['default']['strict'] = strict
    scheduler = Scheduler(
        paths=[projA], includes=projA/'include', config=config,
        seed_routines=['driverC_mod#driverC'], frontend=frontend
    )

    expected_dependencies = {
        'driverc_mod#driverc': ('kernelc_mod#kernelc', 'header_mod#header_type', 'header_mod'),
        'kernelc_mod#kernelc': ('compute_l1_mod#compute_l1', 'proj_c_util_mod#routine_one'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'header_mod#header_type': (),
        'header_mod': (),
        'proj_c_util_mod#routine_one': (),
    }
    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # Ensure that the missing items are not in the graph
    assert isinstance(scheduler['proj_c_util_mod#routine_one'], ExternalItem)
    assert 'proj_c_util_mod#routine_two' not in scheduler.items

    # Check processing with missing items
    class CheckApply(Transformation):

        def apply(self, source, post_apply_rescope_symbols=False, **kwargs):
            assert 'item' in kwargs
            assert not isinstance(kwargs['item'], ExternalItem)
            super().apply(source, post_apply_rescope_symbols=post_apply_rescope_symbols, **kwargs)

    if strict:
        with pytest.raises(RuntimeError):
            scheduler.process(CheckApply())
    else:
        scheduler.process(CheckApply())


@pytest.mark.parametrize('preprocess', [False, True])   # NB: With preprocessing, ext_driver is no longer
                                                        #     wrapped inside a module but instead imported
                                                        #     via an intfb.h
def test_scheduler_dependencies_ignore(here, preprocess, frontend):
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
        'default': {
            'role': 'kernel', 'expand': True, 'strict': True, 'enable_imports': True
        },
        'routines': {
            'driverB': {'role': 'driver'},
            'kernelB': {'ignore': ['ext_driver']},
        }
    })

    configB = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel', 'expand': True, 'strict': True, 'enable_imports': True
        },
        'routines': {
            'ext_driver': {'role': 'kernel'}
        }
    })

    schedulerA = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=configA,
        frontend=frontend, preprocess=preprocess
    )

    schedulerB = Scheduler(
        paths=projB, includes=projB/'include', config=configB,
        frontend=frontend, preprocess=preprocess
    )

    expected_items_a = [
        'driverB_mod#driverB', 'kernelB_mod#kernelB',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2',
        'header_mod', 'header_mod#header_type'
    ]
    if preprocess:
        expected_items_b = [
            '#ext_driver', 'ext_kernel_mod', 'ext_kernel_mod#ext_kernel'
        ]
    else:
        expected_items_b = [
            'ext_driver_mod#ext_driver', 'ext_kernel_mod', 'ext_kernel_mod#ext_kernel'
        ]

    assert set(schedulerA.items) == {n.lower() for n in expected_items_a + expected_items_b}
    assert all(not schedulerA[name].is_ignored for name in expected_items_a)
    assert all(schedulerA[name].is_ignored for name in expected_items_b)

    assert set(schedulerB.items) == {n.lower() for n in expected_items_b}

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_dependencies_ignore'
    fg_path = cg_path.with_name(cg_path.name + '_file_graph')
    schedulerA.callgraph(cg_path, with_file_graph=True)

    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {n.upper() for n in expected_items_a + expected_items_b}

    file_dependencies = {
        'proja/module/driverb_mod.f90': ('proja/module/header_mod.f90', 'proja/module/kernelb_mod.f90'),
        'proja/module/header_mod.f90': (),
        'proja/module/kernelb_mod.f90': ('proja/module/compute_l1_mod.f90', 'projb/external/ext_driver_mod.f90'),
        'proja/module/compute_l1_mod.f90': ('proja/module/compute_l2_mod.f90',),
        'proja/module/compute_l2_mod.f90': (),
        'projb/external/ext_driver_mod.f90': ('projb/module/ext_kernel.f90',),
        'projb/module/ext_kernel.f90': (),
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

    # Apply dependency injection transformation and ensure only the root driver is not transformed
    transformations = (
        ModuleWrapTransformation(module_suffix='_mod'),
        DependencyTransformation(suffix='_test', module_suffix='_mod')
    )
    for transformation in transformations:
        schedulerA.process(transformation)

    assert schedulerA.items[0].source.all_subroutines[0].name == 'driverB'
    assert schedulerA.items[1].source.all_subroutines[0].name == 'kernelB_test'
    assert schedulerA.items[4].source.all_subroutines[0].name == 'compute_l1_test'
    assert schedulerA.items[5].source.all_subroutines[0].name == 'compute_l2_test'

    # Note that 'ext_driver' and 'ext_kernel' are no longer part of the dependency graph because the
    # renaming makes it impossible to discover the non-transformed routines
    assert all(not name in schedulerA for name in expected_items_b)
    assert 'ext_driver_test_mod#ext_driver_test' not in schedulerA

    # For the second target lib, we want the driver to be converted
    for transformation in transformations:
        schedulerB.process(transformation=transformation)

    # Repeat processing to ensure DependencyTransform is idempotent
    for transformation in transformations:
        schedulerB.process(transformation=transformation)

    assert schedulerB.items[0].source.all_subroutines[0].name == 'ext_driver_test'

    # This is the untransformed original module
    assert schedulerB['ext_kernel_mod'].source.all_subroutines[0].name == 'ext_kernel'

    # This is the module-wrapped procedure
    assert schedulerB['ext_kernel_test_mod#ext_kernel_test'].source.all_subroutines[0].name == 'ext_kernel_test'


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
        'default': {'role': 'kernel', 'expand': True, 'strict': True, 'ignore': ('header_mod',)},
        'routines': {
            'driverB': {'role': 'driver'},
            'kernelB': {'ignore': ['ext_driver']},
        }
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

    scheduler.write_cmake_plan(
        filepath=planfile, mode='foobar', buildpath=builddir, rootpath=sourcedir
    )

    # Validate the generated lists
    expected_files = {
        proj_a/'module/driverB_mod.f90', proj_a/'module/kernelB_mod.F90',
        proj_a/'module/compute_l1_mod.f90', proj_a/'module/compute_l2_mod.f90'
    }

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


def test_scheduler_item_dependencies(here):
    """
    Make sure children are correct and unique for items
    """
    config = SchedulerConfig.from_dict({
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routines': {
            'driver': {'role': 'driver'},
            'another_driver': {'role': 'driver'}
        }
    })

    proj_hoist = here/'sources/projHoist'

    scheduler = Scheduler(paths=proj_hoist, config=config)

    assert tuple(
        call.name for call in scheduler['transformation_module_hoist#driver'].dependencies
    ) == (
        'kernel1', 'kernel2'
    )
    assert tuple(
        call.name for call in scheduler['transformation_module_hoist#another_driver'].dependencies
    ) == (
        'kernel1',
    )
    assert not scheduler['subroutines_mod#kernel1'].dependencies
    assert tuple(
        call.name for call in scheduler['subroutines_mod#kernel2'].dependencies
    ) == (
        'device1', 'device2'
    )
    assert tuple(
        call.name for call in scheduler['subroutines_mod#device1'].dependencies
    ) == (
        'device2',
    )
    assert not scheduler['subroutines_mod#device2'].dependencies


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
    assert sorted(scheduler.item_factory.item_cache.keys()) == [
        '#random_call_0', '#random_call_2', '#test',
        str(loki_69_dir/'test.f90').lower()
    ]
    assert '#random_call_1' not in scheduler.item_factory

    children_map = {
        '#test': ('#random_call_0', '#random_call_2'),
        '#random_call_0': (),
        '#random_call_2': ()
    }
    assert len(scheduler.items) == len(children_map)
    for item in scheduler.items:
        assert set(scheduler.sgraph.successors(item)) == set(children_map[item.name])


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_scopes(here, config, frontend):
    """
    Test discovery with import renames and duplicate names in separate scopes

      driver ----> kernel1_mod#kernel ----> kernel1_impl#kernel_impl
        |
        +--------> kernel2_mod#kernel ----> kernel2_impl#kernel_impl
    """
    proj = here/'sources/projScopes'

    scheduler = Scheduler(paths=proj, seed_routines=['driver'], config=config, frontend=frontend)

    expected_dependencies = {
        '#driver': (
            'kernel1_mod#kernel',
            'kernel2_mod#kernel',
        ),
        'kernel1_mod#kernel': (
            'kernel1_impl',
            'kernel1_impl#kernel_impl',
        ),
        'kernel1_impl': (),
        'kernel1_impl#kernel_impl': (),
        'kernel2_mod#kernel': (
            'kernel2_impl',
            'kernel2_impl#kernel_impl',
        ),
        'kernel2_impl': (),
        'kernel2_impl#kernel_impl': (),
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items()
        for b in deps
    }

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_scopes'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {
        n.upper() for n in expected_dependencies
    }
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in expected_dependencies.items()
        for b in deps
    }

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_typebound(here, config, frontend, proj_typebound_dependencies):
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
        full_parse=False, frontend=frontend,
    )

    assert set(scheduler.items) == set(proj_typebound_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_typebound'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {n.upper() for n in proj_typebound_dependencies}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
def test_scheduler_typebound_ignore(here, config, frontend, proj_typebound_dependencies):
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

    config['default']['disable'] += [
        'some_type%some_routine',
        'header_member_routine'
    ]
    config['routines'] = {
        'other_member': {
            'disable': config['default']['disable'] + ['member_routine']
        }
    }

    items_to_remove = (
        'typebound_item#some_type%some_routine',
        'typebound_item#some_routine',
        'typebound_item#some_type%routine',
        'typebound_header#header_member_routine',
    )

    proj_typebound_dependencies = {
        name: tuple(dep for dep in deps if dep not in items_to_remove)
        for name, deps in proj_typebound_dependencies.items()
        if name not in items_to_remove
    }

    scheduler = Scheduler(
        paths=proj, seed_routines=['driver'], config=config,
        full_parse=False, frontend=frontend
    )

    assert set(scheduler.items) == set(proj_typebound_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    # Testing of callgraph visualisation
    cg_path = here/'callgraph_typebound'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {n.upper() for n in proj_typebound_dependencies}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in proj_typebound_dependencies.items() for b in deps
    }

    cg_path.unlink()
    cg_path.with_suffix('.pdf').unlink()


@pytest.mark.parametrize('use_file_graph', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
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
        expected = ['driver_mod.f90', 'subroutines_mod.f90']
        expected = [str(proj/'module'/n).lower() + '::' + n for n in expected]
    else:
        expected = [
            'transformation_module_hoist#driver::driver', 'subroutines_mod#kernel1::kernel1',
            'subroutines_mod#kernel2::kernel2', 'subroutines_mod#device1::device1',
            'subroutines_mod#device2::device2'
        ]

    class LoggingTransformation(Transformation):

        reverse_traversal = reverse

        traverse_file_graph = use_file_graph

        def __init__(self):
            self.record = []

        def transform_file(self, sourcefile, **kwargs):
            self.record += [kwargs['item'].name + '::' + sourcefile.path.name]

        def transform_module(self, module, **kwargs):
            self.record += [kwargs['item'].name + '::' + module.name]

        def transform_subroutine(self, routine, **kwargs):
            self.record += [kwargs['item'].name + '::' + routine.name]

    transformation = LoggingTransformation()
    scheduler.process(transformation=transformation)

    if reverse:
        assert transformation.record == expected[::-1]
    else:
        assert transformation.record == expected


@pytest.mark.parametrize('use_file_graph', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_scheduler_member_routines(config, frontend, use_file_graph, reverse):
    """
    Make sure that transformation processing works also for contained member routines

    Notably, this does currently _NOT_ work and this test is here to document that fact and
    serve as the test base for when this has been corrected.
    """
    fcode_mod = """
module member_mod
    implicit none
contains
    subroutine my_routine(ret)
        integer, intent(out) :: ret
        ret = 1
    end subroutine my_routine

    subroutine driver
        integer :: val
        call my_member
        write(*,*) val
    contains
        subroutine my_member
            call my_routine(val)
        end subroutine my_member
    end subroutine driver
end module member_mod
    """.strip()

    workdir = gettempdir()/'test_scheduler_member_routines'
    workdir.mkdir(exist_ok=True)
    (workdir/'member_mod.F90').write_text(fcode_mod)

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['member_mod#driver'], frontend=frontend)

    class LoggingTransformation(Transformation):

        reverse_traversal = reverse

        traverse_file_graph = use_file_graph

        def __init__(self):
            self.record = []

        def transform_file(self, sourcefile, **kwargs):
            self.record += [kwargs['item'].name + '::' + sourcefile.path.name]

        def transform_module(self, module, **kwargs):
            self.record += [kwargs['item'].name + '::' + module.name]

        def transform_subroutine(self, routine, **kwargs):
            self.record += [kwargs['item'].name + '::' + routine.name]

    transformation = LoggingTransformation()
    scheduler.process(transformation=transformation)

    if use_file_graph:
        expected = [f'{workdir/"member_mod.F90"!s}'.lower() + '::member_mod.F90']
    else:
        expected = [
            'member_mod#driver::driver',
            'member_mod#driver#my_member::my_member',
            'member_mod#my_routine::my_routine',
        ]

    if not use_file_graph:
        # Because the scheduler cannot represent contained member routines currently, it does
        # not find the call dependencies via the member routine and therefore doesn't process
        # these subroutines with the transformation
        if len(scheduler.items) == 1 and transformation.record == flatten(expected[:1]):
            pytest.xfail(reason='Scheduler unable to represent contained member routines as graph items')

    assert transformation.record == flatten(expected)

    rmtree(workdir)


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
        assert isinstance(call.routine, Subroutine)
        assert isinstance(call.name.type.dtype.procedure, Subroutine)

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


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_interface_inline_call(here, config, frontend):
    """
    Test that inline function calls declared via an explicit interface are added as dependencies.
    """

    my_config = config.copy()
    my_config['routines'] = {
        'driver': {
            'role': 'driver',
            # 'disable': ['return_one', 'some_var', 'add_args', 'some_type']
        }
    }

    scheduler = Scheduler(paths=here/'sources/projInlineCalls', config=my_config, frontend=frontend)

    expected_dependencies = {
        '#driver': (
            '#double_real', 'some_module', 'some_module#add_args', 'some_module#return_one',
            'some_module#some_type', 'some_module#some_type%do_something', 'vars_module',
        ),
        '#double_real': ('vars_module',),
        'some_module': (),
        'some_module#add_args': ('some_module#add_two_args', 'some_module#add_three_args'),
        'some_module#add_two_args': (),
        'some_module#add_three_args': (),
        'some_module#return_one': (),
        'some_module#some_type': (),
        'some_module#some_type%do_something': ('some_module#add_const',),
        'some_module#add_const': ('some_module#some_type',),
        'vars_module': (),
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    assert isinstance(scheduler['some_module#add_args'], InterfaceItem)
    assert isinstance(scheduler['#double_real'], ProcedureItem)
    assert isinstance(scheduler['some_module#some_type'], TypeDefItem)
    assert isinstance(scheduler['some_module#add_two_args'], ProcedureItem)
    assert isinstance(scheduler['some_module#add_three_args'], ProcedureItem)

    # Testing of callgraph visualisation with imports
    workdir = gettempdir()/'test_scheduler_import_dependencies'
    workdir.mkdir(exist_ok=True)
    cg_path = workdir/'callgraph'
    scheduler.callgraph(cg_path)

    vgraph = VisGraphWrapper(cg_path)
    assert set(vgraph.nodes) == {i.upper() for i in expected_dependencies}
    assert set(vgraph.edges) == {
        (a.upper(), b.upper()) for a, deps in expected_dependencies.items() for b in deps
    }

    rmtree(workdir)


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_interface_dependencies(frontend, config):
    """
    Ensure that interfaces are treated as intermediate nodes and incur
    dependencies on the actual procedures
    """
    fcode_module = """
module test_scheduler_interface_dependencies_mod
    implicit none
    interface my_intf
        procedure proc1
        procedure proc2
    end interface my_intf
contains
    subroutine proc1(arg)
        integer, intent(inout) :: arg
        arg = arg + 1
    end subroutine proc1
    subroutine proc2(arg)
        real, intent(inout) :: arg
        arg = arg + 1.0
    end subroutine proc2
end module test_scheduler_interface_dependencies_mod
    """
    fcode_driver = """
subroutine test_scheduler_interface_dependencies_driver
    use test_scheduler_interface_dependencies_mod, only: my_intf
    implicit none
    integer i
    real a
    i = 0
    a = 0.0
    call my_intf(i)
    call my_intf(a)
end subroutine test_scheduler_interface_dependencies_driver
    """

    config['routines']['test_scheduler_interface_dependencies_driver'] = {
        'role': 'driver'
    }

    workdir = gettempdir()/'test_scheduler_interface_dependencies'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    (workdir/'test_scheduler_interface_dependencies_mod.F90').write_text(fcode_module)
    (workdir/'test_scheduler_interface_dependencies_driver.F90').write_text(fcode_driver)

    scheduler = Scheduler(
        paths=[workdir], config=config, seed_routines=['test_scheduler_interface_dependencies_driver'],
        frontend=frontend
    )

    expected_dependencies = {
        '#test_scheduler_interface_dependencies_driver': {
            'test_scheduler_interface_dependencies_mod#my_intf'
        },
        'test_scheduler_interface_dependencies_mod#my_intf': {
            'test_scheduler_interface_dependencies_mod#proc1', 'test_scheduler_interface_dependencies_mod#proc2'
        },
        'test_scheduler_interface_dependencies_mod#proc1': set(),
        'test_scheduler_interface_dependencies_mod#proc2': set()
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    assert isinstance(scheduler['test_scheduler_interface_dependencies_mod#my_intf'], InterfaceItem)
    assert isinstance(scheduler['test_scheduler_interface_dependencies_mod#proc1'], ProcedureItem)
    assert isinstance(scheduler['test_scheduler_interface_dependencies_mod#proc2'], ProcedureItem)

    rmtree(workdir)


def test_scheduler_item_successors(here, config, frontend):
    """
    Test that scheduler.item_successors always returns the original item.
    """

    my_config = config.copy()
    my_config['routines'] = {
        'driver': { 'role': 'driver' }
    }

    scheduler = Scheduler(paths=here/'sources/projInlineCalls', config=my_config, frontend=frontend)
    import_item = scheduler['vars_module']
    driver_item = scheduler['#driver']
    kernel_item = scheduler['#double_real']

    idA = id(import_item)

    for successor in scheduler.sgraph.successors(driver_item):
        if successor.name == import_item.name:
            assert id(successor) == idA
    for successor in scheduler.sgraph.successors(kernel_item):
        if successor.name == import_item.name:
            assert id(successor) == idA


@pytest.mark.parametrize('trafo_item_filter', [
    Item,
    ProcedureItem,
    (ProcedureItem, InterfaceItem, ProcedureBindingItem),
    (ProcedureItem, TypeDefItem),
])
def test_scheduler_successors(config, trafo_item_filter):
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
    call t%other
    val = t%a
end subroutine caller
    """.strip()

    expected_dependencies = {
        '#caller': (
            'some_mod#some_type',
            'some_mod#some_type%routine',
            'some_mod#some_type%do',
            'some_mod#some_type%other',
        ),
        'some_mod#some_type': (),
        'some_mod#some_type%routine': ('some_mod#routine',),
        'some_mod#some_type%do': (
            'some_mod#some_type%procedure',
            'some_mod#some_type%routine',
        ),
        'some_mod#some_type%other': ('some_mod#other',),
        'some_mod#routine': (
            'some_mod#some_type',
            'some_mod#some_type%other',
        ),
        'some_mod#other': (
            'some_mod#some_type',
        ),
        'some_mod#some_type%procedure': (
            'some_mod#some_procedure',
        ),
        'some_mod#some_procedure': (
            'some_mod#some_type',
        )
    }

    class SuccessorTransformation(Transformation):

        item_filter = trafo_item_filter

        def __init__(self, expected_successors, **kwargs):
            super().__init__(**kwargs)
            self.counter = {}
            self.expected_successors = expected_successors

        def transform_subroutine(self, routine, **kwargs):
            item = kwargs.get('item')
            assert item.local_name in ('caller', 'routine', 'some_procedure', 'other')
            self.counter[item.local_name] = self.counter.get(item.local_name, 0) + 1

            successors = kwargs.get('successors')
            assert set(successors) == set(self.expected_successors[item.name])

    workdir = gettempdir()/'test_scheduler_successors'
    workdir.mkdir(exist_ok=True)
    (workdir/'some_mod.F90').write_text(fcode_mod)
    (workdir/'caller.F90').write_text(fcode)

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['caller'])

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # Filter expected dependencies
    if trafo_item_filter != Item:
        item_filter = as_tuple(trafo_item_filter)
        if ProcedureItem in item_filter:
            item_filter += (ProcedureBindingItem, InterfaceItem)

        expected_dependencies = {
            item_name: tuple(
                dep for dep in dependencies
                if isinstance(scheduler[dep], item_filter)
            )
            for item_name, dependencies in expected_dependencies.items()
        }

    # Add dependency-dependencies for the full successor list
    expected_successors = {}
    for item_name, dependencies in expected_dependencies.items():
        item_successors = set()
        dep_queue = deque(dependencies)
        while dep_queue:
            dep_name = dep_queue.popleft()
            item_successors.add(dep_name)
            dep_queue.extend(
                dep_dep
                for dep_dep in expected_dependencies[dep_name]
                if dep_dep not in item_successors
            )
        expected_successors[item_name] = item_successors

    transformation = SuccessorTransformation(expected_successors)
    scheduler.process(transformation=transformation)

    assert transformation.counter == {
        'caller': 1,
        'routine': 1,
        'some_procedure': 1,
        'other': 1,
    }

    rmtree(workdir)


@pytest.mark.parametrize('full_parse', [True, False])
def test_scheduler_typebound_inline_call(config, full_parse):
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

    workdir = gettempdir()/'test_scheduler_typebound_inline_call'
    workdir.mkdir(exist_ok=True)
    (workdir/'some_mod.F90').write_text(fcode_mod)
    (workdir/'caller.F90').write_text(fcode_caller)

    def verify_graph(scheduler, expected_dependencies):
        assert set(scheduler.items) == set(expected_dependencies)
        assert set(scheduler.dependencies) == {
            (a, b) for a, deps in expected_dependencies.items() for b in deps
        }

        assert all(item.source._incomplete is not full_parse for item in scheduler.items)

        # Testing of callgraph visualisation
        cg_path = workdir/'callgraph'
        scheduler.callgraph(cg_path)

        vgraph = VisGraphWrapper(cg_path)
        assert set(vgraph.nodes) == {n.upper() for n in expected_dependencies}
        assert set(vgraph.edges) == {
            (a.upper(), b.upper()) for a, deps in expected_dependencies.items()
            for b in deps
        }

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['caller'], full_parse=full_parse)

    expected_dependencies = {
        '#caller': (
            'some_mod#some_type',
            'some_mod#some_type%some_routine',
        ),
        'some_mod#some_type': (),
        'some_mod#some_type%some_routine': ('some_mod#some_routine',),
        'some_mod#some_routine': ('some_mod#some_type',),
    }

    if scheduler.full_parse:
        # Inline Calls can only be fully resolved in a full parse
        expected_dependencies['#caller'] += ('some_mod#some_type%some_function',)
        expected_dependencies['some_mod#some_type%some_function'] = ('some_mod#some_function',)
        expected_dependencies['some_mod#some_function'] = ('some_mod#some_type',)

    verify_graph(scheduler, expected_dependencies)

    # TODO: test adding a nested derived type dependency!

    rmtree(workdir)


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


def test_scheduler_unqualified_imports(config):
    """
    Test that only qualified imports are added as children.
    """

    kernel = """
    subroutine kernel()
       use some_mod
       use other_mod, only: other_routine

       call other_routine
    end subroutine kernel
    """

    source = Sourcefile.from_source(kernel, frontend=REGEX)
    item = ProcedureItem(name='#kernel', source=source, config=config['default'])

    assert len(item.dependencies) == 3
    children = set()
    for dep in item.dependencies:
        if isinstance(dep, Import):
            if dep.symbols:
                children |= {f'{dep.module}#{str(s)}'.lower() for s in dep.symbols}
            else:
                children.add(dep.module.lower())
        elif isinstance(dep, CallStatement):
            children.add(str(dep.name).lower())
        else:
            assert False, 'Unexpected dependency type'
    assert children == {'some_mod', 'other_mod#other_routine', 'other_routine'}


def test_scheduler_depths(here, config, frontend):
    projA = here/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines=['driverA'], frontend=frontend
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


def test_scheduler_disable_wildcard(here, config):

    fcode_mod = """
module field_mod
  type field2d
    contains
    procedure :: init => field_init
  end type

  type field3d
    contains
    procedure :: init => field_init
  end type

  contains
    subroutine field_init()

    end subroutine
end module
"""

    fcode_driver = """
subroutine my_driver
  use field_mod, only: field2d, field3d, field_init
implicit none

  type(field2d) :: a, b
  type(field3d) :: c, d

  call a%init()
  call b%init()
  call c%init()
  call field_init(d)
end subroutine my_driver
"""

    # Set up the test files
    dirname = here/'test_scheduler_disable_wildcard'
    dirname.mkdir(exist_ok=True)
    modfile = dirname/'field_mod.F90'
    modfile.write_text(fcode_mod)
    testfile = dirname/'test.F90'
    testfile.write_text(fcode_driver)

    config['default']['disable'] = ['*%init']

    scheduler = Scheduler(paths=dirname, seed_routines=['my_driver'], config=config)

    expected_items = [
        '#my_driver', 'field_mod#field_init',
    ]
    expected_dependencies = [
        ('#my_driver', 'field_mod#field_init'),
    ]

    assert all(n in scheduler.items for n in expected_items)
    assert all(e in scheduler.dependencies for e in expected_dependencies)

    assert 'field_mod#field2d%init' not in scheduler.items
    assert 'field_mod#field3d%init' not in scheduler.items

    # Clean up
    try:
        modfile.unlink()
        testfile.unlink()
        dirname.rmdir()
    except FileNotFoundError:
        pass


def test_transformation_config(config):
    """
    Test the correct instantiation of :any:`Transformation` objecst from config
    """
    my_config = config.copy()
    my_config['transformations'] = {
        'DependencyTransformation': {
            'module': 'loki.transform',
            'options':
            {
                'suffix': '_rick',
                'module_suffix': '_roll',
                'replace_ignore_items': False,
            }
        }
    }
    cfg = SchedulerConfig.from_dict(my_config)
    assert cfg.transformations['DependencyTransformation']

    transformation = cfg.transformations['DependencyTransformation']
    assert isinstance(transformation, DependencyTransformation)
    assert transformation.suffix == '_rick'
    assert transformation.module_suffix == '_roll'
    assert not transformation.replace_ignore_items

    # Test for errors when failing to instantiate a transformation
    bad_config = config.copy()
    bad_config['transformations'] = {
        'DependencyTrafo': {  # <= typo
            'module': 'loki.transform',
            'options': {}
        }
    }
    with pytest.raises(RuntimeError):
        cfg = SchedulerConfig.from_dict(bad_config)

    worse_config = config.copy()
    worse_config['transformations'] = {
        'DependencyTransform': {
            'module': 'loki.transformats',  # <= typo
            'options': {}
        }
    }
    with pytest.raises(ModuleNotFoundError):
        cfg = SchedulerConfig.from_dict(worse_config)

    worst_config = config.copy()
    worst_config['transformations'] = {
        'DependencyTransform': {
            'module': 'loki.transform',
            'options': {'hello': 'Dave'}
        }
    }
    with pytest.raises(RuntimeError):
        cfg = SchedulerConfig.from_dict(worst_config)


def test_transformation_config_external_with_dimension(here, config):
    """
    Test instantiation of :any:`Transformation` from config with
    :any:`Dimension` argument.
    """
    my_config = config.copy()
    my_config['dimensions'] = {
        'ij': {'size': 'n', 'index': 'i'}
    }
    my_config['transformations'] = {
        'CallMeRick': {
            'classname': 'CallMeMaybeTrafo',
            'module': 'call_me_trafo',
            'path': str(here/'sources'),
            'options': { 'name': 'Rick', 'horizontal': '%dimensions.ij%' }
        }
    }
    cfg = SchedulerConfig.from_dict(my_config)
    assert cfg.transformations['CallMeRick']

    transformation = cfg.transformations['CallMeRick']
    # We don't have the type, so simply check the class name
    assert type(transformation).__name__ == 'CallMeMaybeTrafo'
    assert transformation.name == 'Rick'
    assert isinstance(transformation.horizontal, Dimension)
    assert transformation.horizontal.size == 'n'
    assert transformation.horizontal.index == 'i'


@pytest.mark.parametrize('item_name,keys,use_pattern_matching,match_item_parents,expected', [
    ('comp2', 'comp2', True, True, ('comp2',)),
    ('#comp2', 'comp2', True, True, ('comp2',)),
    ('comp2', '#comp2', True, True, ()),  # This is key: If the config key is provided with explicit scope,
                                          # we don't match unscoped names
    ('#comp2', '#comp2', True, True, ('#comp2',))
])
def test_scheduler_config_match_item_keys(item_name, keys, use_pattern_matching, match_item_parents, expected):
    value = SchedulerConfig.match_item_keys(item_name, keys, use_pattern_matching, match_item_parents)
    assert value == expected


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_filter_items_file_graph(frontend, config):
    """
    Ensure that the ``items`` list given to a transformation in
    a file graph traversal is filtered to include only used items
    """
    fcode = """
module test_scheduler_filter_program_units_file_graph_mod1
implicit none
contains
subroutine proc1(arg)
    integer, intent(inout) :: arg
    arg = arg + 1
end subroutine proc1

subroutine unused_proc(arg)
    integer, intent(inout) :: arg
    arg = arg - 1
end subroutine unused_proc
end module test_scheduler_filter_program_units_file_graph_mod1

module test_scheduler_filter_program_units_file_graph_mod2
implicit none
contains
subroutine proc2(arg)
    integer, intent(inout) :: arg
    arg = arg + 2
end subroutine proc2
end module test_scheduler_filter_program_units_file_graph_mod2

module test_scheduler_filter_program_units_file_graph_mod3
implicit none
integer, parameter :: param3 = 3
contains
subroutine proc3(arg)
    integer, intent(inout) :: arg
    arg = arg + 3
end subroutine proc3
end module test_scheduler_filter_program_units_file_graph_mod3

subroutine test_scheduler_filter_program_units_file_graph_driver
use test_scheduler_filter_program_units_file_graph_mod1, only: proc1
use test_scheduler_filter_program_units_file_graph_mod3, only: param3
implicit none
integer :: i
i = param3
call proc1(i)
end subroutine test_scheduler_filter_program_units_file_graph_driver
    """

    config['routines']['test_scheduler_filter_program_units_file_graph_driver'] = {
        'role': 'driver'
    }

    workdir = gettempdir()/'test_scheduler_filter_program_units_file_graph'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    filepath = workdir/'test_scheduler_filter_program_units_file_graph.F90'
    filepath.write_text(fcode)

    scheduler = Scheduler(
        paths=[workdir], config=config, seed_routines=['test_scheduler_filter_program_units_file_graph_driver'],
        frontend=frontend
    )

    # Only the driver and mod1 are in the Sgraph
    expected_dependencies = {
        '#test_scheduler_filter_program_units_file_graph_driver': {
            'test_scheduler_filter_program_units_file_graph_mod1#proc1',
            'test_scheduler_filter_program_units_file_graph_mod3'
        },
        'test_scheduler_filter_program_units_file_graph_mod1#proc1': set(),
        'test_scheduler_filter_program_units_file_graph_mod3': set()
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # The other module and procedure are in the item_factory's cache
    assert 'test_scheduler_filter_program_units_file_graph_mod2' in scheduler.item_factory.item_cache
    assert 'test_scheduler_filter_program_units_file_graph_mod1#unused_proc' in scheduler.item_factory.item_cache

    # The filegraph consists of the single file
    filegraph = scheduler.file_graph
    assert filegraph.items == (str(filepath).lower(),)

    class MyFileTrafo(Transformation):
        traverse_file_graph = True

        def transform_file(self, sourcefile, **kwargs):
            # Only active items should be passed to the transformation
            assert 'items' in kwargs
            assert set(kwargs['items']) == {
                'test_scheduler_filter_program_units_file_graph_mod1',
                'test_scheduler_filter_program_units_file_graph_mod1#proc1',
                'test_scheduler_filter_program_units_file_graph_mod3',
                '#test_scheduler_filter_program_units_file_graph_driver'
            }

    scheduler.process(transformation=MyFileTrafo())

    rmtree(workdir)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('frontend_args,defines,preprocess,has_cpp_directives,additional_dependencies', [
    # No preprocessing, thus all call dependencies are included
    (None, None, False, [
        '#test_scheduler_frontend_args1', '#test_scheduler_frontend_args2', '#test_scheduler_frontend_args4'
    ], {
        '#test_scheduler_frontend_args2': ('#test_scheduler_frontend_args3',),
        '#test_scheduler_frontend_args3': (),
        '#test_scheduler_frontend_args4': ('#test_scheduler_frontend_args3',),
    }),
    # Global preprocessing setting SOME_DEFINITION, removing dependency on 3
    (None, ['SOME_DEFINITION'], True, [], {}),
    # Global preprocessing with local definition for one file, re-adding a dependency on 3
    (
        {'test_scheduler_frontend_args/file3_4.F90': {'defines': ['SOME_DEFINITION','LOCAL_DEFINITION']}},
        ['SOME_DEFINITION'],
        True,
        [],
        {
            '#test_scheduler_frontend_args3': (),
            '#test_scheduler_frontend_args4': ('#test_scheduler_frontend_args3',),
        }
    ),
    # Global preprocessing with preprocessing switched off for 2
    (
        {'test_scheduler_frontend_args/file2.F90': {'preprocess': False}},
        ['SOME_DEFINITION'],
        True,
        ['#test_scheduler_frontend_args2'],
        {
            '#test_scheduler_frontend_args2': ('#test_scheduler_frontend_args3',),
            '#test_scheduler_frontend_args3': (),
        }
    ),
    # No preprocessing except for 2
    (
        {'test_scheduler_frontend_args/file2.F90': {'preprocess': True, 'defines': ['SOME_DEFINITION']}},
        None,
        False,
        ['#test_scheduler_frontend_args1', '#test_scheduler_frontend_args4'],
        {
            '#test_scheduler_frontend_args3': (),
            '#test_scheduler_frontend_args4': ('#test_scheduler_frontend_args3',),
        }
    ),
])
def test_scheduler_frontend_args(frontend, frontend_args, defines, preprocess,
                                 has_cpp_directives, additional_dependencies, config):
    """
    Test overwriting frontend options via Scheduler config
    """

    fcode1 = """
subroutine test_scheduler_frontend_args1
    implicit none
#ifdef SOME_DEFINITION
    call test_scheduler_frontend_args2
#endif
end subroutine test_scheduler_frontend_args1
    """.strip()

    fcode2 = """
subroutine test_scheduler_frontend_args2
    implicit none
#ifndef SOME_DEFINITION
    call test_scheduler_frontend_args3
#endif
    call test_scheduler_frontend_args4
end subroutine test_scheduler_frontend_args2
    """.strip()

    fcode3_4 = """
subroutine test_scheduler_frontend_args3
implicit none
end subroutine test_scheduler_frontend_args3

subroutine test_scheduler_frontend_args4
implicit none
#ifdef LOCAL_DEFINITION
    call test_scheduler_frontend_args3
#endif
end subroutine test_scheduler_frontend_args4
    """.strip()

    workdir = gettempdir()/'test_scheduler_frontend_args'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    (workdir/'file1.F90').write_text(fcode1)
    (workdir/'file2.F90').write_text(fcode2)
    (workdir/'file3_4.F90').write_text(fcode3_4)

    expected_dependencies = {
        '#test_scheduler_frontend_args1': ('#test_scheduler_frontend_args2',),
        '#test_scheduler_frontend_args2': ('#test_scheduler_frontend_args4',),
        '#test_scheduler_frontend_args4': (),
    }

    for key, value in additional_dependencies.items():
        expected_dependencies[key] = expected_dependencies.get(key, ()) + value

    config['frontend_args'] = frontend_args

    scheduler = Scheduler(
        paths=[workdir], config=config, seed_routines=['test_scheduler_frontend_args1'],
        frontend=frontend, defines=defines, preprocess=preprocess, xmods=[workdir]
    )

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    for item in scheduler.items:
        cpp_directives = FindNodes(PreprocessorDirective).visit(item.ir.ir)
        assert bool(cpp_directives) == (item in has_cpp_directives and frontend != OMNI)
        # NB: OMNI always does preprocessing, therefore we won't find the CPP directives
        #     after the full parse

    rmtree(workdir)


@pytest.mark.skipif(not (HAVE_OMNI and HAVE_FP), reason="OMNI or FP not available")
def test_scheduler_frontend_overwrite(config):
    """
    Test the use of a different frontend via Scheduler config
    """
    fcode_header = """
module test_scheduler_frontend_overwrite_header
    implicit none
    type some_type
        ! We have a comment
        real, dimension(:,:), pointer :: arr
    end type some_type
end module test_scheduler_frontend_overwrite_header
    """.strip()
    fcode_kernel = """
subroutine test_scheduler_frontend_overwrite_kernel
    use test_scheduler_frontend_overwrite_header, only: some_type
    implicit none
    type(some_type) :: var
end subroutine test_scheduler_frontend_overwrite_kernel
    """.strip()

    workdir = gettempdir()/'test_scheduler_frontend_overwrite'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    (workdir/'test_scheduler_frontend_overwrite_header.F90').write_text(fcode_header)
    (workdir/'test_scheduler_frontend_overwrite_kernel.F90').write_text(fcode_kernel)

    # Make sure that OMNI cannot parse the header file
    with pytest.raises(CalledProcessError):
        Sourcefile.from_source(fcode_header, frontend=OMNI, xmods=[workdir])

    # ...and that the problem exists also during Scheduler traversal
    with pytest.raises(CalledProcessError):
        Scheduler(
            paths=[workdir], config=config, seed_routines=['test_scheduler_frontend_overwrite_kernel'],
            frontend=OMNI, xmods=[workdir]
        )

    # Strip the comment from the header file and parse again to generate an xmod
    fcode_header_lines = fcode_header.split('\n')
    Sourcefile.from_source('\n'.join(fcode_header_lines[:3] + fcode_header_lines[4:]), frontend=OMNI, xmods=[workdir])

    # Setup the config with the frontend overwrite
    config['frontend_args'] = {
        'test_scheduler_frontend_overwrite_header.F90': {'frontend': 'FP'}
    }

    # ...and now it works fine
    scheduler = Scheduler(
        paths=[workdir], config=config, seed_routines=['test_scheduler_frontend_overwrite_kernel'],
        frontend=OMNI, xmods=[workdir]
    )

    assert set(scheduler.items) == {
        '#test_scheduler_frontend_overwrite_kernel', 'test_scheduler_frontend_overwrite_header#some_type'
    }

    assert set(scheduler.dependencies) == {
       ('#test_scheduler_frontend_overwrite_kernel', 'test_scheduler_frontend_overwrite_header#some_type')
    }

    # ...and the derived type has it's comment
    comments = FindNodes(Comment).visit(scheduler['test_scheduler_frontend_overwrite_header#some_type'].ir.body)
    assert len(comments) == 1
    assert comments[0].text == '! We have a comment'

    rmtree(workdir)


def test_scheduler_pipeline_simple(here, config, frontend):
    """
    Test processing a :any:`Pipeline` over a simple call-tree.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |
                           | --> another_l1 -> another_l2
    """
    projA = here/'sources/projA'

    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines='driverA', frontend=frontend
    )

    class ZeroMyStuffTrafo(Transformation):
        """ Fill each argument array with 0.0 """

        def transform_subroutine(self, routine, **kwargs):
            for v in routine.variables:
                if isinstance(v, Array):
                    routine.body.append(Assignment(lhs=v, rhs=Literal(0.0)))

    class AddSnarkTrafo(Transformation):
        """ Add a snarky comment to the zeroing """

        def __init__(self, name='Rick'):
            self.name = name

        def transform_subroutine(self, routine, **kwargs):
            routine.body.append(Comment(text=''))  # Add a newline
            routine.body.append(Comment(text=f'! Sorry {self.name}, no values for you!'))

    def has_correct_assigns(routine, num_assign, values=None):
        assigns = FindNodes(Assignment).visit(routine.body)
        values = values or [0.0]
        return len(assigns) == num_assign and all(a.rhs in values for a in assigns)

    def has_correct_comments(routine, name='Dave'):
        text = f'! Sorry {name}, no values for you!'
        comments = FindNodes(Comment).visit(routine.body)
        return len(comments) > 2 and comments[-1].text == text

    # First apply in sequence and check effect
    scheduler.process(transformation=ZeroMyStuffTrafo())
    assert has_correct_assigns(scheduler['drivera_mod#drivera'].ir, 0)
    assert has_correct_assigns(scheduler['kernela_mod#kernela'].ir, 2)
    assert has_correct_assigns(scheduler['compute_l1_mod#compute_l1'].ir, 1)
    assert has_correct_assigns(scheduler['compute_l2_mod#compute_l2'].ir, 2, values=[66.0, 00])
    assert has_correct_assigns(scheduler['#another_l1'].ir, 1)
    assert has_correct_assigns(scheduler['#another_l2'].ir, 2, values=[77.0, 00])

    scheduler.process(transformation=AddSnarkTrafo(name='Dave'))
    assert has_correct_comments(scheduler['drivera_mod#drivera'].ir)
    assert has_correct_comments(scheduler['kernela_mod#kernela'].ir)
    assert has_correct_comments(scheduler['compute_l1_mod#compute_l1'].ir)
    assert has_correct_comments(scheduler['compute_l2_mod#compute_l2'].ir)
    assert has_correct_comments(scheduler['#another_l1'].ir)
    assert has_correct_comments(scheduler['#another_l2'].ir)

    # Rebuild the scheduler to wipe the previous result
    scheduler = Scheduler(
        paths=projA, includes=projA/'include', config=config,
        seed_routines='driverA', frontend=frontend
    )

    # Then apply as a simple pipeline and check again
    MyPipeline = partial(Pipeline, classes=(ZeroMyStuffTrafo, AddSnarkTrafo))
    scheduler.process(transformation=MyPipeline(name='Chad'))
    assert has_correct_assigns(scheduler['drivera_mod#drivera'].ir, 0)
    assert has_correct_assigns(scheduler['kernela_mod#kernela'].ir, 2)
    assert has_correct_assigns(scheduler['compute_l1_mod#compute_l1'].ir, 1)
    assert has_correct_assigns(scheduler['compute_l2_mod#compute_l2'].ir, 2, values=[66.0, 00])
    assert has_correct_assigns(scheduler['#another_l1'].ir, 1)
    assert has_correct_assigns(scheduler['#another_l2'].ir, 2, values=[77.0, 00])

    assert has_correct_comments(scheduler['drivera_mod#drivera'].ir, name='Chad')
    assert has_correct_comments(scheduler['kernela_mod#kernela'].ir, name='Chad')
    assert has_correct_comments(scheduler['compute_l1_mod#compute_l1'].ir, name='Chad')
    assert has_correct_comments(scheduler['compute_l2_mod#compute_l2'].ir, name='Chad')
    assert has_correct_comments(scheduler['#another_l1'].ir, name='Chad')
    assert has_correct_comments(scheduler['#another_l2'].ir, name='Chad')


def test_pipeline_config_compose(config):
    """
    Test the correct instantiation of a custom :any:`Pipeline`
    object from config.
    """
    my_config = config.copy()
    my_config['dimensions'] = {
        'horizontal': { 'size': 'KLON', 'index': 'JL', 'bounds': ['KIDIA', 'KFDIA'] },
        'vertical': { 'size': 'KLEV', 'index': 'JK' },
        'block_dim': { 'size': 'NGPBLKS', 'index': 'IBL' },
    }
    my_config['transformations'] = {
        'VectorWithTrim': {
            'classname': 'SCCVectorPipeline',
            'module': 'transformations.single_column_coalesced',
            'options':
            {
                'horizontal': '%dimensions.horizontal%',
                'vertical': '%dimensions.vertical%',
                'block_dim': '%dimensions.block_dim%',
                'directive': 'openacc',
                'trim_vector_sections': True,
            },
        },
        'preprocess': {
            'classname': 'RemoveCallsTransformation',
            'module': 'transformations.utility_routines',
            'options': {
                'routines': 'dr_hook',
                'include_intrinsics': True
            }
        },
        'postprocess': {
            'classname': 'ModuleWrapTransformation',
            'module': 'loki.transform',
            'options': { 'module_suffix': '_module' }
        }
    }
    my_config['pipelines'] = {
        'MyVectorPipeline': {
            'transformations': [
                'preprocess',
                'VectorWithTrim',
                'postprocess',
            ],
        }
    }
    cfg = SchedulerConfig.from_dict(my_config)

    # Check that transformations and pipelines were created correctly
    assert cfg.transformations['VectorWithTrim']
    assert cfg.transformations['preprocess']
    assert cfg.transformations['postprocess']

    assert cfg.pipelines['MyVectorPipeline']
    pipeline = cfg.pipelines['MyVectorPipeline']
    assert isinstance(pipeline, Pipeline)

    # Check that the pipeline is correctly composed
    assert len(pipeline.transformations) == 7
    assert type(pipeline.transformations[0]).__name__ == 'RemoveCallsTransformation'
    assert type(pipeline.transformations[1]).__name__ == 'SCCBaseTransformation'
    assert type(pipeline.transformations[2]).__name__ == 'SCCDevectorTransformation'
    assert type(pipeline.transformations[3]).__name__ == 'SCCDemoteTransformation'
    assert type(pipeline.transformations[4]).__name__ == 'SCCRevectorTransformation'
    assert type(pipeline.transformations[5]).__name__ == 'SCCAnnotateTransformation'
    assert type(pipeline.transformations[6]).__name__ == 'ModuleWrapTransformation'

    # Check for some specified and default constructor flags
    assert pipeline.transformations[0].include_intrinsics is True
    assert isinstance(pipeline.transformations[1].horizontal, Dimension)
    assert pipeline.transformations[1].horizontal.size == 'KLON'
    assert pipeline.transformations[1].horizontal.index == 'JL'
    assert pipeline.transformations[1].directive == 'openacc'
    assert pipeline.transformations[2].trim_vector_sections is True
    assert isinstance(pipeline.transformations[5].vertical, Dimension)
    assert pipeline.transformations[5].vertical.size == 'KLEV'
    assert pipeline.transformations[5].vertical.index == 'JK'
    assert pipeline.transformations[6].replace_ignore_items is True
