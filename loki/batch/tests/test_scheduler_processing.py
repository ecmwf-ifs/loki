# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import deque

import pytest

from loki import (
    Function, Item, Module, Sourcefile, Subroutine, as_tuple, flatten,
    graphviz_present
)
from loki.batch import (
    InterfaceItem, ModuleItem, ProcedureBindingItem, ProcedureItem,
    ProcessingStrategy, Scheduler, SchedulerConfig,
    Transformation, TransformationError, TypeDefItem
)
from loki.frontend import HAVE_FP, available_frontends
from loki.ir import nodes as ir
from loki.transformations import DependencyTransformation, ModuleWrapTransformation

from .conftest import VisGraphWrapper

pytestmark = pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')


@pytest.mark.parametrize('seed', ['compute_l1', 'compute_l1_mod#compute_l1'])
def test_scheduler_process(testdir, config, frontend, seed, tmp_path):
    """
    Create a simple task graph from a single sub-project
    and apply a simple transformation to it.

    projA: driverA -> kernelA -> compute_l1 -> compute_l2
                           |      <driver>      <kernel>
                           |
                           | --> another_l1 -> another_l2
                                  <driver>      <kernel>
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

    class RoleComment(Transformation):
        """
        Simply add role as a comment in the subroutine body.
        """

        def transform_subroutine(self, routine, **kwargs):
            role = kwargs.get('role', None)
            routine.body.prepend(ir.Comment(f'! {role}'))

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
        assert isinstance(comment, ir.Comment)
        assert comment.text == f'! {role}'


@pytest.mark.skipif(not graphviz_present(), reason='Graphviz is not installed')
@pytest.mark.parametrize('seed', ['driverE_single', 'driverE_mod#driverE_single'])
def test_scheduler_process_filter(testdir, config, frontend, seed, tmp_path):
    """
    Applies simple kernels over complex callgraphs to check that we
    only apply to the entities requested and only once!

    projA: driverE_single -> kernelE -> compute_l1 -> compute_l2
                              |
                              | --> ghost_busters
    """
    projA = testdir/'sources/projA'
    projB = testdir/'sources/projB'

    config['routines'] = {
        seed: {'role': 'driver', 'expand': True},
    }

    scheduler = Scheduler(
        paths=[projA, projB], includes=projA/'include', config=config, frontend=frontend, xmods=[tmp_path]
    )

    class XMarksTheSpot(Transformation):
        """
        Prepend an 'X' comment to a given :any:`Subroutine`
        """

        def transform_subroutine(self, routine, **kwargs):
            routine.body.prepend(ir.Comment('! X'))

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
            item_ir = item.ir
        else:
            assert item is None
            scope_name, local_name = key.split('#')
            # key should not be found in the callgraph but scope should still exist in the
            # item_cache because the file has been indexed
            assert scope_name in scheduler.item_factory.item_cache
            item_ir = scheduler.item_factory.item_cache[scope_name].ir[local_name]
        first_node = item_ir.body.body[0]
        first_node_is_x = isinstance(first_node, ir.Comment) and first_node.text == '! X'
        assert first_node_is_x == is_transformed


@pytest.mark.parametrize('strict', [True, False])
def test_scheduler_graph_multiple_separate_enrich_fail(testdir, config, frontend, strict, tmp_path):
    """
    Tests that explicit enrichment in "strict" mode will fail because it can't
    find ext_driver

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
                     <ext_driver>

    projB:            ext_driver -> ext_kernelfail
    """
    projA = testdir/'sources/projA'

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
        seed_routines=['driverB'], frontend=frontend, xmods=[tmp_path]
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


@pytest.mark.parametrize('preprocess', [False, True])
def test_scheduler_dependencies_ignore(tmp_path, testdir, preprocess, frontend):
    projA = testdir/'sources/projA'
    projB = testdir/'sources/projB'

    # NB: With preprocessing, ext_driver is no longer
    #     wrapped inside a module but instead imported
    #     via an intfb.h

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
        frontend=frontend, preprocess=preprocess, xmods=[tmp_path]
    )

    schedulerB = Scheduler(
        paths=projB, includes=projB/'include', config=configB,
        frontend=frontend, preprocess=preprocess, xmods=[tmp_path]
    )

    expected_items_a = [
        'driverB_mod#driverB', 'kernelB_mod#kernelB',
        'compute_l1_mod#compute_l1', 'compute_l2_mod#compute_l2',
        'header_mod', 'header_mod#header_type'
    ]
    if preprocess:
        expected_items_b = ['#ext_driver', 'ext_kernel_mod', 'ext_kernel_mod#ext_kernel']
    else:
        expected_items_b = ['ext_driver_mod#ext_driver', 'ext_kernel_mod', 'ext_kernel_mod#ext_kernel']

    assert set(schedulerA.items) == {n.lower() for n in expected_items_a + expected_items_b}
    assert all(not schedulerA[name].is_ignored for name in expected_items_a)
    assert all(schedulerA[name].is_ignored for name in expected_items_b)

    assert set(schedulerB.items) == {n.lower() for n in expected_items_b}

    cg_path = tmp_path/'callgraph_dependencies_ignore'
    fg_path = cg_path.with_name(cg_path.name + '_file_graph')
    schedulerA.callgraph(cg_path, with_file_graph=True)

    # Testing of callgraph visualisation
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

    transformations = (
        ModuleWrapTransformation(module_suffix='_mod'),
        DependencyTransformation(suffix='_test', module_suffix='_mod')
    )
    for transformation in transformations:
        # Apply dependency injection transformation and ensure only the root driver is not transformed
        schedulerA.process(transformation)

    assert schedulerA.items[0].source.all_subroutines[0].name == 'driverB'
    assert schedulerA.items[1].source.all_subroutines[0].name == 'kernelB_test'
    assert schedulerA.items[4].source.all_subroutines[0].name == 'compute_l1_test'
    assert schedulerA.items[5].source.all_subroutines[0].name == 'compute_l2_test'

    # Note that 'ext_driver' and 'ext_kernel' are no longer part of the dependency graph because the
    # renaming makes it impossible to discover the non-transformed routines
    assert all(name not in schedulerA for name in expected_items_b)
    assert 'ext_driver_test_mod#ext_driver_test' not in schedulerA

    for transformation in transformations:
        # For the second target lib, we want the driver to be converted
        schedulerB.process(transformation=transformation)
    for transformation in transformations:
        # Repeat processing to ensure DependencyTransform is idempotent
        schedulerB.process(transformation=transformation)

    assert schedulerB.items[0].source.all_subroutines[0].name == 'ext_driver_test'
    # This is the untransformed original module
    assert schedulerB['ext_kernel_mod'].source.all_subroutines[0].name == 'ext_kernel'
    # This is the module-wrapped procedure
    assert schedulerB['ext_kernel_test_mod#ext_kernel_test'].source.all_subroutines[0].name == 'ext_kernel_test'


@pytest.mark.parametrize('use_file_graph', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_scheduler_traversal_order(tmp_path, testdir, config, frontend, use_file_graph, reverse):
    """
    Test correct traversal order for scheduler processing
    """
    proj = testdir/'sources/projHoist'

    scheduler = Scheduler(
        paths=proj, seed_routines=['driver'], config=config,
        full_parse=True, frontend=frontend, xmods=[tmp_path]
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
@pytest.mark.parametrize('ignore_internal_procedures', [True, False])
@pytest.mark.parametrize('ignore_internal_procedures_driver', [None, True, False])
def test_scheduler_member_routines(tmp_path, config, frontend, use_file_graph, reverse,
                                   ignore_internal_procedures, ignore_internal_procedures_driver):
    """
    Make sure that transformation processing works also for contained member routines
    if enabled in the config. This includes internal procedures in module routines as well
    as free routines, and selective config overwrites to allow for fine-grained control of this behaviour
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
        val = my_func(val)
        call kernel
    contains
        subroutine my_member
            call my_routine(val)
        end subroutine my_member
        integer function my_func(val0)
           integer, intent(in) :: val0
           my_func = val0 + 1
        end function
    end subroutine driver
end module member_mod
    """.strip()

    fcode_kernel = """
subroutine kernel
    implicit none
    integer :: val
    call my_member
    write(*,*) val
contains
    subroutine my_member
        val = 1
    end subroutine my_member
end subroutine kernel
    """.strip()

    (tmp_path/'member_mod.F90').write_text(fcode_mod)
    (tmp_path/'kernel.F90').write_text(fcode_kernel)

    config['default']['ignore_internal_procedures'] = ignore_internal_procedures
    if ignore_internal_procedures_driver is not None:
        config['routines']['member_mod#driver'] = {'ignore_internal_procedures': ignore_internal_procedures_driver}

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['member_mod#driver'],
        frontend=frontend, xmods=[tmp_path]
    )

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
        expected = [
            f'{tmp_path/"member_mod.F90"!s}'.lower() + '::member_mod.F90',
            f'{tmp_path/"kernel.F90"!s}'.lower() + '::kernel.F90'
        ]
    else:
        expected = ['member_mod#driver::driver']
        expected_dependencies_driver = ['kernel']

        # Slightly awkward logic to capture the cases that
        #   1) we include internal procedures without any special config overrides for the driver
        #   2) we override the config for the driver to include internal procedures
        #   3) we include internal procedures but disable it for the driver
        # and mark the corresponding excpected dependencies in case 1 and 2
        include_driver_internals = (
            (ignore_internal_procedures_driver is None and not ignore_internal_procedures) or
            ignore_internal_procedures_driver is False
        )

        if include_driver_internals:
            expected += ['member_mod#driver#my_member::my_member', '#kernel::kernel',
                         'member_mod#driver#my_func::my_func']
            expected_dependencies_driver = ['my_member', *expected_dependencies_driver, 'my_func']
        else:
            expected += ['#kernel::kernel']
        expected_dependencies_kernel = []

        if include_driver_internals:
            expected += ['member_mod#my_routine::my_routine']

        if not ignore_internal_procedures:
            expected += ['#kernel#my_member::my_member']
            expected_dependencies_kernel += ['my_member']

        assert [dep.name for dep in scheduler['member_mod#driver'].dependencies] == expected_dependencies_driver
        assert [dep.name for dep in scheduler['#kernel'].dependencies] == expected_dependencies_kernel

    if reverse:
        expected = expected[::-1]

    assert transformation.record == flatten(expected)


def test_scheduler_item_successors(testdir, config, frontend, tmp_path):
    """
    Test that scheduler.item_successors always returns the original item.
    """
    my_config = config.copy()
    my_config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(
        paths=testdir/'sources/projInlineCalls', config=my_config, frontend=frontend, xmods=[tmp_path]
    )
    import_item = scheduler['vars_module']
    driver_item = scheduler['#driver']
    kernel_item = scheduler['#double_real']

    id_a = id(import_item)

    for successor in scheduler.sgraph.successors(driver_item):
        if successor.name == import_item.name:
            assert id(successor) == id_a
    for successor in scheduler.sgraph.successors(kernel_item):
        if successor.name == import_item.name:
            assert id(successor) == id_a


@pytest.mark.parametrize('trafo_item_filter', [
    Item,
    ProcedureItem,
    (ProcedureItem, InterfaceItem, ProcedureBindingItem),
    (ProcedureItem, TypeDefItem),
])
def test_scheduler_successors(tmp_path, config, trafo_item_filter):
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
        'some_mod#routine': ('some_mod#some_type', 'some_mod#some_type%other'),
        'some_mod#other': ('some_mod#some_type',),
        'some_mod#some_type%procedure': ('some_mod#some_procedure',),
        'some_mod#some_procedure': ('some_mod#some_type',)
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

            sub_sgraph = kwargs.get('sub_sgraph')
            successors = as_tuple(sub_sgraph.successors(item))
            assert set(successors) == set(self.expected_successors[item.name])

    (tmp_path/'some_mod.F90').write_text(fcode_mod)
    (tmp_path/'caller.F90').write_text(fcode)

    scheduler = Scheduler(paths=[tmp_path], config=config, seed_routines=['caller'], xmods=[tmp_path])

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    if trafo_item_filter != Item:
        item_filter = as_tuple(trafo_item_filter)
        if ProcedureItem in item_filter:
            item_filter += (ProcedureBindingItem, InterfaceItem)

        # Filter expected dependencies
        expected_dependencies = {
            item_name: tuple(
                dep for dep in dependencies
                if isinstance(scheduler[dep], item_filter)
            )
            for item_name, dependencies in expected_dependencies.items()
        }

    expected_successors = {}
    for item_name, dependencies in expected_dependencies.items():
        item_successors = set()
        dep_queue = deque(dependencies)
        # Add dependency-dependencies for the full successor list
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


@pytest.mark.parametrize('frontend', available_frontends())
def test_scheduler_filter_items_file_graph(tmp_path, frontend, config):
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

    filepath = tmp_path/'test_scheduler_filter_program_units_file_graph.F90'
    filepath.write_text(fcode)

    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines=['test_scheduler_filter_program_units_file_graph_driver'],
        frontend=frontend, xmods=[tmp_path]
    )

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

    filegraph = scheduler.file_graph
    # The filegraph consists of the single file
    assert filegraph.items == (str(filepath).lower(),)

    class MyFileTrafo(Transformation):
        traverse_file_graph = True

        def transform_file(self, sourcefile, **kwargs):
            assert 'items' in kwargs
            # Only active items should be passed to the transformation
            assert set(kwargs['items']) == {
                'test_scheduler_filter_program_units_file_graph_mod1',
                'test_scheduler_filter_program_units_file_graph_mod1#proc1',
                'test_scheduler_filter_program_units_file_graph_mod3',
                '#test_scheduler_filter_program_units_file_graph_driver'
            }

    scheduler.process(transformation=MyFileTrafo())


def test_scheduler_cgraph_preserves_original_state_after_rename(tmp_path, config, frontend):
    """
    Ensure scheduler.cgraph keeps the original code state after item-renaming transformations.
    """
    fcode = """
subroutine cgraph_original_kernel
end subroutine cgraph_original_kernel
    """.strip()
    filepath = tmp_path/'cgraph_original_kernel.F90'
    filepath.write_text(fcode)

    config['default']['role'] = 'kernel'
    scheduler = Scheduler(
        paths=[tmp_path], config=config, seed_routines='cgraph_original_kernel',
        frontend=frontend, xmods=[tmp_path]
    )

    assert '#cgraph_original_kernel' in scheduler.cgraph.items

    scheduler.process(ModuleWrapTransformation(module_suffix='_mod'))

    assert '#cgraph_original_kernel' in scheduler.cgraph.items
    assert 'cgraph_original_kernel_mod#cgraph_original_kernel' not in scheduler.cgraph.items
    assert 'cgraph_original_kernel_mod#cgraph_original_kernel' in scheduler.item_factory.item_cache


@pytest.mark.parametrize('proc_strategy', [ProcessingStrategy.PLAN, ProcessingStrategy.DEFAULT])
@pytest.mark.parametrize('with_filegraph', [True, False])
def test_scheduler_exception_handling(tmp_path, testdir, config, frontend, proc_strategy, with_filegraph):
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
        seed_routines='driverA', frontend=frontend, xmods=[tmp_path]
    )

    class RuntimeErrorTransformation(Transformation):
        traverse_file_graph = with_filegraph
        recurse_to_modules = with_filegraph
        recurse_to_procedures = with_filegraph

        item_filter = (ProcedureItem, ModuleItem)

        def __init__(self, fail_cls, fail_name):
            self.fail_cls = fail_cls
            self.fail_name = fail_name

        def transform_file(self, sourcefile, **kwargs):
            if self.fail_cls == Sourcefile:
                raise RuntimeError

        plan_file = transform_file

        def transform_module(self, module, **kwargs):
            if self.fail_cls == Module and self.fail_name == module.name.lower():
                raise RuntimeError

        plan_module = transform_module

        def transform_subroutine(self, routine, **kwargs):
            if self.fail_cls in (Subroutine, Function) and self.fail_name == routine.name.lower():
                raise RuntimeError

        plan_subroutine = transform_subroutine

    fail_list = [
        (Subroutine, 'compute_l1'),
        (Module, 'header_mod')
    ]
    if with_filegraph:
        fail_list += [(Sourcefile, '')]

    for fail_cls, fail_name in fail_list:
        message_pattern = f'RuntimeErrorTransformation.*?{fail_cls.__name__}.*?{fail_name.lower()}'

        with pytest.raises(TransformationError, match=message_pattern):
            scheduler.process(
                RuntimeErrorTransformation(fail_cls=fail_cls, fail_name=fail_name),
                proc_strategy=proc_strategy
            )
