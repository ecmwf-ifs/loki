# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
A selection of tests for (proof-of-concept) transformations changing
dependencies through e.g., duplicating or removing kernels (and calls).
"""
import re
from pathlib import Path
import pytest

from loki.batch import Pipeline, ProcedureItem, ModuleItem, TransformationError
from loki import (
    Scheduler, SchedulerConfig, ProcessingStrategy
)
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes
from loki.tools import as_tuple
from loki.transformations.dependency import (
    CreateEntryPointsTransformation, DuplicateKernel, RemoveKernel
)
from loki.transformations.build_system import CMakePlanTransformation, FileWriteTransformation


@pytest.fixture(name='fcode_entry_point')
def fixture_fcode_entry_point(tmp_path):
    """
    An external entry-point routine with a module procedure and an external callee,
    whose interface is provided via a C-style header import (as in the IFS).
    """

    (tmp_path/'helper_mod.F90').write_text('''
module helper_mod
contains
  subroutine helper(n, field)
    integer, intent(in) :: n
    real, intent(inout) :: field(n)
    field = field + 1.
  end subroutine helper
end module helper_mod
''')

    (tmp_path/'ext.F90').write_text('''
subroutine ext(n)
  integer, intent(in) :: n
end subroutine ext
''')

    (tmp_path/'target.F90').write_text('''
subroutine target(n, field, flag)
  use helper_mod, only: helper
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: field(n)
  logical, intent(in) :: flag
#include "ext.intfb.h"
  call helper(n, field)
  call ext(n)
end subroutine target
''')


@pytest.fixture(name='entry_point_config')
def fixture_entry_point_config():
    return {
        'default': {'role': 'kernel', 'expand': True, 'strict': True, 'mode': 'gpu', 'lib': 'lib'},
        'routines': {
            'target': {'entry-point': True, 'condition': 'flag'},
        },
    }


@pytest.mark.usefixtures('fcode_entry_point')
@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Interface header not available')]))
def test_create_entry_point(frontend, tmp_path, entry_point_config):
    """Create the dispatch layer and mode-specific driver variants for an external routine."""

    scheduler = Scheduler(paths=tmp_path, config=entry_point_config, frontend=frontend, xmods=[tmp_path])
    assert scheduler.seeds == ('target',)

    scheduler.process(CreateEntryPointsTransformation(mode='gpu', baseline_mode='baseline'))

    drivers = {
        'target_loki_gpu_mod#target_loki_gpu': 'gpu',
        'target_loki_baseline_mod#target_loki_baseline': 'baseline',
    }
    items = {'#target', 'helper_mod#helper', '#ext', *drivers}
    assert scheduler.seeds == ('target', *drivers)
    assert {item.name for item in scheduler.items} == items

    # The dispatch layer retains the original item and has no children
    dispatcher = scheduler['#target']
    assert (dispatcher.role, dispatcher.mode, dispatcher.expand, dispatcher.replicate) == (
        'driver', 'baseline', False, False
    )
    assert dispatcher.targets == ()
    assert not scheduler.sgraph.successors(dispatcher)
    assert set(dispatcher.block) == {'target_loki_gpu_mod', 'target_loki_baseline_mod', *drivers}

    # The driver variants are module-wrapped in new files and call the original dependencies
    for name, mode in drivers.items():
        driver = scheduler[name]
        assert (driver.role, driver.mode, driver.expand) == ('driver', mode, True)
        assert driver.path == tmp_path/f'{driver.scope_name}.F90'
        assert driver.orig_path == tmp_path/'target.F90'
        assert {item.name for item in scheduler.sgraph.successors(driver)} == {'helper_mod#helper', '#ext'}
        assert [call.name for call in FindNodes(ir.CallStatement).visit(driver.ir.body)] == ['helper', 'ext']
        assert [imprt.module for imprt in driver.ir.imports] == ['helper_mod', 'ext.intfb.h']

    # Entry-point selection data is removed and the scheduler config is updated
    for name in ('#target', *drivers):
        assert not {'entry-point', 'condition'} & set(scheduler[name].config)
        assert scheduler.config.create_item_config(name)['mode'] == scheduler[name].mode

    # The dispatch layer calls the driver variants depending on the condition and
    # imports only the driver variants
    routine = dispatcher.ir
    conditional, = FindNodes(ir.Conditional).visit(routine.body)
    assert conditional.condition == 'flag'
    calls = FindNodes(ir.CallStatement).visit(conditional)
    assert [call.name for call in calls] == ['target_loki_gpu', 'target_loki_baseline']
    assert all(call.arguments == ('n', 'field', 'flag') for call in calls)
    assert all(call.arguments[1].dimensions == () for call in calls)
    assert routine.arguments[1].dimensions == ('n',)
    assert [(imprt.module, imprt.symbols) for imprt in routine.imports] == [
        ('target_loki_gpu_mod', ('target_loki_gpu',)),
        ('target_loki_baseline_mod', ('target_loki_baseline',)),
    ]

    # Removing the entry-point configuration makes repeated application idempotent
    scheduler.process(CreateEntryPointsTransformation(mode='gpu', baseline_mode='baseline'))
    assert scheduler.seeds == ('target', *drivers)
    assert {item.name for item in scheduler.items} == items


@pytest.mark.usefixtures('fcode_entry_point')
@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Interface header not available')]))
@pytest.mark.parametrize('proc_strategy', [ProcessingStrategy.DEFAULT, ProcessingStrategy.PLAN])
def test_create_entry_point_pipelines(frontend, tmp_path, entry_point_config, proc_strategy):
    """
    Separate the call trees of the driver variants via mode-specific pipelines,
    and write or plan the generated files.
    """

    output_dir = tmp_path/'build'
    output_dir.mkdir()
    scheduler = Scheduler(
        paths=tmp_path, config=entry_point_config, frontend=frontend, xmods=[tmp_path],
        output_dir=output_dir, full_parse=proc_strategy == ProcessingStrategy.DEFAULT
    )

    pipelines = {
        'gpu': Pipeline(classes=(CreateEntryPointsTransformation,), mode='gpu', baseline_mode='baseline'),
        'baseline': Pipeline(),
    }
    scheduler.process(pipelines, proc_strategy=proc_strategy)

    # Each driver variant calls its own mode-specific copy of the dependencies
    expected_item_modes = {
        '#target': 'baseline',
        'target_loki_gpu_mod#target_loki_gpu': 'gpu',
        'helper_loki_gpu_mod#helper_loki_gpu': 'gpu',
        '#ext_loki_gpu': 'gpu',
        'target_loki_baseline_mod#target_loki_baseline': 'baseline',
        'helper_loki_baseline_mod#helper_loki_baseline': 'baseline',
        '#ext_loki_baseline': 'baseline',
    }
    assert {item.name: item.mode for item in scheduler.items} == expected_item_modes

    for mode in ('gpu', 'baseline'):
        driver = scheduler[f'target_loki_{mode}_mod#target_loki_{mode}']
        assert {item.name for item in scheduler.sgraph.successors(driver)} == {
            f'helper_loki_{mode}_mod#helper_loki_{mode}', f'#ext_loki_{mode}'
        }
        if proc_strategy == ProcessingStrategy.DEFAULT:
            assert [call.name for call in FindNodes(ir.CallStatement).visit(driver.ir.body)] == [
                f'helper_loki_{mode}', f'ext_loki_{mode}'
            ]
            assert [imprt.module for imprt in driver.ir.imports] == [
                f'helper_loki_{mode}_mod', f'ext_loki_{mode}.intfb.h'
            ]

    # Mapping of original source files to the generated files
    expected_files = {
        ('target.F90', 'target.baseline.F90'),
        ('target.F90', 'target_loki_gpu_mod.gpu.F90'),
        ('helper_mod.F90', 'helper_loki_gpu_mod.gpu.F90'),
        ('ext.F90', 'ext_loki_gpu.gpu.F90'),
        ('target.F90', 'target_loki_baseline_mod.baseline.F90'),
        ('helper_mod.F90', 'helper_loki_baseline_mod.baseline.F90'),
        ('ext.F90', 'ext_loki_baseline.baseline.F90'),
    }

    scheduler.process(FileWriteTransformation(), proc_strategy=proc_strategy)

    if proc_strategy == ProcessingStrategy.DEFAULT:
        written_files = {path.name for path in output_dir.iterdir()}
        assert written_files == {generated for _, generated in expected_files}
    else:
        plan = CMakePlanTransformation()
        scheduler.process(plan, proc_strategy=proc_strategy)

        # Transformed and appended sources are aligned to encode the source-to-source mapping
        sources_to_transform = plan.sources_to_transform['lib']
        sources_to_append = plan.sources_to_append['lib']
        assert len(sources_to_transform) == len(sources_to_append)
        planned_files = {
            (source.name, generated.name)
            for source, generated in zip(sources_to_transform, sources_to_append)
        }
        assert planned_files == expected_files
        assert plan.sources_to_remove['lib'] == [tmp_path/'target.F90']


FCODE_ENTRY_POINT_INVALID = {
    'subroutine': '''
subroutine target(flag)
  logical, intent(in) :: flag
end subroutine target
''',
    'function': '''
function target(flag)
  logical, intent(in) :: flag
  logical :: target
  target = flag
end function target
''',
    'module': '''
module entry_mod
  logical :: flag
contains
  subroutine target(a)
    integer, intent(inout) :: a
  end subroutine target
end module entry_mod
''',
}


@pytest.mark.parametrize('proc_strategy', [ProcessingStrategy.PLAN, ProcessingStrategy.DEFAULT])
@pytest.mark.parametrize('fcode,name,condition,match', (
    ('subroutine', 'target', None, 'non-empty string condition'),
    ('subroutine', 'target', '', 'non-empty string condition'),
    ('subroutine', 'target', 'flag .and.', 'Failed to parse'),
    ('function', 'target', 'flag', 'not a function'),
    ('module', 'entry_mod#target', 'flag', 'not a module procedure'),
))
def test_create_entry_point_invalid(fcode, name, condition, match, proc_strategy, tmp_path):
    """
    Reject invalid entry-point configurations, functions and module procedures.
    Conditions are parsed only during transformation.
    """

    (tmp_path/'source.F90').write_text(FCODE_ENTRY_POINT_INVALID[fcode])
    entry_config = {'entry-point': True}
    if condition is not None:
        entry_config['condition'] = condition

    scheduler = Scheduler(
        paths=tmp_path, config={'default': {'strict': True}, 'routines': {name: entry_config}},
        full_parse=proc_strategy == ProcessingStrategy.DEFAULT
    )
    transformation = CreateEntryPointsTransformation(mode='gpu', baseline_mode='baseline')

    if proc_strategy == ProcessingStrategy.PLAN and match == 'Failed to parse':
        scheduler.process(transformation, proc_strategy=proc_strategy)
    else:
        with pytest.raises(TransformationError, match=match):
            scheduler.process(transformation, proc_strategy=proc_strategy)


@pytest.mark.parametrize('mode,baseline_mode', (('gpu', 'gpu'), ('gpu-a', 'gpu_a')))
def test_create_entry_point_identical_modes(mode, baseline_mode):
    """Reject modes that would yield identical driver variant names."""

    with pytest.raises(ValueError, match='distinct'):
        CreateEntryPointsTransformation(mode=mode, baseline_mode=baseline_mode)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


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
            'strict': False,
        },
        'routines': {
            'driver': {
                'role': 'driver',
                'expand': True,
            },
        }
    }


@pytest.fixture(name='fcode_as_module')
def fixture_fcode_as_module(tmp_path):
    fcode_driver = """
subroutine driver(NLON, NB, FIELD1)
    use kernel_mod, only: kernel
    implicit none
    INTEGER, INTENT(IN) :: NLON, NB
    integer :: b
    integer, intent(inout) :: field1(nlon, nb)
    integer :: local_nlon
    local_nlon = nlon
    do b=1,nb
        call kernel(local_nlon, field1(:,b))
    end do
end subroutine driver
    """.strip()
    fcode_kernel = """
module kernel_mod
    implicit none
contains
    subroutine kernel(klon, field1)
        implicit none
        integer, intent(in) :: klon
        integer, intent(inout) :: field1(klon)
        integer :: tmp1(klon)
        integer :: jl

        do jl=1,klon
            tmp1(jl) = 0
            field1(jl) = tmp1(jl)
        end do

    end subroutine kernel
end module kernel_mod
    """.strip()
    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel_mod.F90').write_text(fcode_kernel)


@pytest.fixture(name='fcode_no_module')
def fixture_fcode_no_module(tmp_path):
    fcode_driver = """
subroutine driver(NLON, NB, FIELD1)
    implicit none
    INTEGER, INTENT(IN) :: NLON, NB
    integer :: b
    integer, intent(inout) :: field1(nlon, nb)
    integer :: local_nlon
    local_nlon = nlon
    do b=1,nb
        call kernel(local_nlon, field1(:,b))
    end do
end subroutine driver
    """.strip()
    fcode_kernel = """
subroutine kernel(klon, field1)
    implicit none
    integer, intent(in) :: klon
    integer, intent(inout) :: field1(klon)
    integer :: tmp1(klon)
    integer :: jl

    do jl=1,klon
        tmp1(jl) = 0
        field1(jl) = tmp1(jl)
    end do

end subroutine kernel
    """.strip()
    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel.F90').write_text(fcode_kernel)


@pytest.mark.usefixtures('fcode_as_module')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('suffix,module_suffix', (
    ('_duplicated', None), ('_dupl1', '_dupl2'), ('_d_test_1', '_d_test_2')
))
@pytest.mark.parametrize('full_parse', (True, False))
def test_dependency_duplicate_plan(tmp_path, frontend, suffix, module_suffix, config, full_parse):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path], full_parse=full_parse
    )

    pipeline = Pipeline(classes=(DuplicateKernel, FileWriteTransformation),
                        duplicate_kernels=('kernel',), duplicate_suffix=suffix,
                        duplicate_module_suffix=module_suffix)

    plan_file = tmp_path/'plan.cmake'
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=tmp_path)

    module_suffix = module_suffix or suffix

    # Validate the Scheduler graph:
    # - New procedure item has been added
    # - Module item has been created but is not in the sgraph
    assert f'kernel_mod{module_suffix}' in scheduler.item_factory.item_cache
    item = scheduler.item_factory.item_cache[f'kernel_mod{module_suffix}']
    assert isinstance(item, ModuleItem)
    assert item.ir.name == item.local_name
    assert f'kernel_mod{module_suffix}' not in scheduler

    assert f'kernel_mod{module_suffix}#kernel{suffix}' in scheduler.item_factory.item_cache
    assert f'kernel_mod{module_suffix}#kernel{suffix}' in scheduler
    item = scheduler[f'kernel_mod{module_suffix}#kernel{suffix}']
    assert isinstance(item, ProcedureItem)
    assert item.ir.name == item.local_name

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == {'kernel_mod', 'driver'}
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == {'kernel_mod', 'driver'}
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {f'kernel_mod{module_suffix}.idem', 'kernel_mod.idem', 'driver.idem'}


@pytest.mark.usefixtures('fcode_as_module')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('suffix,module_suffix', (
    ('_duplicated', None), ('_dupl1', '_dupl2'), ('_d_test_1', '_d_test_2')
))
def test_dependency_duplicate_trafo(tmp_path, frontend, suffix, module_suffix, config):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path]
    )

    pipeline = Pipeline(classes=(DuplicateKernel, FileWriteTransformation),
                        duplicate_kernels=('kernel',), duplicate_suffix=suffix,
                        duplicate_module_suffix=module_suffix)

    scheduler.process(pipeline)

    module_suffix = module_suffix or suffix

    # Validate the Scheduler graph:
    # - New procedure item has been added
    # - Module item has been created but is not in the sgraph
    assert f'kernel_mod{module_suffix}' in scheduler.item_factory.item_cache
    item = scheduler.item_factory.item_cache[f'kernel_mod{module_suffix}']
    assert isinstance(item, ModuleItem)
    assert item.ir.name == item.local_name
    assert f'kernel_mod{module_suffix}' not in scheduler

    assert f'kernel_mod{module_suffix}#kernel{suffix}' in scheduler.item_factory.item_cache
    assert f'kernel_mod{module_suffix}#kernel{suffix}' in scheduler
    item = scheduler[f'kernel_mod{module_suffix}#kernel{suffix}']
    assert isinstance(item, ProcedureItem)
    assert item.ir.name == item.local_name

    driver = scheduler["#driver"].ir
    kernel = scheduler["kernel_mod#kernel"].ir
    new_kernel = scheduler[f"kernel_mod{module_suffix}#kernel{suffix}"].ir

    calls_driver = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls_driver) == 2
    assert new_kernel is not kernel
    assert calls_driver[0].routine == kernel
    assert calls_driver[1].routine == new_kernel


@pytest.mark.usefixtures('fcode_as_module')
@pytest.mark.parametrize('frontend', available_frontends())
def test_dependency_remove(tmp_path, frontend, config):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path]
    )
    pipeline = Pipeline(classes=(RemoveKernel, FileWriteTransformation),
                        remove_kernels=('kernel',))

    plan_file = tmp_path/'plan.cmake'
    root_path = tmp_path
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=root_path)

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == {'driver'}
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == {'driver'}
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {'driver.idem'}

    scheduler.process(pipeline)
    driver = scheduler["#driver"].ir
    assert "kernel_mod#kernel" not in scheduler

    calls_driver = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls_driver) == 0


@pytest.mark.usefixtures('fcode_no_module')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('suffix, module_suffix', (
    ('_duplicated', None), ('_dupl1', '_dupl2'), ('_d_test_1', '_d_test_2')
))
@pytest.mark.parametrize('full_parse', (True, False))
def test_dependency_duplicate_plan_no_module(tmp_path, frontend, suffix, module_suffix, config, full_parse):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path], full_parse=full_parse
    )

    pipeline = Pipeline(classes=(DuplicateKernel, FileWriteTransformation),
                        duplicate_kernels=('kernel',), duplicate_suffix=suffix,
                        duplicate_module_suffix=module_suffix)

    plan_file = tmp_path/'plan.cmake'
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=tmp_path)

    # Validate Scheduler graph
    assert f'#kernel{suffix}' in scheduler.item_factory.item_cache
    assert f'#kernel{suffix}' in scheduler
    assert isinstance(scheduler[f'#kernel{suffix}'], ProcedureItem)
    assert scheduler[f'#kernel{suffix}'].ir.name == f'kernel{suffix}'

    # Validate IR objects
    kernel = scheduler["#kernel"].ir
    new_kernel = scheduler[f"#kernel{suffix}"].ir
    assert new_kernel is not kernel

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == {'kernel', 'driver'}
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == {'kernel', 'driver'}
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {f'kernel{suffix}.idem', 'kernel.idem', 'driver.idem'}


@pytest.mark.usefixtures('fcode_no_module')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('suffix, module_suffix', (
    ('_duplicated', None), ('_dupl1', '_dupl2'), ('_d_test_1', '_d_test_2')
))
def test_dependency_duplicate_trafo_no_module(tmp_path, frontend, suffix, module_suffix, config):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path]
    )

    pipeline = Pipeline(classes=(DuplicateKernel, FileWriteTransformation),
                        duplicate_kernels=('kernel',), duplicate_suffix=suffix,
                        duplicate_module_suffix=module_suffix)

    scheduler.process(pipeline)

    # Validate Scheduler graph
    assert f'#kernel{suffix}' in scheduler.item_factory.item_cache
    assert f'#kernel{suffix}' in scheduler
    assert isinstance(scheduler[f'#kernel{suffix}'], ProcedureItem)
    assert scheduler[f'#kernel{suffix}'].ir.name == f'kernel{suffix}'

    # Validate transformed objects
    driver = scheduler["#driver"].ir
    kernel = scheduler["#kernel"].ir
    new_kernel = scheduler[f"#kernel{suffix}"].ir

    calls_driver = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls_driver) == 2
    assert new_kernel is not kernel
    assert calls_driver[0].routine == kernel
    assert calls_driver[1].routine == new_kernel


@pytest.mark.usefixtures('fcode_no_module')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('full_parse', (True, False))
def test_dependency_remove_plan_no_module(tmp_path, frontend, config, full_parse):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path], full_parse=full_parse
    )
    pipeline = Pipeline(classes=(RemoveKernel, FileWriteTransformation),
                        remove_kernels=('kernel',))

    plan_file = tmp_path/'plan.cmake'
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=tmp_path)

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == {'driver'}
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == {'driver'}
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {'driver.idem'}

    assert '#kernel' not in scheduler


@pytest.mark.usefixtures('fcode_no_module')
@pytest.mark.parametrize('frontend', available_frontends())
def test_dependency_remove_trafo_no_module(tmp_path, frontend, config):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path]
    )
    pipeline = Pipeline(classes=(RemoveKernel, FileWriteTransformation),
                        remove_kernels=('kernel',))

    scheduler.process(pipeline)
    driver = scheduler["#driver"].ir
    assert "#kernel" not in scheduler

    assert not FindNodes(ir.CallStatement).visit(driver.body)


@pytest.mark.usefixtures('fcode_as_module')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('duplicate_kernels,remove_kernels', (
    ('kernel', 'kernel'), ('kernel', 'kernel_new'), ('kernel', None), (None, 'kernel')
))
@pytest.mark.parametrize('full_parse', (True, False))
def test_dependency_duplicate_remove_plan(tmp_path, frontend, duplicate_kernels, remove_kernels,
                                          config, full_parse):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path], full_parse=full_parse
    )

    expected_items = {'kernel_mod#kernel', '#driver'}
    assert {item.name for item in scheduler.items} == expected_items

    pipeline = Pipeline(classes=(DuplicateKernel, RemoveKernel, FileWriteTransformation),
                        duplicate_kernels=duplicate_kernels, duplicate_suffix='_new',
                        remove_kernels=remove_kernels)

    plan_file = tmp_path/'plan.cmake'
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=tmp_path)

    for kernel in as_tuple(duplicate_kernels):
        for name in list(expected_items):
            scope_name, local_name = name.split('#')
            if local_name == kernel:
                expected_items.add(f'{scope_name}_new#{local_name}_new')

    for kernel in as_tuple(remove_kernels):
        for name in list(expected_items):
            scope_name, local_name = name.split('#')
            if local_name == kernel:
                expected_items.remove(name)

    # Validate Scheduler graph
    assert {item.name for item in scheduler.items} == expected_items

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    # Duplicated items are generated from, and hence mapped to, the original source file
    transformed_items = {(name.split('#')[0] or name[1:]).removesuffix('_new') for name in expected_items}
    removed_items = {name.split('#')[0] or name[1:] for name in expected_items if not name.endswith('_new')}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == transformed_items
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == removed_items
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {f'{name.split("#")[0] or name[1:]}.idem' for name in expected_items}


@pytest.mark.usefixtures('fcode_no_module')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('duplicate_kernels,remove_kernels', (
    ('kernel', 'kernel'), ('kernel', 'kernel_new'), ('kernel', None), (None, 'kernel')
))
@pytest.mark.parametrize('full_parse', (True, False))
def test_dependency_duplicate_remove_plan_no_module(tmp_path, frontend, duplicate_kernels, remove_kernels,
                                                    config, full_parse):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path], full_parse=full_parse
    )

    expected_items = {'#kernel', '#driver'}
    assert {item.name for item in scheduler.items} == expected_items

    pipeline = Pipeline(classes=(DuplicateKernel, RemoveKernel, FileWriteTransformation),
                        duplicate_kernels=duplicate_kernels, duplicate_suffix='_new',
                        remove_kernels=remove_kernels)

    plan_file = tmp_path/'plan.cmake'
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=tmp_path)

    if duplicate_kernels:
        expected_items.add(f'#{duplicate_kernels}_new')

    if remove_kernels:
        expected_items.remove(f'#{remove_kernels}')

    # Validate Scheduler graph
    assert {item.name for item in scheduler.items} == expected_items

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    # Duplicated items are generated from, and hence mapped to, the original source file
    transformed_items = {name[1:].removesuffix('_new') for name in expected_items}
    removed_items = {name[1:] for name in expected_items if not name.endswith('_new')}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == transformed_items
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == removed_items
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {f'{name[1:]}.idem' for name in expected_items}

@pytest.fixture(name='fcode_as_module_extended')
def fixture_fcode_as_module_extended(tmp_path):
    fcode_driver = """
subroutine driver(NLON, NB, FIELD1)
    use kernel_mod, only: kernel
    implicit none
    INTEGER, INTENT(IN) :: NLON, NB
    integer :: b
    integer, intent(inout) :: field1(nlon, nb)
    integer :: local_nlon
    local_nlon = nlon
    do b=1,nb
        call kernel(local_nlon, field1(:,b))
    end do
end subroutine driver
    """.strip()
    fcode_kernel = """
module kernel_mod
    implicit none
contains
    subroutine kernel(klon, field1)
        use iso_fortran_env, only: real64
        use kernel_nested_mod, only: kernel_nested_vector, kernel_nested_seq
        use compute_2_mod, only: compute_2, compute_2_1
        use compute_3_mod, only: compute_3
        implicit none
        integer, intent(in) :: klon
        integer, intent(inout) :: field1(klon)
        integer :: tmp1(klon)
        integer :: jl

        call kernel_nested_vector(klon, field1)

        do jl=1,klon
            call kernel_nested_seq(field1(jl))
            call compute_2(field1(jl))
            call compute_2_1(field1(jl))
            tmp1(jl) = 0
            field1(jl) = tmp1(jl)
            call compute_3(field1(jl))
        end do

    end subroutine kernel
end module kernel_mod
    """.strip()
    fcode_kernel_nested = """
module kernel_nested_mod
    implicit none
contains
    subroutine kernel_nested_vector(klon, field1)
        implicit none
        integer, intent(in) :: klon
        integer, intent(inout) :: field1(klon)
        integer :: tmp1(klon)
        integer :: jl

        do jl=1,klon
            tmp1(jl) = 0
            field1(jl) = tmp1(jl)
        end do

    end subroutine kernel_nested_vector
    subroutine kernel_nested_seq(val)
        use compute_1_mod, only: compute_1
        implicit none
        integer, intent(inout) :: val

        val = 0
        call compute_1(val)

    end subroutine kernel_nested_seq
end module kernel_nested_mod
    """.strip()
    fcode_compute_1 = """
module compute_1_mod
    implicit none
contains
    subroutine compute_1(val)
        implicit none
        integer, intent(inout) :: val

        val = 0

    end subroutine compute_1
end module compute_1_mod
    """.strip()
    fcode_compute_2 = """
module compute_2_mod
    implicit none
contains
    subroutine compute_2(val)
        implicit none
        integer, intent(inout) :: val

        val = 0

    end subroutine compute_2
    subroutine compute_2_1(val)
        implicit none
        integer, intent(inout) :: val

        val = 0

    end subroutine compute_2_1
end module compute_2_mod
    """.strip()
    fcode_compute_3 = """
module compute_3_mod
    implicit none
contains
    subroutine compute_3(val)
        implicit none
        integer, intent(inout) :: val

        val = 0

    end subroutine compute_3
end module compute_3_mod
    """.strip()

    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel_mod.F90').write_text(fcode_kernel)
    (tmp_path/'kernel_nested_mod.F90').write_text(fcode_kernel_nested)
    (tmp_path/'compute_1_mod.F90').write_text(fcode_compute_1)
    (tmp_path/'compute_2_mod.F90').write_text(fcode_compute_2)
    (tmp_path/'compute_3_mod.F90').write_text(fcode_compute_3)


@pytest.mark.usefixtures('fcode_as_module_extended')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('suffix,module_suffix', (
    ('_duplicated', None), ('_dupl1', '_dupl2'), ('_d_test_1', '_d_test_2')
))
@pytest.mark.parametrize('full_parse', (True, False))
@pytest.mark.parametrize('duplicate_subgraph', (True, False))
def test_dependency_duplicate_subgraph(tmp_path, frontend, suffix, module_suffix, config,
                                       full_parse, duplicate_subgraph):

    config['routines']['kernel'] = {'role': 'kernel', 'ignore': ['compute_2_1', 'compute_3']}
    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path], full_parse=full_parse
    )

    pipeline = Pipeline(classes=(DuplicateKernel, FileWriteTransformation),
                        duplicate_kernels=('kernel',), duplicate_suffix=suffix,
                        duplicate_module_suffix=module_suffix,
                        duplicate_subgraph=duplicate_subgraph)

    module_suffix = module_suffix or suffix

    plan_file = tmp_path/'plan.cmake'

    # dry-run for planning
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=tmp_path)

    expected_items = {'#driver', 'kernel_mod#kernel', 'kernel_nested_mod#kernel_nested_vector',
            'kernel_nested_mod#kernel_nested_seq', 'compute_1_mod#compute_1', 'compute_2_mod#compute_2',
            'compute_2_mod#compute_2_1', 'compute_3_mod#compute_3'
    }
    standard_items = {'iso_fortran_env'}
    expected_items |= {f'kernel_mod{module_suffix}#kernel{suffix}'}
    if duplicate_subgraph:
        expected_items |= {f'kernel_nested_mod{module_suffix}#kernel_nested_vector{suffix}',
                f'kernel_nested_mod{module_suffix}#kernel_nested_seq{suffix}',
                f'compute_1_mod{module_suffix}#compute_1{suffix}',
                f'compute_2_mod{module_suffix}#compute_2{suffix}'
        }
    # Validate Scheduler graph
    assert {item.name for item in scheduler.items} == expected_items | standard_items

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    transformed_items = {name.split('#')[0] if name.split('#')[0] else name[1:]
                         for name in expected_items if not name.endswith(f'{suffix}')}
    transformed_items -= {'compute_3_mod'}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == transformed_items
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == transformed_items
    appended_items = {name.split('#')[0] if name.split('#')[0] else name[1:] for name in expected_items}
    appended_items -= {'compute_3_mod'}
    appended_items = {f'{name}.idem' for name in appended_items}
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == appended_items

    # actual transformation(s) if fully parsed
    if full_parse:
        scheduler.process(pipeline)

        if duplicate_subgraph:
            dupl_kernel_imports = [('iso_fortran_env', ['real64']), (f'kernel_nested_mod{module_suffix}',
                                    [f'kernel_nested_vector{suffix}', f'kernel_nested_seq{suffix}']),
                                   (f'compute_2_mod{module_suffix}', [f'compute_2{suffix}']),
                                   ('compute_2_mod', ['compute_2_1']),
                                   ('compute_3_mod', ['compute_3'])]
            dupl_kernel_calls = [f'kernel_nested_vector{suffix}', f'kernel_nested_seq{suffix}', f'compute_2{suffix}',
                    'compute_2_1', 'compute_3']
        else:
            dupl_kernel_imports = [('iso_fortran_env', ['real64']),
                                   ('kernel_nested_mod', ['kernel_nested_vector', 'kernel_nested_seq']),
                                   ('compute_2_mod', ['compute_2', 'compute_2_1']),
                                   ('compute_3_mod', ['compute_3'])]
            dupl_kernel_calls = ['kernel_nested_vector', 'kernel_nested_seq', 'compute_2', 'compute_2_1', 'compute_3']

        expected_imports = {
                '#driver': [(f'kernel_mod{module_suffix}', [f'kernel{suffix}']), ('kernel_mod', ['kernel'])],
                'kernel_mod#kernel': [('iso_fortran_env', ['real64']),
                                      ('kernel_nested_mod', ['kernel_nested_vector', 'kernel_nested_seq']),
                                      ('compute_2_mod', ['compute_2', 'compute_2_1']),
                                      ('compute_3_mod', ['compute_3'])],
                'kernel_nested_mod#kernel_nested_vector': [],
                'kernel_nested_mod#kernel_nested_seq': [('compute_1_mod', ['compute_1'])],
                'compute_1_mod#compute_1': [],
                'compute_2_mod#compute_2': [],
                f'kernel_mod{module_suffix}#kernel{suffix}': dupl_kernel_imports 
        }
        expected_calls = {
                '#driver': ['kernel', f'kernel{suffix}'],
                'kernel_mod#kernel': ['kernel_nested_vector', 'kernel_nested_seq', 'compute_2',
                    'compute_2_1', 'compute_3'],
                'kernel_nested_mod#kernel_nested_vector': [],
                'kernel_nested_mod#kernel_nested_seq': ['compute_1'],
                'compute_1_mod#compute_1': [],
                'compute_2_mod#compute_2': [],
                f'kernel_mod{module_suffix}#kernel{suffix}': dupl_kernel_calls
        }

        if duplicate_subgraph:
            expected_imports |= {
                    f'kernel_nested_mod{module_suffix}#kernel_nested_vector{suffix}': [],
                    f'kernel_nested_mod{module_suffix}#kernel_nested_seq{suffix}': [(f'compute_1_mod{module_suffix}',
                                                                                     [f'compute_1{suffix}'])],
                    f'compute_1_mod{module_suffix}#compute_1{suffix}': [],
                    f'compute_2_mod{module_suffix}#compute_2{suffix}': []
            }
            expected_calls |= {
                    f'kernel_nested_mod{module_suffix}#kernel_nested_vector{suffix}': [],
                    f'kernel_nested_mod{module_suffix}#kernel_nested_seq{suffix}':  [f'compute_1{suffix}'],
                    f'compute_1_mod{module_suffix}#compute_1{suffix}': [],
                    f'compute_2_mod{module_suffix}#compute_2{suffix}': []
            }

        for item_name, calls in expected_calls.items():
            routine = scheduler[item_name].ir
            calls = [str(call.name).lower() for call in FindNodes(ir.CallStatement).visit(routine.body)]
            imports = [(imp.module.lower(), [symb.name.lower() for symb in imp.symbols]) for imp in routine.imports]
            assert calls == expected_calls[item_name]
            assert imports == expected_imports[item_name]
