# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
A selection of tests for (proof-of-concept) transformations changing
dependencies through e.g., duplicating or removing kernels (and calls).
"""
import re
from pathlib import Path
import pytest

from loki.batch import Pipeline, ProcedureItem, ModuleItem
from loki import (
    Scheduler, SchedulerConfig, ProcessingStrategy
)
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes
from loki.tools import as_tuple
from loki.transformations.dependency import (
        DuplicateKernel, RemoveKernel
)
from loki.backend import fgen
from loki.transformations.build_system import FileWriteTransformation


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

    transformed_items = {name.split('#')[0] or name[1:] for name in expected_items if not name.endswith('_new')}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == transformed_items
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == transformed_items
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

    transformed_items = {name[1:] for name in expected_items if not name.endswith('_new')}
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == transformed_items
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == transformed_items
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
        use kernel_nested_mod, only: kernel_nested_vector, kernel_nested_seq
        use compute_2_mod, only: compute_2
        implicit none
        integer, intent(in) :: klon
        integer, intent(inout) :: field1(klon)
        integer :: tmp1(klon)
        integer :: jl

        call kernel_nested_vector(klon, field1)

        do jl=1,klon
            call kernel_nested_seq(field1(jl))
            call compute_2(field1(jl))
            tmp1(jl) = 0
            field1(jl) = tmp1(jl)
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
end module compute_2_mod
    """.strip()
    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel_mod.F90').write_text(fcode_kernel)
    (tmp_path/'nested_kernel_mod.F90').write_text(fcode_kernel_nested)
    (tmp_path/'compute_1_mod.F90').write_text(fcode_compute_1)
    (tmp_path/'compute_2_mod.F90').write_text(fcode_compute_2)

@pytest.mark.usefixtures('fcode_as_module_extended')
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('suffix,module_suffix', (
    ('_duplicated', None), # ('_dupl1', '_dupl2'), ('_d_test_1', '_d_test_2')
))
@pytest.mark.parametrize('full_parse', (True,)) #  False))
@pytest.mark.parametrize('duplicate_subgraph', (True, False))
def test_dependency_duplicate_plan_test(tmp_path, frontend, suffix, module_suffix, config, full_parse, duplicate_subgraph):

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config),
        frontend=frontend, xmods=[tmp_path], full_parse=full_parse
    )

    pipeline = Pipeline(classes=(DuplicateKernel, FileWriteTransformation),
                        duplicate_kernels=('kernel',), duplicate_suffix=suffix,
                        duplicate_module_suffix=module_suffix,
                        duplicate_subgraph=duplicate_subgraph)

    plan_file = tmp_path/'plan.cmake'
    
    # dry-run for planning
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=tmp_path)
    
    # generate callgraph
    scheduler.callgraph(tmp_path/'callgraph', with_file_graph=True)
    print(f"wrote callgraph to: {tmp_path/'callgraph.pdf'}")

    # actual transformation(s)
    scheduler.process(pipeline)
