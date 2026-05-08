# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from pathlib import Path

import pytest

from loki import Scheduler
from loki.batch import Pipeline, ProcessingStrategy, SchedulerConfig
from loki.frontend import OMNI, available_frontends
from loki.ir import CallStatement, FindNodes, Import
from loki.transformations import (
   IdemTransformation, ReplaceKernels, ReplaceKernels2
)
from loki.transformations.build_system import (
    DependencyTransformation, FileWriteTransformation, ModuleWrapTransformation
)
from loki.logging import warning


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def _read_plan_dict(plan_file):
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    return {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}


def _replace_scheduler(frontend, here, tmp_path, replace_kernels_map, config=None, paths=None):
    if config is None:
        config = {
            'default': {
                'mode': 'idem',
                'role': 'kernel',
                'expand': True,
                'strict': True,
            },
            'routines': {
                'driver': {'role': 'driver'},
            }
        }
    return Scheduler(paths=paths or here/'sources/projReplaceKernel', config=config, frontend=frontend, xmods=[tmp_path]), replace_kernels_map


def _get_item_map(scheduler):
    return {item.name: item for item in scheduler.items}


def _get_call_and_import(item_map, routine_item_name):
    routine_item = item_map[routine_item_name]
    routine = routine_item.source[routine_item.local_name]
    calls = FindNodes(CallStatement).visit(routine.body)
    imports = FindNodes(Import).visit(routine.spec)
    return routine, calls, imports


def _derived_member_scheduler(frontend, here, tmp_path):
    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
        },
        'routines': {
            'kernel_geom_call': {'role': 'driver'},
            'kernel_geom_call_kw': {'role': 'driver'},
        }
    }
    replace_kernels_map = {
        'kernel_geom': {
            'routine': 'kernel_geom_repl',
            'args': {
                'ydgeometry': {'map_to': 'kgpblks', 'member': 'YRDIM%NGPBLKS'}
            }
        }
    }
    scheduler = Scheduler(paths=here/'sources/projReplaceKernel', config=config, frontend=frontend, xmods=[tmp_path])
    return scheduler, replace_kernels_map


def _derived_expr_scheduler(frontend, here, tmp_path):
    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
        },
        'routines': {
            'kernel_geom_expr_call': {'role': 'driver'},
            'kernel_geom_expr_call_kw': {'role': 'driver'},
        }
    }
    replace_kernels_map = {
        'kernel_geom_expr': {
            'routine': 'kernel_geom_expr_repl',
            'args': {
                'kst': {'position': 1},
                'kend': {
                    'map_to': 'kend_in',
                    'placeholders': {
                        'geom_total': 'ydgeometry',
                        'geom_dim': 'ydgeometry',
                    },
                    'expr': 'MOD({geom_total}%YRGEM%NGPTOT, {geom_dim}%YRDIM%NPROMA)'
                },
                'ydgeometry': {'map_to': 'kgpblks', 'member': 'YRDIM%NGPBLKS'},
                'ldacc': '.true.',
            }
        }
    }
    scheduler = Scheduler(paths=here/'sources/projReplaceKernel', config=config, frontend=frontend, xmods=[tmp_path])
    return scheduler, replace_kernels_map


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
@pytest.mark.parametrize('make_replace_kernel_ignore', [False, True])
def test_replace_kernel_pipeline_ignore(frontend, here, tmp_path, make_replace_kernel_ignore):
    source_dir = here/'sources/projReplaceKernel'
    output_dir = tmp_path/'build'
    output_dir.mkdir()

    config = SchedulerConfig.from_dict({
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
        },
        'routines': {
            'driver': {'role': 'driver'},
        }
    })

    replace_kernels_map = {
        'kernel_a1': {
            'routine': 'kernel_a1_repl',
            'ignore': make_replace_kernel_ignore,
        }
    }

    pipeline = Pipeline(classes=(
        IdemTransformation,
        ReplaceKernels,
        ModuleWrapTransformation,
        DependencyTransformation,
        FileWriteTransformation,
    ), replace_kernels_map=replace_kernels_map, module_suffix='_mod', suffix='_test')

    plan_scheduler = Scheduler(
        paths=source_dir, config=config, frontend=frontend, xmods=[tmp_path], output_dir=output_dir
    )
    plan_file = tmp_path/'plan.cmake'
    plan_scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    plan_scheduler.write_cmake_plan(filepath=plan_file, rootpath=source_dir)
    plan_dict = _read_plan_dict(plan_file)

    replacement_file_stem = 'kernel_a1_repl_mod'
    replacement_item_name = 'kernel_a1_repl_mod#kernel_a1_repl'
    transformed_replacement_item_name = 'kernel_a1_repl_test_mod#kernel_a1_repl_test'

    if make_replace_kernel_ignore:
        # With ignore=True, the replacement is added to the caller's block list,
        # so the item never enters the scheduler graph
        assert replacement_item_name not in {item.name for item in plan_scheduler.items}
        assert replacement_file_stem not in plan_dict['LOKI_SOURCES_TO_TRANSFORM']
        assert replacement_file_stem not in plan_dict['LOKI_SOURCES_TO_APPEND']
        assert transformed_replacement_item_name not in {item.name for item in plan_scheduler.items}
    else:
        assert replacement_item_name in {item.name for item in plan_scheduler.items}
        assert not plan_scheduler[replacement_item_name].is_ignored
        assert replacement_file_stem in plan_dict['LOKI_SOURCES_TO_TRANSFORM']

    scheduler = Scheduler(
        paths=source_dir, config=config, frontend=frontend, xmods=[tmp_path], output_dir=output_dir
    )
    scheduler.process(pipeline)

    replacement_items = {item.name for item in scheduler.items if 'kernel_a1_repl' in item.name}
    if make_replace_kernel_ignore:
        # With block, the replacement item is not in the graph at all
        assert not replacement_items
    else:
        assert transformed_replacement_item_name in replacement_items

    kernel_a_candidates = [
        item.name for item in scheduler.items
        if item.name.endswith('#kernel_a') or item.name.endswith('#kernel_a_test')
    ]
    assert kernel_a_candidates
    kernel_a_item = scheduler[kernel_a_candidates[0]]
    kernel_a = kernel_a_item.source[kernel_a_item.local_name]
    calls = FindNodes(CallStatement).visit(kernel_a.body)
    assert any('kernel_a1_repl' in str(call.name).lower() for call in calls)
    imports = FindNodes(Import).visit(kernel_a.spec)
    assert any('kernel_a1_repl' in str(imp.module).lower() for imp in imports)

    written_files = {path.name for path in output_dir.glob('*')}
    replacement_output = 'kernel_a1_repl_mod.idem_test'
    if make_replace_kernel_ignore:
        assert replacement_output not in written_files
    else:
        assert replacement_output in written_files

@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_argument_rename(frontend, here, tmp_path):
    scheduler, replace_kernels_map = _replace_scheduler(
        frontend, here, tmp_path,
        {'kernel_a1': {'routine': 'kernel_a4_repl', 'args': {'flag2': 'flag2_renamed'}}}
    )
    scheduler.process(transformation=ReplaceKernels(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    _, calls, imports = _get_call_and_import(item_map, 'kernel_a_mod#kernel_a')

    assert 'kernel_a4_repl_mod#kernel_a4_repl' in item_map
    assert any(str(call.name).lower() == 'kernel_a4_repl' for call in calls)
    assert any(('flag2_renamed', 'flag2') == (kw[0].lower(), str(kw[1]).lower()) for kw in calls[0].kwarguments)
    assert any(str(imp.module).lower() == 'kernel_a4_repl_mod' for imp in imports)


@pytest.mark.parametrize('trafo_cls', [ReplaceKernels, ReplaceKernels2])
@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_argument_name_remap(frontend, here, tmp_path, trafo_cls):
    scheduler, replace_kernels_map = _replace_scheduler(
        frontend, here, tmp_path,
        {'kernel_a1': {'routine': 'kernel_a5_repl', 'args': {'b': 'b1'}}}
    )
    scheduler.process(transformation=trafo_cls(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    _, calls, imports = _get_call_and_import(item_map, 'kernel_a_mod#kernel_a')

    assert 'kernel_a5_repl_mod#kernel_a5_repl' in item_map
    assert any(str(call.name).lower() == 'kernel_a5_repl' for call in calls)
    assert calls[0].arguments == ()
    assert any(('b1', 'b') == (kw[0].lower(), str(kw[1]).lower()) for kw in calls[0].kwarguments)
    assert any(str(imp.module).lower() == 'kernel_a5_repl_mod' for imp in imports)

@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_argument_override_literal(frontend, here, tmp_path):
    scheduler, replace_kernels_map = _replace_scheduler(
        frontend, here, tmp_path,
        {'kernel_a1': {'routine': 'kernel_a1_repl', 'args': {'flag1': '.true.'}}}
    )
    scheduler.process(transformation=ReplaceKernels(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    _, calls, imports = _get_call_and_import(item_map, 'kernel_a_mod#kernel_a')

    assert 'kernel_a1_repl_mod#kernel_a1_repl' in item_map
    assert any(str(call.name).lower() == 'kernel_a1_repl' for call in calls)
    assert calls[0].arguments == ('b', 'c')
    assert any(('flag1', 'true') == (kw[0].lower(), str(kw[1]).lower()) for kw in calls[0].kwarguments)
    assert any(str(imp.module).lower() == 'kernel_a1_repl_mod' for imp in imports)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_loads_missing_target(frontend, here, tmp_path):
    reduced_source_dir = tmp_path/'projReplaceKernelReduced'
    reduced_source_dir.mkdir()
    full_source_dir = here/'sources/projReplaceKernel'
    for filename in ('driver_mod.F90', 'kernel_a_mod.F90', 'kernel_a1_mod.F90', 'kernel_b_mod.F90'):
        (reduced_source_dir/filename).write_text((full_source_dir/filename).read_text())

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
        },
        'routines': {
            'driver': {'role': 'driver'},
        }
    }
    scheduler, replace_kernels_map = _replace_scheduler(
        frontend, here, tmp_path,
        {'kernel_a1': {'routine': 'kernel_a1_repl', 'args': {'flag1': '.true.'}}},
        config=config, paths=reduced_source_dir
    )
    trafo = ReplaceKernels(replace_kernels_map)
    kernel_a_item = scheduler['kernel_a_mod#kernel_a']
    build_args = dict(scheduler.build_args)
    build_args['paths'] = (full_source_dir,)
    trafo.apply(
        kernel_a_item.ir,
        item=kernel_a_item,
        item_factory=scheduler.item_factory,
        scheduler_config=scheduler.config,
        build_args=build_args,
        plan_mode=True,
    )

    item_map = _get_item_map(scheduler)
    _, calls, imports = _get_call_and_import(item_map, 'kernel_a_mod#kernel_a')
    scheduler.item_factory.item_cache['kernel_a1_repl_mod'].create_definition_items(
        item_factory=scheduler.item_factory, config=scheduler.config
    )

    assert 'kernel_a1_repl_mod#kernel_a1_repl' in scheduler.item_factory.item_cache
    assert any(str(call.name).lower() == 'kernel_a1_repl' for call in calls)
    assert calls[0].arguments == ('b', 'c')
    assert any(('flag1', 'true') == (kw[0].lower(), str(kw[1]).lower()) for kw in calls[0].kwarguments)
    assert any(str(imp.module).lower() == 'kernel_a1_repl_mod' for imp in imports)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_omits_unmapped_optional_argument(frontend, here, tmp_path, caplog):
    scheduler, replace_kernels_map = _replace_scheduler(
        frontend, here, tmp_path,
        {'kernel_a1': {'routine': 'kernel_a6_repl'}}
    )
    with caplog.at_level('WARNING'):
        scheduler.process(transformation=ReplaceKernels(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    _, calls, imports = _get_call_and_import(item_map, 'kernel_a_mod#kernel_a')

    assert 'kernel_a6_repl_mod#kernel_a6_repl' in item_map
    assert any(str(call.name).lower() == 'kernel_a6_repl' for call in calls)
    assert all(kw[0].lower() != 'optional_flag' for kw in calls[0].kwarguments)
    assert any('optional replacement argument' in message for message in caplog.messages)
    assert any(str(imp.module).lower() == 'kernel_a6_repl_mod' for imp in imports)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_intfb_include(frontend, here, tmp_path):
    """
    Regression test: when the original kernel is imported via #include "kernel.intfb.h"
    (c_import=True, no symbols), ReplaceKernels must rewrite/remove the include and
    inject a proper USE statement for the replacement module, so that scheduler
    rediscovery can resolve the replacement routine.
    """
    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': False,
        },
        'routines': {
            'kernel_a_intfb': {'role': 'driver'},
        }
    }
    replace_kernels_map = {'kernel_a1': {'routine': 'kernel_a1_repl'}}
    source_dir = here/'sources/projReplaceKernel'
    scheduler = Scheduler(
        paths=source_dir, config=config, frontend=frontend,
        xmods=[tmp_path], includes=[source_dir]
    )
    scheduler.process(transformation=ReplaceKernels(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    routine, calls, imports = _get_call_and_import(item_map, 'kernel_a_intfb_mod#kernel_a_intfb')

    # The call should now target kernel_a1_repl
    assert any(str(call.name).lower() == 'kernel_a1_repl' for call in calls)

    # The old #include "kernel_a1.intfb.h" should be removed or replaced
    c_imports = [imp for imp in FindNodes(Import).visit(routine.spec) if imp.c_import]
    assert not any('kernel_a1' in str(imp.module) for imp in c_imports), \
        'Old #include "kernel_a1.intfb.h" should be removed after replacement'

    # A proper USE statement for the replacement module should exist
    all_imports = FindNodes(Import).visit(routine.spec) + FindNodes(Import).visit(routine.parent.spec)
    assert any(str(imp.module).lower() == 'kernel_a1_repl_mod' for imp in all_imports), \
        'A USE statement for the replacement module should be injected'

    # The replacement item should be in the scheduler
    assert 'kernel_a1_repl_mod#kernel_a1_repl' in item_map


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_derived_member_mapping_positional(frontend, here, tmp_path):
    scheduler, replace_kernels_map = _derived_member_scheduler(frontend, here, tmp_path)
    scheduler.process(transformation=ReplaceKernels(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    routine, calls, _ = _get_call_and_import(item_map, 'kernel_geom_call_mod#kernel_geom_call')
    imports = FindNodes(Import).visit(routine.parent.spec)

    assert 'kernel_geom_repl_mod#kernel_geom_repl' in item_map
    assert any(str(call.name).lower() == 'kernel_geom_repl' for call in calls)
    assert any(
        ('kgpblks', 'ydgeometry%yrdim%ngpblks') == (kw[0].lower(), str(kw[1]).lower())
        for kw in calls[0].kwarguments
    )
    assert any(str(imp.module).lower() == 'kernel_geom_repl_mod' for imp in imports)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_derived_member_mapping_keyword_actual(frontend, here, tmp_path):
    scheduler, replace_kernels_map = _derived_member_scheduler(frontend, here, tmp_path)
    scheduler.process(transformation=ReplaceKernels(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    routine, calls, _ = _get_call_and_import(item_map, 'kernel_geom_call_kw_mod#kernel_geom_call_kw')
    imports = FindNodes(Import).visit(routine.parent.spec)

    assert 'kernel_geom_repl_mod#kernel_geom_repl' in item_map
    assert any(str(call.name).lower() == 'kernel_geom_repl' for call in calls)
    assert any(
        ('kgpblks', 'ydgeo%yrdim%ngpblks') == (kw[0].lower(), str(kw[1]).lower())
        for kw in calls[0].kwarguments
    )
    assert any(str(imp.module).lower() == 'kernel_geom_repl_mod' for imp in imports)


@pytest.mark.parametrize(
    'routine_item_name, expected_geom, expected_start',
    [
        ('kernel_geom_expr_call_mod#kernel_geom_expr_call', 'ydgeometry', 'kst'),
        ('kernel_geom_expr_call_kw_mod#kernel_geom_expr_call_kw', 'ydgeo', 'start_idx'),
    ]
)
@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_expression_template_multi_placeholder(
    frontend, here, tmp_path, routine_item_name, expected_geom, expected_start
):
    scheduler, replace_kernels_map = _derived_expr_scheduler(frontend, here, tmp_path)
    scheduler.process(transformation=ReplaceKernels(replace_kernels_map))

    item_map = _get_item_map(scheduler)
    routine, calls, _ = _get_call_and_import(item_map, routine_item_name)
    imports = FindNodes(Import).visit(routine.parent.spec)

    assert 'kernel_geom_expr_repl_mod#kernel_geom_expr_repl' in item_map
    assert any(str(call.name).lower() == 'kernel_geom_expr_repl' for call in calls)
    assert len(calls[0].arguments) == 1
    assert str(calls[0].arguments[0]).lower() == '1'
    assert all(kw[0].lower() != 'kst' for kw in calls[0].kwarguments)
    assert any(
        ('kend_in', f'mod({expected_geom}%yrgem%ngptot, {expected_geom}%yrdim%nproma)')
        == (kw[0].lower(), str(kw[1]).lower())
        for kw in calls[0].kwarguments
    )
    assert any(
        ('kgpblks', f'{expected_geom}%yrdim%ngpblks') == (kw[0].lower(), str(kw[1]).lower())
        for kw in calls[0].kwarguments
    )
    assert any(
        ('ldacc', 'true') == (kw[0].lower(), str(kw[1]).lower())
        for kw in calls[0].kwarguments
    )
    assert any(str(imp.module).lower() == 'kernel_geom_expr_repl_mod' for imp in imports)
