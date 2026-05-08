# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from pathlib import Path

import pytest

from loki import Scheduler, SchedulerConfig
from loki.batch import Pipeline, ProcessingStrategy
from loki.frontend import OMNI, available_frontends
from loki.ir import CallStatement, FindNodes, Import
from loki.transformations import IdemTransformation, ReplaceKernels
from loki.transformations.build_system import (
    DependencyTransformation, FileWriteTransformation, ModuleWrapTransformation,
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def _read_plan_dict(plan_file):
    """Parse the generated CMake plan into a stem-only dictionary for assertions."""
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    loki_plan = plan_file.read_text()
    plan_dict = {key: value.split() for key, value in plan_pattern.findall(loki_plan)}
    return {key: {Path(item).stem for item in value} for key, value in plan_dict.items()}


def _scheduler_config(driver_names):
    """Create a minimal scheduler config with the selected routines marked as drivers."""
    return SchedulerConfig.from_dict({
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': False,
        },
        'routines': {name: {'role': 'driver'} for name in driver_names},
    })


def _write_replace_sources(tmp_path):
    """Write anonymized Fortran sources that cover the replacement-call patterns under test."""
    # These fixtures mirror the relevant call-shape patterns we care about for
    # PR 1 without pulling in real IFS names or source files.
    sources = {
        'geom_types_mod.F90': """
module geom_types_mod
  implicit none
  type dim_type
    integer :: nproma
    integer :: ngpblks
  end type dim_type
  type total_type
    integer :: ngptot
  end type total_type
  type geometry_type
    type(dim_type) :: dim
    type(total_type) :: total
  end type geometry_type
end module geom_types_mod
""".strip(),
        'leaf_mod.F90': """
module leaf_mod
implicit none
contains
subroutine leaf(flag1, flag2, start_idx, end_idx, geom, field)
  use geom_types_mod, only: geometry_type
  logical, intent(in) :: flag1, flag2
  integer, intent(in) :: start_idx, end_idx
  type(geometry_type), intent(in) :: geom
  real, intent(inout) :: field(:)
  field(start_idx:end_idx) = field(start_idx:end_idx) + 1.0
end subroutine leaf
end module leaf_mod
""".strip(),
        'leaf_replacement_mod.F90': """
module leaf_replacement_mod
implicit none
contains
subroutine leaf_replacement(flag1, flag2_renamed, field, start_idx, block_count, end_in, on_gpu, optional_flag)
  logical, intent(in) :: flag1, flag2_renamed
  real, intent(inout) :: field(:)
  integer, intent(in) :: start_idx, block_count, end_in
  logical, intent(in) :: on_gpu
  logical, intent(in), optional :: optional_flag
  field(start_idx:end_in) = field(start_idx:end_in) + real(block_count)
end subroutine leaf_replacement
end module leaf_replacement_mod
""".strip(),
        'leaf_passthrough_replacement_mod.F90': """
module leaf_passthrough_replacement_mod
implicit none
contains
subroutine leaf_passthrough_replacement(flag1, flag2, start_idx, end_idx, geom, field, optional_flag)
  use geom_types_mod, only: geometry_type
  logical, intent(in) :: flag1, flag2
  integer, intent(in) :: start_idx, end_idx
  type(geometry_type), intent(in) :: geom
  real, intent(inout) :: field(:)
  logical, intent(in), optional :: optional_flag
  field(start_idx:end_idx) = field(start_idx:end_idx) + real(geom%dim%ngpblks)
end subroutine leaf_passthrough_replacement
end module leaf_passthrough_replacement_mod
""".strip(),
        'include_driver.F90': """
module include_driver_mod
implicit none
contains
subroutine include_driver(start_idx, end_idx, geom, field)
  use geom_types_mod, only: geometry_type
  implicit none
  integer, intent(in) :: start_idx, end_idx
  type(geometry_type), intent(in) :: geom
  real, intent(inout) :: field(:)
#include "leaf.intfb.h"
  call leaf(.false., .true., start_idx, end_idx, geom, field)
end subroutine include_driver
end module include_driver_mod
""".strip(),
        'leaf.intfb.h': '#include "leaf_mod.intfb.h"\n',
        'pipeline_driver_mod.F90': """
module pipeline_driver_mod
implicit none
contains
subroutine tile_driver_parallel(span, geom, field)
  use geom_types_mod, only: geometry_type
  use pipeline_band_mod, only: band_kernel
  implicit none
  type span_type
    integer :: begin_idx
    integer :: end_idx
  end type span_type
  type(span_type), intent(in) :: span
  type(geometry_type), intent(in) :: geom
  real, intent(inout) :: field(:)
  call band_kernel(span, geom, field)
end subroutine tile_driver_parallel
end module pipeline_driver_mod
""".strip(),
        'pipeline_band_mod.F90': """
module pipeline_band_mod
implicit none
contains
subroutine band_kernel(span, geom, field)
  use geom_types_mod, only: geometry_type
  use edge_wrapper_mod, only: edge_wrapper
  implicit none
  type span_type
    integer :: begin_idx
    integer :: end_idx
  end type span_type
  type(span_type), intent(in) :: span
  type(geometry_type), intent(in) :: geom
  real, intent(inout) :: field(:)
  call edge_wrapper(span%begin_idx, span%end_idx, geom, field)
end subroutine band_kernel
end module pipeline_band_mod
""".strip(),
        'edge_wrapper_mod.F90': """
module edge_wrapper_mod
implicit none
contains
subroutine edge_wrapper(start_idx, end_idx, geom, field)
  use geom_types_mod, only: geometry_type
  use leaf_mod, only: leaf
  implicit none
  integer, intent(in) :: start_idx, end_idx
  type(geometry_type), intent(in) :: geom
  real, intent(inout) :: field(:)
  call leaf(.true., .false., start_idx, end_idx, geom, field)
end subroutine edge_wrapper
end module edge_wrapper_mod
""".strip(),
    }
    for name, code in sources.items():
        (tmp_path/name).write_text(code)


def _get_item_map(scheduler):
    """Index scheduler items by fully qualified item name for direct lookup in tests."""
    return {item.name: item for item in scheduler.items}


def _get_routine_and_calls(item_map, item_name):
    """Return one routine together with its calls and local or parent imports."""
    item = item_map[item_name]
    routine = item.source[item.local_name]
    calls = FindNodes(CallStatement).visit(routine.body)
    imports = FindNodes(Import).visit(routine.spec)
    parent_imports = ()
    if routine.parent is not None and getattr(routine.parent, 'spec', None) is not None:
        parent_imports = FindNodes(Import).visit(routine.parent.spec)
    return routine, calls, imports, parent_imports


@pytest.fixture(name='replace_source_dir')
def fixture_replace_source_dir(tmp_path):
    _write_replace_sources(tmp_path)
    return tmp_path


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_argument_remap(frontend, replace_source_dir, tmp_path):
    """Remap replacement arguments across rename, member, template, position, and literal rules."""
    # This replacement exercises the full remapping surface in one focused test.
    scheduler = Scheduler(
        paths=[replace_source_dir],
        config=_scheduler_config(('include_driver', 'tile_driver_parallel')),
        frontend=frontend,
        xmods=[tmp_path],
        includes=[replace_source_dir],
    )
    scheduler.process(transformation=ReplaceKernels({
        'leaf': {
            'routine': 'leaf_replacement',
            'args': {
                'flag2': 'flag2_renamed',
                'geom': {'map_to': 'block_count', 'member': 'DIM%NGPBLKS'},
                'end_idx': {
                    'map_to': 'end_in',
                    'placeholders': {'geom': 'geom'},
                    'expr': 'MOD({geom}%TOTAL%NGPTOT, {geom}%DIM%NPROMA)',
                },
                'start_idx': {'position': 1},
                'on_gpu': '.true.',
            },
        }
    }))

    item_map = _get_item_map(scheduler)
    assert 'leaf_replacement_mod#leaf_replacement' in item_map

    _, calls, imports, parent_imports = _get_routine_and_calls(item_map, 'include_driver_mod#include_driver')
    call = calls[0]

    assert call.name == 'leaf_replacement'
    assert len(call.arguments) == 1
    assert call.arguments[0] == 'false'
    assert ('flag2_renamed', 'true') in call.kwarguments
    assert ('field', 'field') in call.kwarguments
    assert ('start_idx', 1) in call.kwarguments
    assert ('block_count', 'geom%dim%ngpblks') in call.kwarguments
    assert ('end_in', 'mod(geom%total%ngptot, geom%dim%nproma)') in call.kwarguments
    assert ('on_gpu', 'true') in call.kwarguments
    assert any(imp.module == 'leaf_replacement_mod' for imp in imports + parent_imports)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_omits_optional_argument(frontend, replace_source_dir, tmp_path, caplog):
    """Skip unmapped optional replacement arguments and emit a warning for the omission."""
    scheduler = Scheduler(
        paths=[replace_source_dir],
        config=_scheduler_config(('include_driver',)),
        frontend=frontend,
        xmods=[tmp_path],
        includes=[replace_source_dir],
    )

    with caplog.at_level('WARNING'):
        # The passthrough replacement keeps the original dummy names so this
        # test isolates optional-argument omission from remapping behavior.
        scheduler.process(transformation=ReplaceKernels({
            'leaf': {'routine': 'leaf_passthrough_replacement'}
        }))

    item_map = _get_item_map(scheduler)
    _, calls, _, _ = _get_routine_and_calls(item_map, 'include_driver_mod#include_driver')
    assert all(name.lower() != 'optional_flag' for name, _ in calls[0].kwarguments)
    assert any('optional replacement argument' in message for message in caplog.messages)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_rewrites_intfb_include(frontend, replace_source_dir, tmp_path):
    """Replace include-style imports with the replacement module import when needed."""
    scheduler = Scheduler(
        paths=[replace_source_dir],
        config=_scheduler_config(('include_driver',)),
        frontend=frontend,
        xmods=[tmp_path],
        includes=[replace_source_dir],
    )
    scheduler.process(transformation=ReplaceKernels({
        'leaf': {'routine': 'leaf_passthrough_replacement'}
    }))

    item_map = _get_item_map(scheduler)
    routine, calls, imports, parent_imports = _get_routine_and_calls(item_map, 'include_driver_mod#include_driver')

    assert calls[0].name == 'leaf_passthrough_replacement'
    c_imports = tuple(imp for imp in FindNodes(Import).visit(routine.spec) if imp.c_import)
    assert not any(imp.module and 'leaf.intfb.h' in imp.module.lower() for imp in c_imports)
    assert any(imp.module == 'leaf_passthrough_replacement_mod' for imp in imports + parent_imports)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
@pytest.mark.parametrize('block_replacement', [False, True])
def test_replace_kernel_updates_plan_and_block_list(frontend, replace_source_dir, tmp_path, block_replacement):
    """Block flagged replacements from traversal and generated plan output while still rewriting calls."""
    output_dir = tmp_path/'build'
    output_dir.mkdir()
    config = _scheduler_config(('include_driver',))
    replace_map = {'leaf': {'routine': 'leaf_passthrough_replacement', 'block': block_replacement}}
    pipeline = Pipeline(classes=(
        IdemTransformation,
        ReplaceKernels,
        ModuleWrapTransformation,
        DependencyTransformation,
        FileWriteTransformation,
    ), replace_kernels_map=replace_map, module_suffix='_mod', suffix='_test')

    scheduler = Scheduler(
        paths=[replace_source_dir], config=config, frontend=frontend, xmods=[tmp_path], output_dir=output_dir,
        includes=[replace_source_dir],
    )
    plan_file = tmp_path/'replace_plan.cmake'
    scheduler.process(pipeline, proc_strategy=ProcessingStrategy.PLAN)
    scheduler.write_cmake_plan(filepath=plan_file, rootpath=replace_source_dir)
    plan_dict = _read_plan_dict(plan_file)
    item_names = {item.name for item in scheduler.items}

    if block_replacement:
        assert 'leaf_passthrough_replacement_mod#leaf_passthrough_replacement' not in item_names
        assert 'leaf_passthrough_replacement_mod' not in plan_dict['LOKI_SOURCES_TO_TRANSFORM']
    else:
        assert 'leaf_passthrough_replacement_mod#leaf_passthrough_replacement' in item_names
        assert 'leaf_passthrough_replacement_mod' in plan_dict['LOKI_SOURCES_TO_TRANSFORM']


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI module type definitions not available')]))
def test_replace_kernel_pipeline_fixture(frontend, replace_source_dir, tmp_path):
    """Rewrite a downstream kernel call inside a small pipeline-shaped anonymized call chain."""
    scheduler = Scheduler(
        paths=[replace_source_dir],
        config=_scheduler_config(('tile_driver_parallel',)),
        frontend=frontend,
        xmods=[tmp_path],
        includes=[replace_source_dir],
    )
    scheduler.process(transformation=ReplaceKernels({
        'leaf': {
            'routine': 'leaf_replacement',
            'args': {
                'flag2': 'flag2_renamed',
                'geom': {'map_to': 'block_count', 'member': 'DIM%NGPBLKS'},
                'end_idx': 'end_in',
                'on_gpu': '.false.',
            },
        }
    }))

    item_map = _get_item_map(scheduler)
    _, calls, imports, _ = _get_routine_and_calls(item_map, 'edge_wrapper_mod#edge_wrapper')

    assert len(calls) == 1
    assert calls[0].name == 'leaf_replacement'
    assert ('block_count', 'geom%dim%ngpblks') in calls[0].kwarguments
    assert any(imp.module == 'leaf_replacement_mod' for imp in imports)
