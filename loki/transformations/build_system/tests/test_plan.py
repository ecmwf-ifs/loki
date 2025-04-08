# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import re

import pytest

from loki.batch import Scheduler, SchedulerConfig, ProcessingStrategy
from loki.transformations.build_system import CMakePlanTransformation, FileWriteTransformation


@pytest.mark.parametrize('use_rootpath', [False, True])
@pytest.mark.parametrize('use_fullpath', [False, True])
def test_plan_relative_paths(tmp_path, monkeypatch, use_rootpath, use_fullpath):
    """
    A test that emulates the use of overlay file systems that may cause issues
    if paths are resolved prematurely.

    This can generate file names in the lists produced by the CMakePlanTransformation
    that don't match the internal lists of files in the CMake target, thus breaking
    the source list update process.

    This test creates a file system hierarchy like this:

    - real_path/
        - module/
            - mymod.F90
        - src/
            - mysub.F90
    - overlay_path -> real_path
    - build/

    and initiates the Scheduler on the overlay path

    """
    (tmp_path/'real_path').mkdir()
    (tmp_path/'real_path/src').mkdir()
    (tmp_path/'real_path/module').mkdir()
    (tmp_path/'overlay_path').symlink_to('real_path')
    (tmp_path/'build').mkdir()

    rootpath = tmp_path if use_rootpath else None
    srcpath = f'{tmp_path}/' if use_fullpath else ''

    fcode_mymod = """
module mymod
    implicit none
    contains
        subroutine mod_sub
        end subroutine mod_sub
end module mymod
    """
    fcode_mysub = """
subroutine mysub
    use mymod, only: mod_sub
    implicit none
    call mod_sub
end subroutine mysub
    """

    (tmp_path/'real_path/src/mysub.F90').write_text(fcode_mysub)
    (tmp_path/'real_path/module/mymod.F90').write_text(fcode_mymod)

    assert (tmp_path/'overlay_path/src/mysub.F90').exists()

    # Run the test in tmp_path to be able to specify relative paths to the scheduler
    monkeypatch.chdir(tmp_path)

    # Initialize the Scheduler
    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'mode': 'test',
        },
        'routines': {
            'mysub': {'role': 'driver'}
        }
    })
    scheduler = Scheduler(
        paths=[f'{srcpath}overlay_path'],
        config=config,
        full_parse=False,
        output_dir=tmp_path/'build'
    )
    assert scheduler.items == ('#mysub', 'mymod#mod_sub')

    # Scheduler items are all relative paths
    assert scheduler['#mysub'].source.path == Path(f'{srcpath}overlay_path/src/mysub.F90')
    assert scheduler['mymod#mod_sub'].source.path == Path(f'{srcpath}overlay_path/module/mymod.F90')

    # Run the planning transformation pipeline
    scheduler.process(
        transformation=FileWriteTransformation(),
        proc_strategy=ProcessingStrategy.PLAN
    )
    plan_trafo = CMakePlanTransformation(rootpath=rootpath)
    scheduler.process(
        transformation=plan_trafo,
        proc_strategy=ProcessingStrategy.PLAN
    )
    planfile = tmp_path/'build/plan.cmake'
    plan_trafo.write_plan(planfile)

    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(planfile.read_text())}
    plan_dict = {k: [Path(p) for p in v] for k, v in plan_dict.items()}

     # The newly generated files will always have fully qualified paths
    to_append = [tmp_path/'build/mysub.test.F90', tmp_path/'build/mymod.test.F90']

    # The list of files to transform (this property is currently not used by the CMake macros)
    # will provide the original relative paths - unless we resolve them relative to a provided
    # root directory, which will also resolve symlinks
    if rootpath:
        to_transform = [Path('real_path/src/mysub.F90'), Path('real_path/module/mymod.F90')]
    else:
        to_transform = [Path(f'{srcpath}overlay_path/src/mysub.F90'), Path(f'{srcpath}overlay_path/module/mymod.F90')]

    assert plan_trafo.sources_to_append[None] == to_append
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == to_append
    assert plan_trafo.sources_to_transform[None] == to_transform
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == to_transform
    assert plan_trafo.sources_to_remove[None] == to_transform
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == to_transform
