# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for build system interaction
"""

from pathlib import Path
import re
from subprocess import CalledProcessError

import pytest

from loki.batch import Scheduler, SchedulerConfig
from loki.frontend import available_frontends, OMNI
from loki.logging import log_levels
from loki.transformations.build_system import FileWriteTransformation


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('enable_imports', [False, True])
@pytest.mark.parametrize('import_level', ['module', 'subroutine'])
@pytest.mark.parametrize('qualified_imports', [False, True])
@pytest.mark.parametrize('use_rootpath', [False, True])
@pytest.mark.parametrize('suffix', [None, '.F90', '.Fstar'])
def test_file_write_module_imports(frontend, tmp_path, enable_imports, import_level,
                                   qualified_imports, use_rootpath, suffix):
    """
    Set up a four file mini-project with some edge cases around
    import behaviour (see in-source comments for details) and verify
    that the generated CMake plan matches the list of files we expect
    to transform, and that the FileWriteTransformation writes exactly these
    files
    """
    fcode_mod_a = """
module a_mod
    implicit none
    public
    integer :: global_a = 1
end module a_mod
"""

    fcode_mod_b = """
module b_mod
    implicit none
    public
    type type_b
        integer :: val
    end type type_b
end module b_mod
"""

    if qualified_imports:
        import_stmt = "use a_mod, only: global_a\n    use b_mod, only: type_b"
    else:
        import_stmt = "use a_mod\n    use b_mod"

    module_import_stmt = ""
    routine_import_stmt = ""
    if import_level == 'module':
        module_import_stmt = import_stmt
    elif import_level == 'subroutine':
        routine_import_stmt = import_stmt

    fcode_mod_c = f"""
module c_mod
    {module_import_stmt}
    implicit none
contains
    subroutine c(val)
        {routine_import_stmt}
        implicit none
        integer, intent(inout) :: val
        type(type_b) :: b
        b%val = global_a
        val = b%val
    end subroutine c
end module c_mod
"""

    fcode_mod_d = """
module d_mod
    implicit none
contains
    subroutine d
        use c_mod, only: c
        implicit none
        integer :: v
        call c(v)
    end subroutine d
end module d_mod
"""

    # Set-up paths and write sources
    src_path = tmp_path/'src'
    src_path.mkdir()
    out_path = tmp_path/'build'
    out_path.mkdir()

    (src_path/'a.F90').write_text(fcode_mod_a)
    (src_path/'b.F90').write_text(fcode_mod_b)
    (src_path/'c.F90').write_text(fcode_mod_c)
    (src_path/'d.F90').write_text(fcode_mod_d)

    # Expected items in the dependency graph
    expected_items = {'c_mod#c', 'd_mod#d'}

    if import_level == 'subroutine':
        if qualified_imports:
            # With qualified imports, we do not have a dependency
            # on 'b_mod' but directly on 'b_mod#type_b'
            expected_items |= {'a_mod', 'b_mod#type_b'}
        else:
            # Without qualified imports, we assume a dependency
            # for the subroutine on the imported module
            expected_items |= {'a_mod', 'b_mod'}

    elif import_level == 'module':
        if qualified_imports:
            # If we have a qualified import for the derived type
            # then we will recognize the dependency
            expected_items |= {'b_mod#type_b'}

    # Create the Scheduler
    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': enable_imports,
            'mode': 'foobar'
        },
        'routines': {'d': {'role': 'driver'}}
    })
    try:
        scheduler = Scheduler(
            paths=[src_path], config=config, frontend=frontend,
            output_dir=out_path, xmods=[tmp_path]
        )
    except CalledProcessError as e:
        all_modules_expected = 'a_mod' in expected_items and (expected_items | {'b_mod', 'b_mod#type_b'})
        if frontend == OMNI and not (enable_imports and all_modules_expected):
            # If not all header modules appear in the dependency graph, then these
            # will not be parsed by OMNI and therefore the required xmod files will
            # not be generated, thus making modules 'c' and 'd' fail at parsing
            pytest.xfail('Without parsing imports, OMNI does not have the xmod for imported modules')
        raise e

    # Check the dependency graph
    assert expected_items == {item.name for item in scheduler.items}

    # Generate the CMake plan
    plan_file = tmp_path/'plan.cmake'
    root_path = tmp_path if use_rootpath else None
    scheduler.write_cmake_plan(
        filepath=plan_file, mode=config.default['mode'], buildpath=out_path,
        rootpath=root_path
    )

    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)

    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    if enable_imports:
        # We expect to write all files that correspond to items in the graph
        expected_files = {item[0] for item in expected_items}

        if qualified_imports:
            # ...but we want to never write 'b' if we have fully qualified imports
            # because that only contains a type definition
            expected_files -= {'b'}
    else:
        # We expect to only write the subroutine files
        expected_files = {'c', 'd'}

    assert 'LOKI_SOURCES_TO_TRANSFORM' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == expected_files

    assert 'LOKI_SOURCES_TO_REMOVE' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == expected_files

    assert 'LOKI_SOURCES_TO_APPEND' in plan_dict
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {
        f'{name}.foobar' for name in expected_files
    }

    # Write the outputs
    transformation = FileWriteTransformation(
        suffix=suffix,
        include_module_var_imports=enable_imports
    )
    scheduler.process(transformation)

    # Validate the list of written files
    if suffix is None:
        suffix = '.F90'
    written_files = {f.name for f in out_path.glob('*')}
    assert written_files == {
        f'{name}.foobar{suffix}' for name in expected_files
    }


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('have_non_replicate_conflict', [False, True])
def test_file_write_replicate(tmp_path, caplog, frontend, have_non_replicate_conflict):
    fcode_a = """
module a_mod
    implicit none
    integer :: a
end module a_mod
    """
    fcode_b = """
module b_mod
    implicit none
    integer :: b
end module b_mod
    """
    if have_non_replicate_conflict:
        other_routine = "subroutine not_c()\n    end subroutine not_c"
    else:
        other_routine = ""
    fcode_c = f"""
module c_mod
contains
    subroutine c(val)
        use a_mod, only: a
        use b_mod, only: b
        integer, intent(inout) :: val
        val = val + a + b
    end subroutine c
    {other_routine}
end module c_mod
    """
    fcode_d = """
subroutine d()
    use c_mod, only: c
    implicit none
    integer :: var
    call c(var)
end subroutine d
    """


    # Set-up paths and write sources
    src_path = tmp_path/'src'
    src_path.mkdir()
    out_path = tmp_path/'build'
    out_path.mkdir()

    (src_path/'a.F90').write_text(fcode_a)
    (src_path/'b.F90').write_text(fcode_b)
    (src_path/'c.F90').write_text(fcode_c)
    (src_path/'d.F90').write_text(fcode_d)

    # Expected items in the dependency graph
    expected_items = {'a_mod', 'b_mod', 'c_mod#c', '#d'}

    if have_non_replicate_conflict:
        expected_items |= {'c_mod#not_c'}

    # Create the Scheduler
    config = SchedulerConfig.from_dict({
        'default': {
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': True,
            'mode': 'foobar',
            'replicate': True
        },
        'routines': {
            'b_mod': {'replicate': False},
            'not_c': {'replicate': False},
            'd': {'role': 'driver', 'replicate': False},
        }
    })

    scheduler = Scheduler(
        paths=[src_path], config=config, frontend=frontend,
        output_dir=out_path, xmods=[tmp_path]
    )

    # Check the dependency graph
    assert expected_items == {item.name for item in scheduler.items}

    # Generate the CMake plan
    plan_file = tmp_path/'plan.cmake'

    caplog.clear()
    with caplog.at_level(log_levels['WARNING']):
        scheduler.write_cmake_plan(
            filepath=plan_file, mode=config.default['mode'], buildpath=out_path,
            rootpath=tmp_path
        )
        if have_non_replicate_conflict:
            assert len(caplog.records) == 1
            assert 'c.f90' in caplog.records[0].message
            assert 'c_mod#not_c' in caplog.records[0].message
        else:
            assert not caplog.records


    # Validate the plan file content
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)

    loki_plan = plan_file.read_text()
    plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
    plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

    assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == {'a', 'b', 'c', 'd'}
    assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == {'b', 'd'}
    assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {'a.foobar', 'b.foobar', 'c.foobar', 'd.foobar'}

    # Write the outputs
    transformation = FileWriteTransformation(include_module_var_imports=True)
    scheduler.process(transformation)

    # Validate the list of written files
    written_files = {f.name for f in out_path.glob('*')}
    assert written_files == {'a.foobar.F90', 'b.foobar.F90', 'c.foobar.F90', 'd.foobar.F90'}
