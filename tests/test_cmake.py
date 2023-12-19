# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Functional tests for cmake macros.
"""

from pathlib import Path
import re
import shutil
from subprocess import CalledProcessError
from contextlib import contextmanager
import pytest
import toml

from conftest import graphviz_present
from loki import (
    gettempdir, execute
)


def check_cmake():
    """
    Check if CMake is available
    """
    # TODO: Check CMake version
    try:
        execute(['cmake', '--version'], silent=True)
    except CalledProcessError:
        return False
    return True


pytest.mark.skipif(not check_cmake(), reason='CMake not available')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    """Current test directory"""
    return Path(__file__).parent


@pytest.fixture(scope='module', name='silent')
def fixture_silent(pytestconfig):
    """Whether to run commands without output"""
    return pytestconfig.getoption("verbose") == 0


@pytest.fixture(scope='module', name='srcdir')
def fixture_srcdir(here):
    """Base directory of CMake sources"""
    return here/'sources'


@pytest.fixture(scope='module', name='config')
def fixture_config(here):
    """
    Write default configuration as a temporary file and return
    the file path
    """
    default_config = {
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routines': {
            'driverB': {'role': 'driver'},
        }
    }
    filepath = here/'test_cmake_loki.config'
    filepath.write_text(toml.dumps(default_config))
    yield filepath
    filepath.unlink()


@pytest.fixture(scope='module', name='ecbuild')
def fixture_ecbuild():
    """
    Download and install ecbuild
    """
    srcdir = gettempdir()/'ecbuild_tmp'
    if srcdir.exists():
        shutil.rmtree(srcdir)
    ecbuilddir = gettempdir()/'ecbuild'
    if ecbuilddir.exists():
        shutil.rmtree(ecbuilddir)

    execute(['git', 'clone', 'https://github.com/ecmwf/ecbuild.git', str(srcdir)])
    (srcdir/'bootstrap').mkdir()
    execute(['cmake', '..'], cwd=srcdir/'bootstrap')
    execute(['cmake', '--install', '.', '--prefix', str(ecbuilddir)], cwd=srcdir/'bootstrap')

    shutil.rmtree(srcdir)
    yield ecbuilddir
    shutil.rmtree(ecbuilddir)


@pytest.fixture(scope='module', name='loki_install', params=[True, False])
def fixture_loki_install(here, ecbuild, silent, request):
    """
    Install Loki using CMake into an install directory
    """
    builddir = gettempdir()/'loki_bootstrap'
    if builddir.exists():
        shutil.rmtree(builddir)
    builddir.mkdir()
    cmd = [f'{ecbuild}/bin/ecbuild', str(here.parent), '-DENABLE_CLAW=OFF']
    if request.param:
        cmd += ['-DENABLE_EDITABLE=ON']
    else:
        cmd += ['-DENABLE_EDITABLE=OFF']
    execute(cmd, silent=silent, cwd=builddir)

    lokidir = gettempdir()/'loki'
    if lokidir.exists():
        shutil.rmtree(lokidir)
    execute(
        ['cmake', '--install', '.', '--prefix', str(lokidir)],
        silent=True, cwd=builddir
    )

    yield builddir, lokidir
    if builddir.exists():
        shutil.rmtree(builddir)
    if lokidir.exists():
        shutil.rmtree(lokidir)


@contextmanager
def clean_builddir(name):
    """
    Create a build directory in the temp directory
    """
    builddir = gettempdir()/str(name)
    if builddir.exists():
        shutil.rmtree(builddir)
    builddir.mkdir()
    yield builddir
    shutil.rmtree(builddir)


@pytest.fixture(scope='module', name='cmake_project')
def fixture_cmake_project(here, config, srcdir):
    """
    Create a CMake project and set-up paths
    """
    proj_a = '${CMAKE_CURRENT_SOURCE_DIR}/projA'
    proj_b = '${CMAKE_CURRENT_SOURCE_DIR}/projB'

    file_content = f"""
cmake_minimum_required( VERSION 3.19 FATAL_ERROR )
find_package( ecbuild REQUIRED )

project( cmake_test VERSION 1.0.0 LANGUAGES Fortran )

ecbuild_find_package( loki REQUIRED )

loki_transform_plan(
    MODE      idem
    FRONTEND  fp
    CONFIG    {config}
    SOURCEDIR ${{CMAKE_CURRENT_SOURCE_DIR}}
    CALLGRAPH ${{CMAKE_CURRENT_BINARY_DIR}}/loki_callgraph
    PLAN      ${{CMAKE_CURRENT_BINARY_DIR}}/loki_plan.cmake
    SOURCES
        {proj_a}
        {proj_b}
    HEADERS
        {proj_a}/module/header_mod.f90
)
    """
    filepath = srcdir/'CMakeLists.txt'
    filepath.write_text(file_content)

    # Create a symlink to loki
    (srcdir/'loki').symlink_to(here.parent)

    yield filepath

    filepath.unlink()
    (srcdir/'loki').unlink()


def test_cmake_plan(srcdir, config, cmake_project, loki_install, ecbuild, silent):
    """
    Test the `loki_transform_plan` CMake function with a single task
    graph spanning two projects

    projA: driverB -> kernelB -> compute_l1<replicated> -> compute_l2
                         |
    projB:          ext_driver -> ext_kernel
    """
    plan_pattern = re.compile(r'set\(\s*(\w+)\s*(.*?)\s*\)', re.DOTALL)

    assert config.exists()
    assert cmake_project.exists()

    for loki_root in loki_install:
        with clean_builddir('test_cmake_plan') as builddir:
            execute(
                [f'{ecbuild}/bin/ecbuild', str(srcdir), f'-Dloki_ROOT={loki_root}'],
                cwd=builddir, silent=silent
            )

            # Make sure the plan files have been created
            assert (builddir/'loki_plan.cmake').exists()
            if graphviz_present():
                assert (builddir/'loki_callgraph.pdf').exists()

            # Validate the content of the plan file
            loki_plan = (builddir/'loki_plan.cmake').read_text()
            plan_dict = {k: v.split() for k, v in plan_pattern.findall(loki_plan)}
            plan_dict = {k: {Path(s).stem for s in v} for k, v in plan_dict.items()}

            expected_files = {
                'driverB_mod', 'kernelB_mod',
                'compute_l1_mod', 'compute_l2_mod',
                'ext_driver_mod', 'ext_kernel'
            }

            assert 'LOKI_SOURCES_TO_TRANSFORM' in plan_dict
            assert plan_dict['LOKI_SOURCES_TO_TRANSFORM'] == expected_files

            assert 'LOKI_SOURCES_TO_REMOVE' in plan_dict
            assert plan_dict['LOKI_SOURCES_TO_REMOVE'] == expected_files

            assert 'LOKI_SOURCES_TO_APPEND' in plan_dict
            assert plan_dict['LOKI_SOURCES_TO_APPEND'] == {
                f'{name}.idem' for name in expected_files
            }

        shutil.rmtree(loki_root)
