"""
Functional tests for cmake macros.
"""

from pathlib import Path
import re
import shutil
from subprocess import CalledProcessError
import pytest
import toml

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


@pytest.fixture(scope='module', name='srcdir')
def fixture_srcdir(here):
    """Base directory of CMake sources"""
    return here/'sources'


@pytest.fixture(scope='module', name='builddir')
def fixture_builddir():
    """
    Create a build directory in the temp directory
    """
    builddir = gettempdir()/'test_cmake'
    if builddir.exists():
        shutil.rmtree(builddir)
    builddir.mkdir()
    yield builddir
    shutil.rmtree(builddir)


@pytest.fixture(scope='module', name='config')
def fixture_config(here):
    """
    Write default configuration as a temporary file and return
    the file path
    """
    default_config = {
        'default': {'role': 'kernel', 'expand': True, 'strict': True},
        'routine': [
            {'name': 'driverB', 'role': 'driver'},
        ]
    }
    filepath = here/'test_cmake_loki.config'
    filepath.write_text(toml.dumps(default_config))
    yield filepath
    filepath.unlink()


@pytest.fixture(scope='module', name='cmake_project')
def fixture_cmake_project(here, config, srcdir, builddir):
    """
    Create a CMake project and set-up paths
    """
    proj_a = '${CMAKE_CURRENT_SOURCE_DIR}/projA'
    proj_b = '${CMAKE_CURRENT_SOURCE_DIR}/projB'

    file_content = f"""
cmake_minimum_required( VERSION 3.17 FATAL_ERROR )

project( cmake_test LANGUAGES Fortran )

include( FetchContent )
FetchContent_Declare( ecbuild
  GIT_REPOSITORY    https://github.com/ecmwf/ecbuild.git
  GIT_TAG           master
)
FetchContent_MakeAvailable( ecbuild )

set( LOKI_ENABLE_CLAW OFF )
set( LOKI_ENABLE_NO_INSTALL ON )
add_subdirectory( loki )
find_package( loki )

loki_transform_plan(
    MODE      idem
    FRONTEND  fp
    CONFIG    {config}
    SOURCEDIR ${{CMAKE_CURRENT_SOURCE_DIR}}
    CALLGRAPH {builddir}/loki_callgraph
    PLAN      {builddir}/loki_plan.cmake
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


def test_cmake_plan(srcdir, builddir, config, cmake_project):
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

    execute(['cmake', str(srcdir)], cwd=builddir)
    assert (builddir/'loki_plan.cmake').exists()
    assert (builddir/'loki_callgraph.pdf').exists()

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
