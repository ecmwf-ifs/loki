# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Functional tests for cmake macros.
"""

import os
from pathlib import Path
import re
import shutil
from subprocess import CalledProcessError
from contextlib import contextmanager
import pytest
import toml

from loki import gettempdir, execute, graphviz_present


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

@pytest.fixture(scope='module', name='tmp_dir')
def fixture_tmp_dir():
    """Return a test module lifetime tmp directory"""
    tmp_dir = gettempdir()/'test_cmake'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    yield tmp_dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


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
def fixture_config(tmp_dir):
    """
    Write default configuration as a temporary file and return
    the file path
    """
    default_config = {
        'default': {
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': True
        },
        'routines': {
            'driverB': {'role': 'driver'},
        },
        'transformations': {
            'IdemTrafo': {
                'classname': 'IdemTransformation',
                'module': 'loki.transformations.idempotence',
            },
            'FileWriteTransformation': {
                'classname': 'FileWriteTransformation',
                'module': 'loki.transformations.build_system',
                'options': {
                    'include_module_var_imports': True
                }
            }
        },
        'pipelines': {
            'idem': {
                'transformations': ['IdemTrafo']
            }
        }
    }
    filepath = tmp_dir/'test_cmake_loki.config'
    filepath.write_text(toml.dumps(default_config))
    yield filepath
    filepath.unlink()


@pytest.fixture(scope='module', name='ecbuild')
def fixture_ecbuild(tmp_dir):
    """
    Download ecbuild
    """
    ecbuilddir = tmp_dir/'ecbuild'
    if ecbuilddir.exists():
        shutil.rmtree(ecbuilddir)
    execute(['git', 'clone', 'https://github.com/ecmwf/ecbuild.git', str(ecbuilddir)])
    yield ecbuilddir
    shutil.rmtree(ecbuilddir)


@pytest.fixture(scope='module', name='loki_artifacts_and_env', params=[True, False])
def fixture_loki_artifacts_and_env(here, tmp_dir, silent, request):
    """
    Download wheels using the populate mechanism and provide the artifacts dir
    """
    artifacts_dir = tmp_dir/'artifacts'
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)

    cmake_args = []
    env = os.environ.copy()
    if request.param:
        env['ARTIFACTS_DIR'] = str(artifacts_dir)
        env['LOKI_INSTALL_OPTIONS'] = '[tests]'
        execute(['./populate'], silent=silent, cwd=str(here.parent.parent), env=env)
        cmake_args += [f'-DARTIFACTS_DIR={artifacts_dir}']
        # Set http_proxy and https_proxy to nonsense, which should prevent PIP from connecting
        # to a package index during the configure step
        env['http_proxy'] = 'http://foo.bar.baz'
        env['https_proxy'] = 'http://foo.bar.baz'

    yield cmake_args, env

    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)


@pytest.fixture(scope='module', name='loki_install', params=['editable', 'relative_install', 'default'])
def fixture_loki_install(here, tmp_dir, ecbuild, loki_artifacts_and_env, silent, request):
    """
    Install Loki using CMake into an install directory
    """
    builddir = tmp_dir/'loki_bootstrap'
    installdir = tmp_dir/'loki'
    artifacts_arg, env = loki_artifacts_and_env
    cmd = [
        'cmake', f'-DCMAKE_MODULE_PATH={ecbuild}/cmake',
        '-S', str(here.parent.parent),
        '-B', str(builddir)
    ]
    cmd += artifacts_arg
    if request.param == 'editable':
        cmd += ['-DENABLE_EDITABLE=ON']
    else:
        cmd += ['-DENABLE_EDITABLE=OFF']

    execute(cmd, silent=silent, cwd=tmp_dir, env=env)

    if request.param == 'relative_install':
        prefix = 'loki'
    else:
        prefix = installdir
    execute(
        ['cmake', '--install', str(builddir), '--prefix', str(prefix)],
        silent=True, cwd=tmp_dir, env=env
    )

    yield builddir, installdir


@contextmanager
def clean_builddir(builddir):
    """
    Clean the build directory in the temp directory
    """
    builddir = Path(builddir)
    if builddir.exists():
        shutil.rmtree(builddir)
    builddir.mkdir()
    yield builddir


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
    CONFIG    {config}
    SOURCEDIR ${{CMAKE_CURRENT_SOURCE_DIR}}
    CALLGRAPH ${{CMAKE_CURRENT_BINARY_DIR}}/loki_callgraph
    PLAN      ${{CMAKE_CURRENT_BINARY_DIR}}/loki_plan.cmake
    SOURCES
        {proj_a}
        {proj_b}
)
    """
    filepath = srcdir/'CMakeLists.txt'
    filepath.write_text(file_content)

    # Create a symlink to loki
    (srcdir/'loki').symlink_to(here.parent)

    yield filepath

    filepath.unlink()
    (srcdir/'loki').unlink()


def test_cmake_plan(srcdir, tmp_dir, config, cmake_project, loki_install, ecbuild, silent):
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
        with clean_builddir(tmp_dir/'test_cmake_plan') as builddir:
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
                'ext_driver_mod', 'ext_kernel',
                'header_mod'
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
