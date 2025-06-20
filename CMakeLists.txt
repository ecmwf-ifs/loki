# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

##############################################################################
#.rst:
#
# loki
# ====
#
# Install Loki with dependencies. ::
#
# Features
# --------
#
# :NO_INSTALL:  Do not install Loki itself but make the CMake configuration
#               available (Default: ``OFF``)
# :EDITABLE:    Install Loki as an editable package (Default: ``OFF``)
#
# Installation procedure
# ----------------------
#
# A virtual environment is created for Loki into which it is installed along
# with any dependencies. The CLI scripts ``loki-transform.py`` and ``loki-lint.py``
# are made available as executable targets, thus can be used from any subsequent
# ``add_custom_command`` statements.
#
##############################################################################

# Version 3.12 required to use FindPython
# Version 3.15 officially required to use Python3_FIND_VIRTUALENV (not working on 3.15.3,
# though, and use 3.17 for conda support anyway)
# Version 3.19 for support of find_package version range and file(CHMOD)
cmake_minimum_required( VERSION 3.19 FATAL_ERROR )
find_package( ecbuild 3.7 REQUIRED )

# Specify project and configuration options
project( loki LANGUAGES NONE )

# Allow negating ENABLE_NO_INSTALL with a leading '~'
macro( apply_negation VAR_NAME )
    if( DEFINED ${VAR_NAME} )
        if( ${${VAR_NAME}} MATCHES ^~ )
            string( REPLACE ~ "" ${VAR_NAME} ${${VAR_NAME}} )
            if( ${${VAR_NAME}} )
                set( ${VAR_NAME} OFF )
            else()
                set( ${VAR_NAME} ON )
            endif()
        endif()
    endif()
endmacro()

apply_negation( ENABLE_NO_INSTALL )
apply_negation( LOKI_ENABLE_NO_INSTALL )

# Declare options
ecbuild_add_option(
    FEATURE NO_INSTALL
    DEFAULT OFF
    DESCRIPTION "Disable Loki (and dependency) installation"
)
ecbuild_add_option(
    FEATURE EDITABLE
    DEFAULT OFF
    DESCRIPTION "Install Loki as an editable Python package"
)
ecbuild_add_option(
    FEATURE OMNI
    DEFAULT OFF
    DESCRIPTION "Build OMNI compiler as Loki frontend"
)

include( loki_transform )

# Make CMake script files available in build and install directory
add_subdirectory( cmake )
install( DIRECTORY cmake DESTINATION ${INSTALL_DATA_DIR} PATTERN "CMakeLists.txt" EXCLUDE )

# The list of Loki frontend scripts
file( GLOB _LOKI_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/loki_*.py" )
list( TRANSFORM _LOKI_SCRIPTS REPLACE "scripts/loki_" "scripts/loki-" )
set( LOKI_EXECUTABLES "" )
foreach( _exe IN LISTS _LOKI_SCRIPTS )
    get_filename_component( _exe_name ${_exe} NAME )
    list( APPEND LOKI_EXECUTABLES ${_exe_name} )
endforeach()

# Install Loki and dependencies
if( NOT HAVE_NO_INSTALL )

    if( HAVE_OMNI )
        include( omni_compiler )
        install_omni_compiler( master )
    endif()

    # Setup Python virtual environment
    include( loki_python_macros )
    set( PYTHON_VERSION 3.9 )
    loki_setup_python_venv(
        VENV_NAME loki_env
        PYTHON_VERSION ${PYTHON_VERSION}
        INSTALL_VENV
    )

    # Enable Pytest tests as ecbuild/ctest targets
    if( HAVE_TESTS )

        if( HAVE_OMNI )
            set( _TEST_SELECTOR "not ofp" )
            set( _TEST_PATH "${OMNI_DIR}/bin:$ENV{PATH}" )
        else()
            set( _TEST_SELECTOR "not ofp and not omni" )
            set( _TEST_PATH "$ENV{PATH}" )
        endif()

        # Nesting the CMake tests into CTest does not correctly resolve
        # search paths, therefore these are getting disabled here
        set( _TEST_SELECTOR "${_TEST_SELECTOR} and not cmake")

        # ecbuild_add_test relies on the variables set by the _very_ outdated
        # FindPythonInterp, so we set the bare minimum here using the values
        # from our FindPython3 variables
        set( PYTHONINTERP_FOUND True )
        set( PYTHON_EXECUTABLE ${Python3_EXECUTABLE} )

        ecbuild_add_test(
            TYPE PYTHON
            TARGET loki_tests
            ARGS -m pytest -k ${_TEST_SELECTOR} -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            ENVIRONMENT PATH=${Python3_VENV_BIN}:${_TEST_PATH}
        )

        ecbuild_add_test(
            TYPE PYTHON
            TARGET loki_lint_rules
            ARGS -m pytest -k ${_TEST_SELECTOR} -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lint_rules
            ENVIRONMENT PATH=${Python3_VENV_BIN}:${_TEST_PATH}
        )

        list( APPEND LOKI_INSTALL_OPTIONS "tests" )

    endif()

    # Determine whether this is a Git worktree or if we have to provide
    # the version number to setuptools_scm
    if( NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git )
        set( ENV{SETUPTOOLS_SCM_PRETEND_VERSION} ${loki_VERSION} )
    endif()

    # Install Loki python package with dependencies
    set( _INSTALL_OPTIONS "" )
    if( LOKI_INSTALL_OPTIONS )
        list( JOIN LOKI_INSTALL_OPTIONS "," _INSTALL_OPT_STR )
        set( _INSTALL_OPTIONS "[${_INSTALL_OPT_STR}]" )
    endif()

    # Optionally use the ARTIFACTS_DIR as wheelhouse, if provided
    if( DEFINED ARTIFACTS_DIR )
        set( WHEELS_DIR_OPTION WHEELS_DIR "${ARTIFACTS_DIR}" )
    else()
        set( WHEELS_DIR_OPTION "" )
    endif()

    if( HAVE_EDITABLE )
       set( EDITABLE_OPTION "EDITABLE" )
    else()
       set( EDITABLE_OPTION "" )
    endif()

    # We install Loki at configure time (for now), since bulk-transformation planning
    # requires configure time execution to allow injection with CMake targets.

    ecbuild_info( "Install Loki in virtual environment" )
    loki_install_python_package(
        REQUIREMENT_SPEC ${CMAKE_CURRENT_SOURCE_DIR}${_INSTALL_OPTIONS}
        ${EDITABLE_OPTION}
        ${WHEELS_DIR_OPTION}
    )
    loki_install_python_package(
        REQUIREMENT_SPEC ${CMAKE_CURRENT_SOURCE_DIR}/lint_rules
        ${EDITABLE_OPTION}
        ${WHEELS_DIR_OPTION}
    )
    ecbuild_info( "Install Loki in virtual environment - done" )

endif()

# Discover Loki executables and make available as CMake targets
include( loki_find_executables )
loki_find_executables()

# Install the project so it can be used within the bundle
ecbuild_install_project( NAME loki )

# print summary
ecbuild_print_summary()
