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
# :EDITABLE:    Install Loki as an editable package (Default: ``ON``)
# :CLAW:        Install CLAW and OMNI Compiler (Default: ``ON``)
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
cmake_minimum_required( VERSION 3.17 FATAL_ERROR )
find_package( ecbuild 3.4 REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../ecbuild )

# Specify project and configuration options
project( loki VERSION 0.0.5 LANGUAGES NONE )

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
    FEATURE CLAW
    DEFAULT OFF
    DESCRIPTION "Build OMNI Compiler and CLAW Compiler"
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

    if( HAVE_CLAW )
        include( claw_compiler )
        install_claw_compiler( mlange-dev )
    endif()

    # Setup Python virtual environment
    include( python_venv )
    set( loki_VENV_PATH ${CMAKE_CURRENT_BINARY_DIR}/loki_env )
    setup_python_venv( ${loki_VENV_PATH} )
    install( DIRECTORY ${loki_VENV_PATH} DESTINATION . USE_SOURCE_PERMISSIONS )

    # Enable Pytest tests as ecbuild/ctest targets
    if( HAVE_TESTS )

        if( HAVE_CLAW )
            set( _TEST_SELECTOR "not ofp" )
            set( _TEST_PATH "${CLAW_DIR}/bin:$ENV{PATH}" )
        else()
            set( _TEST_SELECTOR "not ofp and not omni" )
            set( _TEST_PATH "$ENV{PATH}" )
        endif()

        # ecbuild_add_test relies on the variables set by the _very_ outdated
        # FindPythonInterp, so we set the bare minimum here using the values
        # from our FindPython3 variables
        set( PYTHONINTERP_FOUND True )
        set( PYTHON_EXECUTABLE ${Python3_EXECUTABLE} )

        ecbuild_add_test(
            TYPE PYTHON
            TARGET loki_tests
            ARGS -m pytest -k ${_TEST_SELECTOR} -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ENVIRONMENT PATH=${Python3_VENV_BIN}:${_TEST_PATH}
        )

        ecbuild_add_test(
            TYPE PYTHON
            TARGET loki_transformations
            ARGS -m pytest -k ${_TEST_SELECTOR} -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/transformations/transformations/tests
            ENVIRONMENT PATH=${Python3_VENV_BIN}:${_TEST_PATH}
        )

        ecbuild_add_test(
            TYPE PYTHON
            TARGET loki_lint_rules
            ARGS -m pytest -k ${_TEST_SELECTOR} -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint_rules/tests
            ENVIRONMENT PATH=${Python3_VENV_BIN}:${_TEST_PATH}
        )

        list( APPEND LOKI_INSTALL_OPTIONS "tests" )

    endif()

    # Install Loki python package with dependencies
    if( NOT ${CMAKE_VERBOSE_MAKEFILE} )
        # Less verbose output
        list ( APPEND PIP_OPTIONS "-q" )
    endif()
    if( HAVE_EDITABLE )
        # Use checked-out source instead of installing into venv
        list( APPEND PIP_OPTIONS "-e" )
    endif()

    set( _INSTALL_OPTIONS "" )
    if( LOKI_INSTALL_OPTIONS )
        list( JOIN LOKI_INSTALL_OPTIONS "," _INSTALL_OPT_STR )
        set( _INSTALL_OPTIONS "[${_INSTALL_OPT_STR}]" )
    endif()

    # We install Loki at configure time (for now), since bulk-transformation planning
    # requires configure time execution to allow injection with CMake targets.
    message( STATUS "Install Loki in virtual environment" )
    execute_process( COMMAND ${Python3_EXECUTABLE} -m pip install ${PIP_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR}${_INSTALL_OPTIONS} )
    message( STATUS "Install Loki in virtual environment - done" )

    message( STATUS "Install Loki transformations in virtual environment" )
    execute_process( COMMAND ${Python3_EXECUTABLE} -m pip install ${PIP_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/transformations )
    message( STATUS "Install Loki transformations in virtual environment - done" )

    message( STATUS "Install Loki lint_rules in virtual environment" )
    execute_process( COMMAND ${Python3_EXECUTABLE} -m pip install ${PIP_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint_rules )
    message( STATUS "Install Loki lint_rules in virtual environment - done" )

    # Make the shebang in Python scripts use relative paths
    list(
        TRANSFORM LOKI_EXECUTABLES
        PREPEND ${Python3_VENV_BIN}/
        OUTPUT_VARIABLE _venv_executables
    )
    update_python_shebang( ${_venv_executables} )

endif()

# Discover Loki executables and make available as CMake targets
include( loki_find_executables )
loki_find_executables()

# Install the project so it can be used within the bundle
ecbuild_install_project( NAME loki )

# print summary
ecbuild_print_summary()
