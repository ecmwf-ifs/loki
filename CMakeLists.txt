# Version 3.12 required to use FindPython
# Version 3.15 required to use Python3_FIND_VIRTUALENV (use 3.17 for conda support)
cmake_minimum_required( VERSION 3.15 FATAL_ERROR )
include( FetchContent )
include( ExternalProject )
find_package( ecbuild REQUIRED )

# Specify project and configuration options
project( loki VERSION 0.0.5 LANGUAGES NONE )

# Declare options
ecbuild_add_option(
    FEATURE EDITABLE
    DEFAULT ON
    DESCRIPTION "Install Loki as an editable Python package"
)
ecbuild_add_option(
    FEATURE CLAW
    DEFAULT ON
    DESCRIPTION "Build OMNI Compiler and CLAW Compiler"
)

if( NOT ${HAVE_EDITABLE} )
    message( FATAL_ERROR "Loki currently supports only editable install" )
endif()

set( CLAW_DIR "" )
if( ${HAVE_CLAW} )

    # Bootstrap OpenJDK and Ant, if necessary
    FetchContent_Declare(
        cmake-jdk-ant
        GIT_REPOSITORY https://git.ecmwf.int/scm/~nabr/cmake-jdk-ant.git
    )
    FetchContent_MakeAvailable( cmake-jdk-ant )

    # Build OMNI Compiler and the CLAW
    FetchContent_Declare(
        claw_compiler
        GIT_REPOSITORY  https://github.com/reuterbal/claw-compiler.git
        GIT_TAG         mlange-dev
        GIT_SHALLOW     ON
    )

    # Need to fetch manually to be able to do an "in-build installation"
    FetchContent_GetProperties( claw_compiler )
    if( NOT claw_compiler_POPULATED )
        FetchContent_Populate( claw_compiler )

        set( CLAW_DIR ${CMAKE_CURRENT_BINARY_DIR}/claw-compiler )

    endif()

    ExternalProject_Add(
        claw
        SOURCE_DIR ${claw_compiler_SOURCE_DIR}
        BINARY_DIR ${claw_compiler_BINARY_DIR}
        INSTALL_DIR ${CLAW_DIR}

        # Can skip this as FetchContent will take care of it at configure time
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        # Specify in-build installation target and unset any CFLAGS
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CLAW_DIR} -DCFLAGS="" -DJAVA_HOME=${JAVA_HOME} -DANT_HOME=${ANT_HOME}
    )
endif()

# Discover only system install Python 3
set( Python3_FIND_VIRTUALENV STANDARD )
find_package( Python3 COMPONENTS Interpreter REQUIRED )

# Create a loki virtualenv
execute_process( COMMAND ${Python3_EXECUTABLE} -m venv "${CMAKE_CURRENT_BINARY_DIR}/loki_venv" )

# Update the environment with VIRTUAL_ENV variable (mimic the activate script)
set( ENV{VIRTUAL_ENV} "${CMAKE_CURRENT_BINARY_DIR}/loki_venv" )

# Change the context of the search to only find the venv
set( Python3_FIND_VIRTUALENV ONLY )

# Unset Python3_EXECUTABLE because it is also an input variable
#  (see documentation, Artifacts Specification section)
unset( Python3_EXECUTABLE )

# Launch a new search
find_package( Python3 COMPONENTS Interpreter Development REQUIRED )

# Find the binary directory of the virtual environment
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys; import os.path; print(os.path.dirname(sys.executable), end='')"
    OUTPUT_VARIABLE VIRTUAL_ENV_BIN
)

# Enable Pytest testing
if( ${ENABLE_TESTS} )

    if( ${HAVE_CLAW} )
        set( _TEST_SELECTOR "not ofp" )
    else()
        set( _TEST_SELECTOR "not ofp and not omni" )
    endif()

    ecbuild_add_test(
        TARGET loki_tests
        COMMAND ${Python3_EXECUTABLE}
        ARGS -m pytest -k ${_TEST_SELECTOR} -v tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENVIRONMENT PATH=${VIRTUAL_ENV_BIN}:${CLAW_DIR}/bin:$ENV{PATH}
    )

    ecbuild_add_test(
        TARGET loki_scripts_tests
        COMMAND ${Python3_EXECUTABLE}
        ARGS -m pytest -k ${_TEST_SELECTOR} -v scripts
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENVIRONMENT PATH=${VIRTUAL_ENV_BIN}:${CLAW_DIR}/bin:$ENV{PATH}
    )

    list( APPEND LOKI_INSTALL_OPTIONS "tests" )
endif()

# Install Loki with dependencies
if( NOT ${CMAKE_VERBOSE_MAKEFILE} )
    list ( APPEND PIP_OPTIONS "-q" )
endif()
if( ${HAVE_EDITABLE} )
    list( APPEND PIP_OPTIONS "-e" )
endif()

set( _INSTALL_OPTIONS "" )
if( LOKI_INSTALL_OPTIONS )
    list( JOIN LOKI_INSTALL_OPTIONS "," _INSTALL_OPT_STR )
    set( _INSTALL_OPTIONS "[${_INSTALL_OPT_STR}]" )
endif()

add_custom_target( loki ALL ${Python3_EXECUTABLE} -m pip install ${PIP_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR}${_INSTALL_OPTIONS} )

# Make CLI executables available in add_custom_command
add_executable( loki-lint.py IMPORTED GLOBAL )
set_property( TARGET loki-lint.py PROPERTY IMPORTED_LOCATION ${VIRTUAL_ENV_BIN}/loki-lint.py )
add_dependencies( loki-lint.py loki )

add_executable( loki-transform.py IMPORTED GLOBAL )
set_property( TARGET loki-transform.py PROPERTY IMPORTED_LOCATION ${VIRTUAL_ENV_BIN}/loki-transform.py )
add_dependencies( loki-transform.py loki )

if( ${HAVE_CLAW} )
    add_executable( clawfc IMPORTED GLOBAL )
    set_property( TARGET clawfc PROPERTY IMPORTED_LOCATION ${CLAW_DIR}/bin/clawfc )
    add_dependencies( clawfc claw )
    add_dependencies( loki-transform.py loki )
endif()

# Install the project so it can be used within the bundle
ecbuild_install_project( NAME loki )

# print summary
ecbuild_print_summary()
