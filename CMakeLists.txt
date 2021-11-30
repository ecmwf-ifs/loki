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

include( python_venv )
include( claw_compiler )

if( NOT ${HAVE_EDITABLE} )
    message( FATAL_ERROR "Loki currently supports only editable install" )
endif()

if( ${HAVE_CLAW} )
    install_claw_compiler( mlange-dev )
endif()

# Setup Python virtual environment
python_venv( loki_env )

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
        ENVIRONMENT PATH=${Python3_VENV_BIN}:${CLAW_DIR}/bin:$ENV{PATH}
    )

    ecbuild_add_test(
        TARGET loki_scripts
        COMMAND ${Python3_EXECUTABLE}
        ARGS -m pytest -k ${_TEST_SELECTOR} -v scripts
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENVIRONMENT PATH=${Python3_VENV_BIN}:${CLAW_DIR}/bin:$ENV{PATH}
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
set_property( TARGET loki-lint.py PROPERTY IMPORTED_LOCATION ${Python3_VENV_BIN}/loki-lint.py )
add_dependencies( loki-lint.py loki )

add_executable( loki-transform.py IMPORTED GLOBAL )
set_property( TARGET loki-transform.py PROPERTY IMPORTED_LOCATION ${Python3_VENV_BIN}/loki-transform.py )
add_dependencies( loki-transform.py loki )

if( ${HAVE_CLAW} )
    add_dependencies( loki-transform.py clawfc )
endif()

include( loki_transform )

# Install the project so it can be used within the bundle
ecbuild_install_project( NAME loki )

# print summary
ecbuild_print_summary()
