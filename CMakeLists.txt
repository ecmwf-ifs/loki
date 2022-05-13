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
    FEATURE NO_INSTALL
    DEFAULT OFF
    DESCRIPTION "Disable Loki (and dependency) installation"
)
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

# Make CMake script files available in build and install directory
add_subdirectory( cmake )
file( MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/share/loki )
execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_BINARY_DIR}/share/loki/cmake )
install( DIRECTORY cmake DESTINATION ${INSTALL_DATA_DIR} PATTERN "CMakeLists.txt" EXCLUDE )

if( NOT ${HAVE_EDITABLE} )
    message( FATAL_ERROR "Loki currently supports only editable install" )
endif()

if( HAVE_NO_INSTALL )

    # Find loki-transform.py and loki-lint.py on the PATH to make available as targets
    find_program ( _LOKI_LINT_EXECUTABLE NAMES loki-lint.py )
    add_executable( loki-lint.py IMPORTED GLOBAL )
    set_property( TARGET loki-lint.py PROPERTY IMPORTED_LOCATION ${_LOKI_LINT_EXECUTABLE} )

    find_program ( _LOKI_TRANSFORM_EXECUTABLE NAMES loki-transform.py )
    add_executable( loki-transform.py IMPORTED GLOBAL )
    set_property( TARGET loki-transform.py PROPERTY IMPORTED_LOCATION ${_LOKI_TRANSFORM_EXECUTABLE} )

else()

    if( ${HAVE_CLAW} )
        include( claw_compiler )
        install_claw_compiler( mlange-dev )
    endif()

    # Setup Python virtual environment
    include( python_venv )
    setup_python_venv( ${CMAKE_CURRENT_BINARY_DIR}/loki_env )
    install( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/loki_env DESTINATION . USE_SOURCE_PERMISSIONS )

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

    # We install Loki at configure time (for now), since bulk-transformation planning
    # requires configure time execution to allow injection with CMake targets.
    message( STATUS "Install Loki in virtual environment" )
    execute_process( COMMAND ${Python3_EXECUTABLE} -m pip install ${PIP_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR}${_INSTALL_OPTIONS} )
    message( STATUS "Install Loki in virtual environment - done" )

    # Make CLI executables available in add_custom_command
    add_executable( loki-lint.py IMPORTED GLOBAL )
    set_property( TARGET loki-lint.py PROPERTY IMPORTED_LOCATION ${Python3_VENV_BIN}/loki-lint.py )

    add_executable( loki-transform.py IMPORTED GLOBAL )
    set_property( TARGET loki-transform.py PROPERTY IMPORTED_LOCATION ${Python3_VENV_BIN}/loki-transform.py )

    if( HAVE_CLAW )
        add_dependencies( loki-transform.py clawfc )
    endif()

    update_python_shebang( ${Python3_VENV_BIN}/loki-lint.py ${Python3_VENV_BIN}/loki-transform.py )

endif()

if ( HAVE_NO_INSTALL OR NOT HAVE_CLAW )

    # Find clawfc on the PATH to make available as target, if it hasn't been installed by us
    find_program ( _CLAWFC_EXECUTABLE NAMES clawfc )
    add_executable( clawfc IMPORTED GLOBAL )
    set_property( TARGET clawfc PROPERTY IMPORTED_LOCATION ${_CLAWFC_EXECUTABLE} )

endif()

# Install the project so it can be used within the bundle
ecbuild_install_project( NAME loki )

# print summary
ecbuild_print_summary()
