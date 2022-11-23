# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

##############################################################################
# .rst:
#
# install_claw_compiler
# =====================
#
# Download and install CLAW Compiler and OMNI Compiler. ::
#
#   install_claw_compiler(VERSION)
#
# Installation procedure
# ----------------------
#
# CLAW and OMNI will be installed during the build step into folder
# `claw-compiler` in the current binary directory ``${CMAKE_CURRENT_BINARY_DIR}``.
#
# Options
# -------
#
# :VERSION:     The git branch or tag to download
#
# Output variables
# ----------------
# :CLAW_DIR:    The directory into which CLAW and OMNI have been installed
#
# Targets
# -------
# :clawfc:      The CLAW compiler CLI binary, usable, e.g. in ``add_custom_command``
#
##############################################################################

include( FetchContent )
include( ExternalProject )

function(install_claw_compiler VERSION)

    set( CLAW_DIR "" )
    message( STATUS "Downloading OMNI Compiler and CLAW Compiler")

    # Bootstrap OpenJDK and Ant, if necessary
    add_subdirectory( cmake/cmake-jdk-ant )

    # Build OMNI Compiler and the CLAW
    FetchContent_Declare(
        claw_compiler
        GIT_REPOSITORY  https://github.com/mlange05/claw-compiler.git
        GIT_TAG         ${VERSION}
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
        CMAKE_ARGS -DOMNI_CONF_OPTION=JAR=${Java_JAR_EXECUTABLE} -DCMAKE_INSTALL_PREFIX=${CLAW_DIR} -DJAVA_HOME=${JAVA_HOME} -DANT_HOME=${ANT_HOME}
    )

    add_executable( clawfc IMPORTED GLOBAL )
    set_property( TARGET clawfc PROPERTY IMPORTED_LOCATION ${CLAW_DIR}/bin/clawfc )
    add_dependencies( clawfc claw )

    # Forward variables to parent scope
    foreach ( _VAR_NAME CLAW_DIR )
        set( ${_VAR_NAME} ${${_VAR_NAME}} PARENT_SCOPE )
    endforeach()

endfunction()
