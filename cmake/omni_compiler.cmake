# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

##############################################################################
# .rst:
#
# install_omni_compiler
# =====================
#
# Download and install OMNI Compiler. ::
#
#   install_omni_compiler(VERSION)
#
# Installation procedure
# ----------------------
#
# OMNI will be installed during the build step into folder
# `omni-compiler` in the current binary directory ``${CMAKE_CURRENT_BINARY_DIR}``.
#
# Options
# -------
#
# :VERSION:     The git branch or tag to download
#
# Output variables
# ----------------
# :OMNI_DIR:    The directory into which OMNI has been installed
#
##############################################################################

include( FetchContent )
include( ExternalProject )

function(install_omni_compiler VERSION)

    set( OMNI_DIR "" )
    message( STATUS "Downloading OMNI Compiler")

    # Bootstrap OpenJDK and Ant, if necessary
    add_subdirectory( cmake/cmake-jdk-ant )

    # Build OMNI Compiler
    FetchContent_Declare(
        omni_compiler
        GIT_REPOSITORY  https://github.com/omni-compiler/xcodeml-tools.git
        GIT_TAG         ${VERSION}
        GIT_SHALLOW     ON
    )

    # Need to fetch manually to be able to do an "in-build installation"
    FetchContent_GetProperties( omni_compiler )
    if( NOT omni_compiler_POPULATED )
        FetchContent_Populate( omni_compiler )

        set( OMNI_DIR ${CMAKE_CURRENT_BINARY_DIR}/omni-compiler )

    endif()

    find_program(MAKE_EXECUTABLE NAMES gmake make mingw32-make REQUIRED)

    ExternalProject_Add(
        omni
        SOURCE_DIR ${omni_compiler_SOURCE_DIR}
        BINARY_DIR ${omni_compiler_BINARY_DIR}
        INSTALL_DIR ${OMNI_DIR}

        # Can skip this as FetchContent will take care of it at configure time
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        # Specify in-build installation target
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OMNI_DIR} -DJAVA_HOME=${JAVA_HOME}
    )

    add_executable( F_Front IMPORTED GLOBAL )
    set_property( TARGET F_Front PROPERTY IMPORTED_LOCATION ${OMNI_DIR}/bin/F_Front )
    add_dependencies( F_Front omni )

    # Forward variables to parent scope
    foreach ( _VAR_NAME OMNI_DIR )
        set( ${_VAR_NAME} ${${_VAR_NAME}} PARENT_SCOPE )
    endforeach()

endfunction()
