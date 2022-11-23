# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# CMake 3.10 required to parse Java version
# CMake 3.14 required to use FetchContent_MakeAvailable
cmake_minimum_required( VERSION 3.14 )
project( cmake-jdk-ant VERSION 0.1 LANGUAGES NONE )

set( MINIMUM_JAVA_VERSION 11 CACHE STRING "Minimum Java version required" )
set( MINIMUM_ANT_VERSION 1.10 CACHE STRING "Minimum ant version required" )

set( FORCE_OPEN_JDK_INSTALL OFF CACHE BOOL "Force installation of OpenJDK" )
set( FORCE_ANT_INSTALL OFF CACHE BOOL "Force installation of OpenJDK" )

set( OPEN_JDK_INSTALL_VERSION 11.0.2 CACHE STRING "OpenJDK version to install if Java >= ${MINIMUM_JAVA_VERSION} not found" )
set( OPEN_JDK_MIRROR https://download.java.net/java/GA/jdk11/9/GPL/ CACHE STRING "OpenJDK download mirror" )

set( ANT_INSTALL_VERSION 1.10.12 CACHE STRING "ant version to install if Ant >= ${MINIMUM_ANT_VERSION} not found" )
set( ANT_MIRROR https://archive.apache.org/dist/ant/binaries/ CACHE STRING "ant download mirror" )

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/module" )

include( FetchContent )

find_package( Java ${MINIMUM_JAVA_VERSION} COMPONENTS Development )

if( NOT Java_FOUND OR FORCE_OPEN_JDK_INSTALL )

    # Remove variables from cache to make sure find_package(Java) works correctly later
    unset( JAVA_HOME CACHE )
    unset( Java_JAVA_EXECUTABLE CACHE )

    # Use FetchContent to make available at configure time instead of build time (as would be the case with ExternalProject_Add)
    FetchContent_Declare(
        OpenJDK
        URL ${OPEN_JDK_MIRROR}/openjdk-${OPEN_JDK_INSTALL_VERSION}_linux-x64_bin.tar.gz
    )

    list( APPEND FETCH_CONTENT_LIST "OpenJDK" )
    message( STATUS "OpenJDK: Download and install version ${OPEN_JDK_INSTALL_VERSION}" )

elseif( DEFINED ENV{JAVA_HOME} )

    set( JAVA_HOME $ENV{JAVA_HOME} CACHE STRING "" )

endif()

find_package( Ant ${MINIMUM_ANT_VERSION} )

if( NOT Ant_FOUND OR FORCE_ANT_INSTALL )

    # Remove variables from cache to make sure find_package(Ant) works correctly later
    unset( ANT_HOME CACHE )
    unset( Ant_EXECUTABLE CACHE )

    # Use FetchContent to make available at configure time instead of build time (as would be the case with ExternalProject_Add)
    FetchContent_Declare(
        Ant
        URL ${ANT_MIRROR}/apache-ant-${ANT_INSTALL_VERSION}-bin.tar.gz
    )

    list( APPEND FETCH_CONTENT_LIST "Ant" )
    message( STATUS "Ant: Download and install version ${ANT_INSTALL_VERSION}" )

endif()

if( FETCH_CONTENT_LIST )

    # Trigger the actual downloads
    FetchContent_MakeAvailable ( ${FETCH_CONTENT_LIST} )

    # Re-discover Java
    if( "OpenJDK" IN_LIST FETCH_CONTENT_LIST )
        FetchContent_GetProperties( OpenJDK SOURCE_DIR OPEN_JDK_SOURCE_DIR BINARY_DIR OPEN_JDK_BINARY_DIR )

        # Create wrapper scripts that set JAVA_HOME for Java binaries
        file( MAKE_DIRECTORY "${OPEN_JDK_BINARY_DIR}/bin" )
        foreach( _JAVA_BINARY java javac javah jar javadoc )
            file(
                WRITE "${OPEN_JDK_SOURCE_DIR}/${_JAVA_BINARY}"
                "#!/bin/bash
                JAVA_HOME=${OPEN_JDK_SOURCE_DIR} ${OPEN_JDK_SOURCE_DIR}/bin/${_JAVA_BINARY} \"$@\""
            )
            file(
                COPY "${OPEN_JDK_SOURCE_DIR}/${_JAVA_BINARY}"
                DESTINATION "${OPEN_JDK_BINARY_DIR}/bin"
                FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            )
        endforeach()

        # Re-discover Java
        set( JAVA_HOME ${OPEN_JDK_BINARY_DIR} CACHE STRING "" )
        find_package( Java ${OPEN_JDK_INSTALL_VERSION} EXACT REQUIRED COMPONENTS Development )
    endif()

    # Re-discover Ant and fetch dependencies
    if( "Ant" IN_LIST FETCH_CONTENT_LIST )
        FetchContent_GetProperties( Ant SOURCE_DIR ANT_SOURCE_DIR BINARY_DIR ANT_BINARY_DIR )

        if( JAVA_HOME )
            set( _JAVA_HOME "JAVA_HOME=${JAVA_HOME}" )
        elseif( DEFINED ENV{JAVA_HOME} )
            set( _JAVA_HOME "JAVA_HOME=$ENV{JAVA_HOME}" )
        else()
            set( _JAVA_HOME "" )
        endif()

        # Create a wrapper script that sets ANT_HOME
        file(
            WRITE "${ANT_SOURCE_DIR}/ant"
            "#!/bin/bash
            ANT_HOME=${ANT_SOURCE_DIR} ${_JAVA_HOME} ${ANT_SOURCE_DIR}/bin/ant \"$@\""
        )
        file( MAKE_DIRECTORY "${ANT_BINARY_DIR}/bin" )
        file(
            COPY "${ANT_SOURCE_DIR}/ant"
            DESTINATION "${ANT_BINARY_DIR}/bin"
            FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        )

        # Download dependencies
        message( STATUS "Ant: Fetch dependencies" )
        execute_process( COMMAND "${ANT_BINARY_DIR}/bin/ant" -f "${ANT_SOURCE_DIR}/fetch.xml" -Ddest=optional OUTPUT_QUIET )

        # Re-discover ant
        set( ANT_HOME ${ANT_BINARY_DIR} CACHE STRING "" )
        find_package( Ant ${ANT_INSTALL_VERSION} EXACT REQUIRED )
    endif()

endif()

message ( VERBOSE "JAVA_HOME=\"${JAVA_HOME}\"" )
message ( VERBOSE "ANT_HOME=\"${ANT_HOME}\"" )
