#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
#

# This file has been adapted from the FindAnt.cmake module
# of CLAW compiler:
# https://github.com/claw-project/claw-compiler/blob/master/cmake/module/FindAnt.cmake


#  ANT_FOUND - system has Ant
#  Ant_EXECUTABLE - the Ant executable
#  Ant_VERSION - the Ant version
#
# It will search the environment variable ANT_HOME if it is set

include(FindPackageHandleStandardArgs)

set ( _ANT_HOME "" )
if ( ANT_HOME AND IS_DIRECTORY "${ANT_HOME}" )
    set ( _ANT_HOME "${ANT_HOME}" )
else()
    set ( _ENV_ANT_HOME "" )
    if ( DEFINED ENV{ANT_HOME} )
        file ( TO_CMAKE_PATH "$ENV{ANT_HOME}" _ENV_ANT_HOME )
    endif ()
    if ( _ENV_ANT_HOME AND IS_DIRECTORY "${_ENV_ANT_HOME}" )
        set ( _ANT_HOME "${_ENV_ANT_HOME}" )
    endif ()
    unset ( _ENV_ANT_HOME )
endif()

find_program(Ant_EXECUTABLE NAMES ant HINTS ${_ANT_HOME}/bin)

unset ( _ANT_HOME )

if(Ant_EXECUTABLE)

    # Try to determine Ant version
    execute_process(COMMAND ${Ant_EXECUTABLE} -version
        RESULT_VARIABLE res
        OUTPUT_VARIABLE var
        ERROR_VARIABLE var
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if( res )
        message( STATUS "Warning, could not run ant -version")
        unset(Ant_EXECUTABLE CACHE)
        unset(Ant_VERSION)
    else()
        # extract major/minor version and patch level from "ant -version" output
        if(var MATCHES "Apache Ant(.*)version ([0-9]+\\.[0-9]+\\.[0-9_.]+)(.*)")
            set(Ant_VERSION_STRING "${CMAKE_MATCH_2}")
        endif()
        string( REGEX REPLACE "([0-9]+).*" "\\1" Ant_VERSION_MAJOR "${Ant_VERSION_STRING}" )
        string( REGEX REPLACE "[0-9]+\\.([0-9]+).*" "\\1" Ant_VERSION_MINOR "${Ant_VERSION_STRING}" )
        string( REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" Ant_VERSION_PATCH "${Ant_VERSION_STRING}" )
        set(Ant_VERSION ${Ant_VERSION_MAJOR}.${Ant_VERSION_MINOR}.${Ant_VERSION_PATCH})
    endif()

endif()

find_package_handle_standard_args(Ant REQUIRED_VARS Ant_EXECUTABLE VERSION_VAR Ant_VERSION)
mark_as_advanced(Ant_EXECUTABLE)
