# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
cmake_minimum_required( VERSION 3.19 FATAL_ERROR )

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}" )
include( loki_python_macros )

set( WHEELS_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheels" CACHE PATH "" )
set( REQUIREMENT_SPEC "${CMAKE_CURRENT_LIST_DIR}/.." CACHE STRING "" )
set( LOKI_WHEEL_ARCH NONE CACHE STRING "" )
set( LOKI_WHEEL_PYTHON_VERSION CACHE STRING "" )

loki_download_python_wheels(
    REQUIREMENT_SPEC        ${REQUIREMENT_SPEC}
    WHEELS_DIR              ${WHEELS_DIR}
    WHEEL_ARCH              ${LOKI_WHEEL_ARCH}
    WHEEL_PYTHON_VERSION    ${LOKI_WHEEL_PYTHON_VERSION}
)
