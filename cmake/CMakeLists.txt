# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

file( GLOB_RECURSE loki_support_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*" )

ecbuild_add_resources(
    TARGET ${PROJECT_NAME}_loki_support_files
    SOURCES_PACK
        ${loki_support_files}
)
