# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

#
# Utility macro to translate single value and multi value arguments
# in loki_transform to command line arguments for loki-transform.py
#
macro( _loki_transform_parse_args )

    if( _PAR_MODE )
        list( APPEND _ARGS --mode ${_PAR_MODE} )
    endif()

    if( _PAR_CONFIG )
        list( APPEND _ARGS --config ${_PAR_CONFIG} )
    endif()

    if( _PAR_BUILDDIR )
        file( MAKE_DIRECTORY ${_PAR_BUILDDIR} )
        list( APPEND _ARGS --build ${_PAR_BUILDDIR} )
    endif()

    if( _PAR_DIRECTIVE )
        list( APPEND _ARGS --directive ${_PAR_DIRECTIVE} )
    endif()

    if( _PAR_FRONTEND )
        list( APPEND _ARGS --frontend ${_PAR_FRONTEND} )
    endif()

    if( _PAR_SOURCES )
        foreach( _SOURCE ${_PAR_SOURCES} )
            list( APPEND _ARGS --source ${_SOURCE} )
        endforeach()
    endif()

    if( _PAR_HEADERS )
        foreach( _HEADER ${_PAR_HEADERS} )
            list( APPEND _ARGS --header ${_HEADER} )
        endforeach()
    endif()

    if( _PAR_INCLUDES )
        foreach( _INCLUDE ${_PAR_INCLUDES} )
            list( APPEND _ARGS --include ${_INCLUDE} )
        endforeach()
    endif()

    if( _PAR_DEFINITIONS )
        foreach( _DEFINE ${_PAR_DEFINITIONS} )
            list( APPEND _ARGS --define ${_DEFINE} )
        endforeach()
    endif()

    if( _PAR_OMNI_INCLUDE )
        foreach( _OMNI_INCLUDE ${_PAR_OMNI_INCLUDE} )
            list( APPEND _ARGS --omni-include ${_OMNI_INCLUDE} )
        endforeach()
    endif()

    if( _PAR_XMOD )
        foreach( _XMOD ${_PAR_XMOD} )
            file( MAKE_DIRECTORY ${XMOD_DIR} )
            list( APPEND _ARGS --xmod ${_XMOD} )
        endforeach()
    endif()

endmacro()


##############################################################################

#
# Utility macro to translate options in loki_transform to command line
# arguments for loki-transform.py
#
macro( _loki_transform_parse_options )

    if( _PAR_CPP )
        list( APPEND _ARGS --cpp )
    endif()

    if( _PAR_DATA_OFFLOAD )
        list( APPEND _ARGS --data-offload )
    endif()

    if( _PAR_REMOVE_OPENMP )
        list( APPEND _ARGS --remove-openmp )
    endif()

    if( _PAR_ASSUME_DEVICEPTR )
        list( APPEND _ARGS --assume-deviceptr )
    endif()

    if( _PAR_GLOBAL_VAR_OFFLOAD )
        list( APPEND _ARGS --global-var-offload )
    endif()

    if( _PAR_TRIM_VECTOR_SECTIONS )
        list( APPEND _ARGS --trim-vector-sections )
    endif()

    if( _PAR_REMOVE_DERIVED_ARGS )
        list( APPEND _ARGS --remove-derived-args )
    endif()

    if( _PAR_INLINE_MEMBERS )
        list( APPEND _ARGS --inline-members )
    endif()

    if( _PAR_RESOLVE_SEQUENCE_ASSOCIATION )
        list( APPEND _ARGS --resolve-sequence-association )
    endif()

    if( _PAR_DERIVE_ARGUMENT_ARRAY_SHAPE )
        list( APPEND _ARGS --derive-argument-array-shape )
    endif()

endmacro()

##############################################################################

macro( _loki_transform_env_setup )

    # The full path of the loki-transform.py executable
    get_target_property( _LOKI_TRANSFORM_EXECUTABLE loki-transform.py IMPORTED_LOCATION )

    set( _LOKI_TRANSFORM_ENV )
    set( _LOKI_TRANSFORM_PATH )

    if( TARGET clawfc AND ${_PAR_FRONTEND} STREQUAL "omni" )
        # Ugly hack but I don't have a better solution: We need to add F_FRONT
        # (which is installed in the same directory as clawfc) to the PATH, if
        # OMNI is used as a frontend. Hence we have to update the environment in the below
        # add_custom_command calls to loki-transform.py.
        get_target_property( _CLAWFC_EXECUTABLE clawfc IMPORTED_LOCATION )
        get_filename_component( _CLAWFC_LOCATION ${_CLAWFC_EXECUTABLE} DIRECTORY )
        list( APPEND _LOKI_TRANSFORM_PATH ${_CLAWFC_LOCATION} )
    endif()

    if( _PAR_OUTPATH AND (${_PAR_FRONTEND} STREQUAL "omni" OR ${_PAR_FRONTEND} STREQUAL "ofp") )
        # With pre-processing, we may end up having a race condition on the preprocessed
        # source files in parallel builds. Ensuring we use the outpath of the call to Loki
        # should ensure in most cases that parallel builds write to different directories
        # Note: this does not affect Fparser as we don't have to write preprocessed files
        # to disk there
        list( APPEND _LOKI_TRANSFORM_ENV LOKI_TMP_DIR=${_PAR_OUTPATH} )
    endif()

    if( _LOKI_TRANSFORM_ENV OR _LOKI_TRANSFORM_PATH )
        if( TARGET loki-transform.py )
            # Unfortunately, an environment update breaks the CMake feature of recognizing
            # the executable in add_custom_command as a previously declared target, which would
            # enable choosing the correct path automatically. Therefore, we have to insert also
            # loki-transform.py into the PATH variable.
            get_filename_component( _LOKI_TRANSFORM_LOCATION ${_LOKI_TRANSFORM_EXECUTABLE} DIRECTORY )
            list( APPEND _LOKI_TRANSFORM_PATH ${_LOKI_TRANSFORM_LOCATION} )
        endif()

        # Join all declared paths
        string( REPLACE ";" ":" _LOKI_TRANSFORM_PATH "${_LOKI_TRANSFORM_PATH}" )
        list( APPEND _LOKI_TRANSFORM_ENV PATH=${_LOKI_TRANSFORM_PATH}:$ENV{PATH} )

        # Run loki-transform.py via the CMake ENV wrapper
        set( _LOKI_TRANSFORM ${CMAKE_COMMAND} -E env ${_LOKI_TRANSFORM_ENV} loki-transform.py )

        # Also, now it breaks the dependency chain and we have to declare manual dependencies on
        # loki-transform.py...
        set( _LOKI_TRANSFORM_DEPENDENCY loki-transform.py )
    else()
        # This is how it is meant to be: We can rely on CMake's ability to set the correct
        # path of loki-transform.py if it was declared as an executable before (otherwise it
        # will assume it has been already on the path when CMake was called
        set( _LOKI_TRANSFORM loki-transform.py )
        set( _LOKI_TRANSFORM_DEPENDENCY "" )
    endif()

endmacro()

##############################################################################
# .rst:
#
# loki_copy_compile_flags
# =======================
#
# Copy compile flags from a list of source files to a list of source files.::
#
#
#   loki_copy_compile_flags( ORIG_LIST NEW_LIST )
#
# ``ORIG_LIST`` and ``NEW_LIST`` must have the same length. Compile flags are
# copied per-entry, this means matching indices between ``ORIG_LIST`` and
# ``NEW_LIST`` is assumed.
#
##############################################################################
function( loki_copy_compile_flags )

    set( options "" )
    set( single_value_args "" )
    set( multi_value_args ORIG_LIST NEW_LIST )

    cmake_parse_arguments( _PAR "${options}" "${single_value_args}" "${multi_value_args}" ${ARGN} )

    # Copy over compile flags for generated source. Note that this assumes
    # matching indexes between ORIG_LIST and NEW_LIST to encode the source-to-source mapping.
    list( LENGTH _PAR_ORIG_LIST nsources )
    math( EXPR maxidx "${nsources} - 1" )
    if ( nsources GREATER 0 )
        foreach( idx RANGE 0 ${maxidx} )
            list( GET _PAR_ORIG_LIST ${idx} orig )
            list( GET _PAR_NEW_LIST ${idx} newsrc )

            ecbuild_debug( "[Loki] loki_copy_compile_flags: ${orig} -> ${newsrc}" )

            foreach( _prop COMPILE_FLAGS
                     COMPILE_FLAGS_${CMAKE_BUILD_TYPE_CAPS}
                     OVERRIDE_COMPILE_FLAGS
                     OVERRIDE_COMPILE_FLAGS_${CMAKE_BUILD_TYPE_CAPS} )

                get_source_file_property( ${orig}_${_prop} ${orig} ${_prop} )
                if( ${orig}_${_prop} )
                    set_source_files_properties( ${newsrc} PROPERTIES ${_prop} ${${orig}_${_prop}} )
                endif()
            endforeach()
        endforeach()
    endif()

endfunction()

##############################################################################
