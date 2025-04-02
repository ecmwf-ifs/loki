# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

include( loki_transform_helpers )

##############################################################################
# .rst:
#
# loki_transform
# ==============
#
# Invoke loki-transform.py using the given options.::
#
#   loki_transform(
#       COMMAND <convert|...>
#       OUTPUT <outfile1> [<outfile2> ...]
#       DEPENDS <dependency1> [<dependency2> ...]
#       MODE <mode>
#       CONFIG <config-file>
#       [CPP]
#       [FRONTEND <frontend>]
#       [BUILDDIR <build-path>]
#       [SOURCES <source1> [<source2> ...]]
#       [HEADERS <header1> [<header2> ...]]
#   )
#
# Call ``loki-transform.py <convert|...> ...`` with the provided arguments.
# See ``loki-transform.py`` for a description of all options.
#
# Options
# -------
#
# :OUTPUT:     The output files generated by Loki. Providing them here allows
#              to declare dependencies on this command later.
# :DEPENDS:    The input files or targets this call depends on.
#
##############################################################################

function( loki_transform )

    set( options CPP )
    set( oneValueArgs COMMAND MODE FRONTEND CONFIG BUILDDIR )
    set( multiValueArgs OUTPUT DEPENDS SOURCES HEADERS INCLUDES DEFINITIONS OMNI_INCLUDE XMOD )

    cmake_parse_arguments( _PAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if( _PAR_UNPARSED_ARGUMENTS )
        ecbuild_critical( "Unknown keywords given to loki_transform(): \"${_PAR_UNPARSED_ARGUMENTS}\"")
    endif()

    # Select command for loki-transform.py
    if( NOT _PAR_COMMAND )
        ecbuild_critical( "No COMMAND specified for loki_transform()" )
    endif()
    set( _ARGS ${_PAR_COMMAND} )

    if( NOT _PAR_OUTPUT )
        ecbuild_critical( "No OUTPUT specified for loki_transform()" )
    endif()

    if( NOT _PAR_DEPENDS )
        ecbuild_critical( "No DEPENDS specified for loki_transform()" )
    endif()

    # Translate function args to arguments for loki-transform.py
    _loki_transform_parse_args()

    if( _PAR_CPP )
        list( APPEND _ARGS --cpp )
    endif()

    # Ensure transformation script and environment is available
    _loki_transform_env_setup()

    ecbuild_debug( "COMMAND ${_LOKI_TRANSFORM} ${_ARGS}" )

    add_custom_command(
        OUTPUT ${_PAR_OUTPUT}
        COMMAND ${_LOKI_TRANSFORM} ${_ARGS}
        DEPENDS ${_PAR_DEPENDS} ${_LOKI_TRANSFORM_DEPENDENCY}
        COMMENT "[Loki] Pre-processing: command=${_PAR_COMMAND} mode=${_PAR_MODE} frontend=${_PAR_FRONTEND}"
    )

endfunction()

##############################################################################
# .rst:
#
# loki_transform_plan
# ===================
#
# Run Loki bulk transformation in plan mode.::
#
#   loki_transform_plan(
#       MODE <mode>
#       FRONTEND <frontend>
#       [CPP]
#       [CONFIG <config-file>]
#       [BUILDDIR <build-path>]
#       [NO_SOURCEDIR | SOURCEDIR <source-path>]
#       [CALLGRAPH <callgraph-path>]
#       [PLAN <plan-file>]
#       [SOURCES <source1> [<source2> ...]]
#       [HEADERS <header1> [<header2> ...]]
#   )
#
# Call ``loki-transform.py plan ...`` with the provided arguments.
# See ``loki-transform.py`` for a description of all options.
#
##############################################################################

function( loki_transform_plan )

    set( options NO_SOURCEDIR CPP )
    set( oneValueArgs MODE FRONTEND CONFIG BUILDDIR SOURCEDIR CALLGRAPH PLAN )
    set( multiValueArgs SOURCES HEADERS )

    cmake_parse_arguments( _PAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if( _PAR_UNPARSED_ARGUMENTS )
        ecbuild_critical( "Unknown keywords given to loki_transform_plan(): \"${_PAR_UNPARSED_ARGUMENTS}\"")
    endif()

    set( _ARGS )

    # Translate function args to arguments for loki-transform.py
    _loki_transform_parse_args()

    if( _PAR_CPP )
        list( APPEND _ARGS --cpp )
    endif()

    if( NOT _PAR_NO_SOURCEDIR )
        if( _PAR_SOURCEDIR )
            list( APPEND _ARGS --root ${_PAR_SOURCEDIR} )
        else()
            ecbuild_critical( "No SOURCEDIR specified for loki_transform_plan()" )
        endif()
    endif()

    if( _PAR_CALLGRAPH )
        list( APPEND _ARGS --callgraph ${_PAR_CALLGRAPH} )
    endif()

    if( _PAR_PLAN )
        list( APPEND _ARGS --plan-file ${_PAR_PLAN} )
    else()
        ecbuild_critical( "No PLAN file specified for loki_transform_plan()" )
    endif()

    _loki_transform_env_setup()

    # Create a source transformation plan to tell CMake which files will be affected
    ecbuild_info( "[Loki] Creating plan: mode=${_PAR_MODE} frontend=${_PAR_FRONTEND} config=${_PAR_CONFIG}" )
    ecbuild_debug( "COMMAND ${_LOKI_TRANSFORM_EXECUTABLE} plan ${_ARGS}" )

    execute_process(
        COMMAND ${_LOKI_TRANSFORM_EXECUTABLE} plan ${_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_ERROR_IS_FATAL ANY
        ECHO_ERROR_VARIABLE
    )

endfunction()

##############################################################################
# .rst:
#
# loki_transform_target
# ======================
#
# Apply Loki source transformations to sources in a CMake target.::
#
#   loki_transform_target(
#       TARGET <target>
#       [COMMAND <convert|...>]
#       MODE <mode>
#       CONFIG <config-file>
#       PLAN <plan-file>
#       [CPP] [CPP_PLAN]
#       [FRONTEND <frontend>]
#       [SOURCES <source1> [<source2> ...]]
#       [HEADERS <header1> [<header2> ...]]
#       [NO_PLAN_SOURCEDIR COPY_UNMODIFIED]
#   )
#
# Applies a Loki bulk transformation to the source files belonging to a particular
# CMake target according to the specified entry points in the ``config-file``.
#
# This is done via a call to ``loki-transform.py plan ...`` during configure,
# from which the specific additions and deletions of source objects within the
# target are derived. See ``loki_transform_plan`` for more details.
#
# Subsequently, the actual bulk transformation of source files is scheduled
# via ``loki-transform.py <command>``, where ``<command>`` is provided via ``COMMAND``.
# If none is given, this defaults to ``convert``.
#
# Preprocessing of source files during plan or transformation stage can be
# enabled using ``CPP_PLAN`` and ``CPP`` options, respectively.
#
# ``NO_PLAN_SOURCEDIR`` can optionally be specified to call the plan stage without
# an explicit root directory. That means, Loki will generate absolute paths in the
# CMake plan file. This requires the ``SOURCES`` of the target to transform also
# to be given with absolute paths, otherwise the file list update mechanism will not
# work as expected.
#
# See ``loki-transform.py`` for a description of all options.
#
##############################################################################

function( loki_transform_target )

    set( options NO_PLAN_SOURCEDIR COPY_UNMODIFIED CPP CPP_PLAN )
    set( single_value_args COMMAND MODE FRONTEND CONFIG PLAN )
    set( multi_value_args TARGET SOURCES HEADERS DEFINITIONS INCLUDES )

    cmake_parse_arguments( _PAR_T "${options}" "${single_value_args}" "${multi_value_args}" ${ARGN} )

    if( _PAR_UNPARSED_ARGUMENTS )
        ecbuild_critical( "Unknown keywords given to loki_transform_target(): \"${_PAR_UNPARSED_ARGUMENTS}\"")
    endif()

    if( NOT _PAR_T_TARGET )
        ecbuild_critical( "The call to loki_transform_target() doesn't specify the TARGET(S)." )
    endif()

    if( NOT _PAR_T_COMMAND )
        set( _PAR_T_COMMAND "convert" )
    endif()

    if( NOT _PAR_T_PLAN )
        ecbuild_critical( "No PLAN(S) specified for loki_transform_target()" )
    endif()

    ecbuild_info( "[Loki] Loki scheduler:: target(s)=${_PAR_T_TARGET} mode=${_PAR_T_MODE} frontend=${_PAR_T_FRONTEND}")

    # Ensure that changes to the config file trigger the planning stage
    foreach( target ${_PAR_T_TARGET} )
        configure_file( ${_PAR_T_CONFIG} ${CMAKE_CURRENT_BINARY_DIR}/loki_${target}.config )
    endforeach()

    # Create the bulk-transformation plan
    set( _PLAN_OPTIONS "" )
    if( _PAR_T_CPP_PLAN )
        list( APPEND _PLAN_OPTIONS CPP )
    endif()
    if( _PAR_T_NO_PLAN_SOURCEDIR )
        list( APPEND _PLAN_OPTIONS NO_SOURCEDIR )
    endif()

    string(REPLACE ";" "_" CALLGRAPH_NAME "${_PAR_T_TARGET}")

    loki_transform_plan(
        MODE      ${_PAR_T_MODE}
        CONFIG    ${_PAR_T_CONFIG}
        FRONTEND  ${_PAR_T_FRONTEND}
        SOURCES   ${_PAR_T_SOURCES}
        PLAN      ${_PAR_T_PLAN}
        CALLGRAPH ${CMAKE_CURRENT_BINARY_DIR}/callgraph_${CALLGRAPH_NAME}
        BUILDDIR  ${CMAKE_CURRENT_BINARY_DIR}
        SOURCEDIR ${CMAKE_CURRENT_SOURCE_DIR}
        ${_PLAN_OPTIONS}
    )

    include("${_PAR_T_PLAN}")
    ecbuild_info( "[Loki] Imported transformation plan: ${_PAR_T_PLAN}" )
    ecbuild_debug( "[Loki] Loki all transform: ${LOKI_SOURCES_TO_TRANSFORM}" )
    ecbuild_debug( "[Loki] Loki all append: ${LOKI_SOURCES_TO_APPEND}" )
    ecbuild_debug( "[Loki] Loki all remove: ${LOKI_SOURCES_TO_REMOVE}" )

    # Schedule the source-to-source transformation on the source files from the schedule
    list( LENGTH LOKI_SOURCES_TO_TRANSFORM LOKI_APPEND_LENGTH )
    if ( LOKI_APPEND_LENGTH GREATER 0 )

        # Apply the bulk-transformation according to the plan
        set( _TRANSFORM_OPTIONS "" )
        if( _PAR_T_CPP )
            list( APPEND _TRANSFORM_OPTIONS CPP )
        endif()

        loki_transform(
            COMMAND     ${_PAR_T_COMMAND}
            OUTPUT      ${LOKI_SOURCES_TO_APPEND}
            MODE        ${_PAR_T_MODE}
            CONFIG      ${_PAR_T_CONFIG}
            FRONTEND    ${_PAR_T_FRONTEND}
            BUILDDIR    ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES     ${_PAR_T_SOURCES}
            HEADERS     ${_PAR_T_HEADERS}
            DEFINITIONS ${_PAR_T_DEFINITIONS}
            INCLUDES    ${_PAR_T_INCLUDES}
            DEPENDS     ${LOKI_SOURCES_TO_TRANSFORM} ${_PAR_T_HEADERS} ${_PAR_T_CONFIG}
            ${_TRANSFORM_OPTIONS}
        )
    endif()

    set(_TARGETS_POSTFIX "")
    list( LENGTH _PAR_T_TARGET TARGETS_LENGTH )
    if (TARGETS_LENGTH GREATER 1)
        foreach(_target ${_PAR_T_TARGET})
	    # sanitize target name, e.g., replace '.' with '-' (same things happen within Loki)
            string(REPLACE "." "_" _sanitized_target ${_target})
            list( APPEND _TARGETS_POSTFIX "_${_sanitized_target}")
        endforeach()
    endif()


    foreach(_target _postfix IN ZIP_LISTS _PAR_T_TARGET _TARGETS_POSTFIX)
        ecbuild_debug( "[Loki] Loki ${_target} transform: ${LOKI_SOURCES_TO_TRANSFORM${_postfix}}")
        ecbuild_debug( "[Loki] Loki ${_target} append: ${LOKI_SOURCES_TO_APPEND${_postfix}}" )
        ecbuild_debug( "[Loki] Loki ${_target} remove: ${LOKI_SOURCES_TO_REMOVE${_postfix}}" )

        # Exclude source files that Loki has re-generated.
        # Note, this is done explicitly here because the HEADER_FILE_ONLY
        # property is not always being honoured by CMake.
        get_target_property( _target_sources ${_target} SOURCES )
        foreach( source ${LOKI_SOURCES_TO_REMOVE${_postfix}} )
            # get_property( source_deps SOURCE ${source} PROPERTY OBJECT_DEPENDS )
            list( FILTER _target_sources EXCLUDE REGEX ${source} )
        endforeach()

        if( NOT _PAR_T_COPY_UNMODIFIED )
            # Update the target source list
            set_property( TARGET ${_target} PROPERTY SOURCES ${_target_sources} )
        else()
            # Copy the unmodified source files to the build dir
            set( _target_sources_copy "" )
            foreach( source ${_target_sources} )
                get_filename_component( _source_name ${source} NAME )
                list( APPEND _target_sources_copy ${CMAKE_CURRENT_BINARY_DIR}/${_source_name} )
                ecbuild_debug( "[Loki] copy: ${source} -> ${CMAKE_CURRENT_BINARY_DIR}/${_source_name}" )
            endforeach()
            file( COPY ${_target_sources} DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

            # Mark the copied files as build-time generated
            set_source_files_properties( ${_target_sources_copy} PROPERTIES GENERATED TRUE )

            # Update the target source list
            set_property( TARGET ${_target} PROPERTY SOURCES ${_target_sources_copy} )
        endif()

        if ( LOKI_APPEND_LENGTH GREATER 0 )
            # Mark the generated stuff as build-time generated
            set_source_files_properties( ${LOKI_SOURCES_TO_APPEND${_postfix}} PROPERTIES GENERATED TRUE )

            # Add the Loki-generated sources to our target
            target_sources( ${_target} PRIVATE ${LOKI_SOURCES_TO_APPEND${_postfix}} )
        endif()

        # Copy over compile flags for generated source. Note that this assumes
        # matching indexes between LOKI_SOURCES_TO_TRANSFORM and LOKI_SOURCES_TO_APPEND
        # to encode the source-to-source mapping. This matching is strictly enforced
        # in the `CMakePlannerTransformation`.
        loki_copy_compile_flags(
            ORIG_LIST ${LOKI_SOURCES_TO_TRANSFORM${_postfix}}
            NEW_LIST ${LOKI_SOURCES_TO_APPEND${_postfix}}
        )

        if( _PAR_T_COPY_UNMODIFIED )
            loki_copy_compile_flags(
                ORIG_LIST ${_target_sources}
                NEW_LIST ${_target_sources_copy}
            )
        endif()
    endforeach()
endfunction()


##############################################################################
# .rst:
#
# generate_xmod
# =============
#
# Call OMNI's F_Front on a file to generate its xml-parse tree and, as a
# side effect, xmod-file.::
#
#   generate_xmod(
#       OUTPUT <xml-file>
#       SOURCE <source>
#       [XMOD <xmod-dir1> [<xmod-dir2> ...]]
#       [DEPENDS <dependency1> [<dependency2> ...]]
#   )
#
# Note that the xmod-file will be located in the first path given to ``XMOD``.
#
##############################################################################
function( generate_xmod )

    set( options )
    set( oneValueArgs SOURCE OUTPUT )
    set( multiValueArgs XMOD DEPENDS )

    cmake_parse_arguments( _PAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if( NOT _PAR_OUTPUT )
        ecbuild_critical( "No OUTPUT given for generate_xmod()" )
    endif()

    if( NOT _PAR_SOURCE )
        ecbuild_critical( "No SOURCE given for generate_xmod()" )
    endif()

    set( _ARGS )
    list( APPEND _ARGS -fleave-comment )

    if( _PAR_XMOD )
        foreach( XMOD ${_PAR_XMOD} )
            list( APPEND _ARGS -M ${XMOD} )
        endforeach()
    endif()

    set( _F_FRONT_EXECUTABLE F_Front )

    add_custom_command(
        OUTPUT ${_PAR_OUTPUT}
        COMMAND ${_F_FRONT_EXECUTABLE} ${_ARGS} -o ${_PAR_OUTPUT} ${_PAR_SOURCE}
        DEPENDS ${_PAR_SOURCE} ${_PAR_DEPENDS}
        COMMENT "[OMNI] Pre-processing: ${_PAR_SOURCE}"
    )

endfunction()
