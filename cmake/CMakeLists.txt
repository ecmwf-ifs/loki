file( GLOB_RECURSE loki_support_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*" )

ecbuild_add_resources(
    TARGET ${PROJECT_NAME}_loki_support_files
    SOURCES_PACK
        ${loki_support_files} )
