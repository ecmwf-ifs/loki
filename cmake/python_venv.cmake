##############################################################################
#.rst:
#
# python_venv
# ===========
#
# Find Python 3 and create a virtual environment. ::
#
#   python_venv(VENV_NAME)
#
# Installation procedure
# ----------------------
#
# It creates a virtual environment with the name provided in `VENV_NAME`
# in the current binary directory (``${CMAKE_CURRENT_BINARY_DIR}``).
#
# Options
# -------
#
# :VENV_NAME: The name of the virtual environment
#
# Output variables
# ----------------
# :Python3_FOUND:       Exported into parent scope from FindPython3
# :Python3_EXECUTABLE:  Exported into parent scope from FindPython3
# :Python3_VENV_BIN:    The path to the virtual environment's `bin` directory
# :ENV{VIRTUAL_ENV}:    Environment variable with the virtual environment directory,
#                       emulating the activate script
#
##############################################################################

function(python_venv VENV_NAME )

    # Discover only system install Python 3
    set( Python3_FIND_VIRTUALENV STANDARD )
    find_package( Python3 COMPONENTS Interpreter REQUIRED )

    # Create a loki virtualenv
    message( STATUS "Create Python virtual environment ${VENV_NAME}" )
    execute_process( COMMAND ${Python3_EXECUTABLE} -m venv "${CMAKE_CURRENT_BINARY_DIR}/${VENV_NAME}" )

    # Update the environment with VIRTUAL_ENV variable (mimic the activate script)
    set( ENV{VIRTUAL_ENV} "${CMAKE_CURRENT_BINARY_DIR}/${VENV_NAME}" )

    # Change the context of the search to only find the venv
    set( Python3_FIND_VIRTUALENV ONLY )

    # Unset Python3_EXECUTABLE because it is also an input variable
    #  (see documentation, Artifacts Specification section)
    unset( Python3_EXECUTABLE )

    # Launch a new search
    find_package( Python3 COMPONENTS Interpreter Development REQUIRED )

    # Find the binary directory of the virtual environment
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sys; import os.path; print(os.path.dirname(sys.executable), end='')"
        OUTPUT_VARIABLE Python3_VENV_BIN
    )

    # Forward variables to parent scope
    foreach ( _VAR_NAME Python3_FOUND Python3_EXECUTABLE Python3_VENV_BIN )
        set( ${_VAR_NAME} ${${_VAR_NAME}} PARENT_SCOPE )
    endforeach()

endfunction()
