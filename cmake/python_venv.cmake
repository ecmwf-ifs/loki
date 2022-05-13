##############################################################################
#.rst:
#
# find_python_venv
# ================
#
# Find Python 3 inside a virtual environment. ::
#
#   find_python_venv(VENV_PATH)
#
# It finds the Python3 Interpreter from a virtual environment at
# the given location (`VENV_PATH`)
#
# Options
# -------
#
# :VENV_PATH: The path to the virtual environment
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

function( find_python_venv VENV_PATH )

    message( STATUS "VENV_PATH=${VENV_PATH}")

    # Update the environment with VIRTUAL_ENV variable (mimic the activate script)
    set( ENV{VIRTUAL_ENV} ${VENV_PATH} )

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

##############################################################################
#.rst:
#
# create_python_venv
# ==================
#
# Find Python 3 and create a virtual environment. ::
#
#   create_python_venv(VENV_PATH)
#
# Installation procedure
# ----------------------
#
# It creates a virtual environment at the given location (`VENV_PATH`)
#
# Options
# -------
#
# :VENV_PATH: The path to use for the virtual environment
#
##############################################################################

function( create_python_venv VENV_PATH )

    # Discover only system install Python 3
    set( Python3_FIND_VIRTUALENV STANDARD )
    find_package( Python3 COMPONENTS Interpreter REQUIRED )

    # Create a loki virtualenv
    message( STATUS "Create Python virtual environment ${VENV_PATH}" )
    execute_process( COMMAND ${Python3_EXECUTABLE} -m venv --copies "${VENV_PATH}" )

    # Make the virtualenv portable by automatically deducing the VIRTUAL_ENV path from
    # the 'activate' script's location in the filesystem
    execute_process(
        COMMAND
            sed -i "s/^VIRTUAL_ENV=\".*\"$/VIRTUAL_ENV=\"$(cd \"$(dirname \"$(dirname \"\${BASH_SOURCE[0]}\" )\")\" \\&\\& pwd)\"/" "${VENV_PATH}/bin/activate"
    )

endfunction()

##############################################################################
#.rst:
#
# setup_python_venv
# =================
#
# Find Python 3, create a virtual environment and make it available. ::
#
#   setup_python_venv(VENV_PATH)
#
# It combines calls to `create_python_venv` and `find_python_venv`
#
# Options
# -------
#
# :VENV_PATH: The path to use for the virtual environment
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

macro( setup_python_venv VENV_PATH )

    # Create the virtual environment
    create_python_venv( ${VENV_PATH} )

    # Discover Python in the virtual environment and set-up variables
    find_python_venv( ${VENV_PATH} )

endmacro()

##############################################################################
#.rst:
#
# update_python_shebang
# =====================
#
# Update the shebang in the given executable scripts to link them to a
# Python executable that is located in the same directory. ::
#
#   update_python_shebang( executable1 [executable2] [...] )
#
##############################################################################

function( update_python_shebang )

    foreach( _exe IN LISTS ARGV )

        # Replace the shebang in the executable script by the following to use the
        # Python binary that resides in the same directory as the script
        # (see https://stackoverflow.com/a/57567228).
        # That allows to move the script elsewhere along with the rest of the virtual
        # environment without breaking the link to the venv-interpreter
        #
        # #!/bin/sh
        # "true" '''\'
        # exec "$(dirname "$(readlink -f "$0")")"/python "$0" "$@"
        # '''

        execute_process(
            COMMAND
                sed -i "1s/^.*$/#\\!\\/bin\\/sh\\n\\\"true\\\" '''\\\\'\\nexec \\\"$(dirname \\\"$(readlink -f \\\"\\$0\\\")\\\")\\\"\\/python \\\"\\$0\\\" \\\"\\$@\\\"\\n'''/" ${_exe}
        )

    endforeach()

endfunction()
