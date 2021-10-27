# Version 3.12 required to use FindPython
# Version 3.15 required to use Python3_FIND_VIRTUALENV (use 3.17 for conda support)
cmake_minimum_required ( VERSION 3.15 FATAL_ERROR )

# Specify project and configuration options
project ( loki )

option ( EDITABLE "Install Loki as an editable Python package" ON )

# Discover only system install Python 3
set ( Python3_FIND_VIRTUALENV STANDARD )
find_package ( Python3 COMPONENTS Interpreter REQUIRED )

# Create a loki virtualenv
execute_process ( COMMAND ${Python3_EXECUTABLE} -m venv "${CMAKE_CURRENT_BINARY_DIR}/loki_venv" )

# Update the environment with VIRTUAL_ENV variable (mimic the activate script)
set ( ENV{VIRTUAL_ENV} "${CMAKE_CURRENT_BINARY_DIR}/loki_venv" )

# Change the context of the search to only find the venv
set ( Python3_FIND_VIRTUALENV ONLY )

# Unset Python3_EXECUTABLE because it is also an input variable
#  (see documentation, Artifacts Specification section)
unset ( Python3_EXECUTABLE )

# Launch a new search
find_package ( Python3 COMPONENTS Interpreter Development REQUIRED )

# Find the binary directory of the virtual environment
execute_process (
    COMMAND ${Python3_EXECUTABLE} -c "import sys; import os.path; print(os.path.dirname(sys.executable), end='')"
    OUTPUT_VARIABLE VIRTUAL_ENV_BIN
)

# Install Loki with dependencies
if ( EDITABLE )
    list(APPEND PIP_OPTIONS "-e")
endif()
add_custom_command (
    OUTPUT ${VIRTUAL_ENV_BIN}/loki-transform.py ${VIRTUAL_ENV_BIN}/loki-lint.py
    COMMAND ${Python3_EXECUTABLE} -m pip install ${PIP_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create targets for Loki CLI executables
add_custom_target ( loki-transform.py ALL DEPENDS ${VIRTUAL_ENV_BIN}/loki-transform.py )
add_custom_target ( loki-lint.py ALL DEPENDS ${VIRTUAL_ENV_BIN}/loki-lint.py )
