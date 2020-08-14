#!/usr/bin/env bash
source /usr/local/apps/module/init/bash

# Parameters 
LOKI_HOME=$(git rev-parse --show-toplevel)
VENV_HOME=$(realpath ${VIRTUAL_ENV:-${1:-$LOKI_HOME/loki_env}})
USE_ANT_VERSION=1.10.8
USE_JAVA_VERSION=11.0.1
USE_PYTHON_VERSION=3.7.1-01
USE_CMAKE_VERSION=3.15.0

# Derived variables
ANT_INSTALL_DIR=${VENV_HOME}/opt/ant
export ANT_HOME=${ANT_INSTALL_DIR}/apache-ant-${USE_ANT_VERSION}
export ANT_OPTS="-Dhttp.proxyHost=proxy.ecmwf.int -Dhttp.proxyPort=3333 -Dhttps.proxyHost=proxy.ecmwf.int -Dhttps.proxyPort=3333"

CLAW_INSTALL_DIR=${VENV_HOME}/opt/claw
OFP_HOME=${VENV_HOME}/src/open-fortran-parser

# Load modules
module unload java cmake
module load java/${USE_JAVA_VERSION}
module load cmake/${USE_CMAKE_VERSION}

#
# Create Python virtualenv
#

if [ -z $VIRTUAL_ENV ]
then
  module unload python3
  module load python3/${USE_PYTHON_VERSION}
  python3 -m venv "${VENV_HOME}"
  source "${VENV_HOME}/bin/activate"
fi
pip install --upgrade pip
pip install wheel

#
# Clone and install dev version of Loki
#

cd "${LOKI_HOME}"
pip install numpy  # Needed during next step
pip install -r requirements.txt
pip install -e .  # Installs Loki dev copy in editable mode

#
# Install ant
#

# Cache NetRexx if it doesn't exist (Download fails from time to time)
NETREXX_TEMP=${HOME}/.ant/tempcache/NetRexx.zip
mkdir -p "${HOME}/.ant/tempcache"
if [[ $(shasum -a 1 "${NETREXX_TEMP}" | cut -d ' ' -f1) != "1a47bf7b5d0055d4a94befc999c593d15b66c119" ]]
then
  wget -O "${NETREXX_TEMP}" ftp://ftp.software.ibm.com/software/awdtools/netrexx/NetRexx.zip
fi

# Download, unpack and install ant
ANT_ARCHIVE=apache-ant-${USE_ANT_VERSION}-bin.tar.gz
mkdir -p "${ANT_INSTALL_DIR}"
rm -rf "${ANT_HOME}" "${ANT_INSTALL_DIR}/${ANT_ARCHIVE}"
cd "${ANT_INSTALL_DIR}"
wget -O "${ANT_ARCHIVE}" http://mirror.ox.ac.uk/sites/rsync.apache.org/ant/binaries/${ANT_ARCHIVE}
tar -xzf "${ANT_ARCHIVE}"
"${ANT_HOME}/bin/ant" -f "${ANT_HOME}/fetch.xml" -Ddest=optional

#
# Install CLAW+OMNI
#

mkdir -p "${CLAW_INSTALL_DIR}"
cd "${CLAW_INSTALL_DIR}"
rm -rf claw-compiler
git clone --recursive https://github.com/claw-project/claw-compiler.git claw-compiler
cd claw-compiler
cmake -DCMAKE_INSTALL_PREFIX="${CLAW_INSTALL_DIR}" .
make
make install

#
# Install OFP
#

# HACK: Force OFP version and install Java deps
echo "VERSION = '0.5.3'" > "${OFP_HOME}/open_fortran_parser/_version.py"
python3 -m open_fortran_parser --deps

# Copy downloaded binaries to lib
cd "${OFP_HOME}" 
mkdir -p lib
cp open_fortran_parser/*.jar lib

# Rebuild OFP binaries to include custom changes
"${ANT_HOME}/bin/ant"

#
# Create env activation script
#

echo "
# This script activates Loki's virtual environment, loads additional modules and sets dependend paths.
#
# Run as 'source loki-activate'
VENV_HOME=${VENV_HOME}
. \${VENV_HOME}/bin/activate
_OLD_CLASSPATH=\"\${CLASSPATH}\"
CLASSPATH=\"\${CLASSPATH}:\${VENV_HOME}/src/open-fortran-parser/open_fortran_parser/*\"
export CLASSPATH
module load java/${USE_JAVA_VERSION}
export PATH=\"${CLAW_INSTALL_DIR}/bin:\${PATH}\"
" > "${LOKI_HOME}/loki-activate"

#
# Finish
#

echo ""
echo "+---------------------------------------------------------------+"
echo "|                                                               |"
echo "|  Successfully installed Loki with dependencies.               |"
echo "|  Activate with:                                               |"
echo "|                                                               |"
echo "|     source loki-activate                                      |"
echo "|                                                               |"
echo "+---------------------------------------------------------------+"
