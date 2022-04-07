# Installation

Loki is a pure Python package that depends on a range of upstream packages,
including some dependencies on dev branches. It is therefore recommended
to create a Loki-specific virtual environment. The provided install script will do this automatically.

## Requirements

- Python 3.6+ (3.8 recommended) with virtualenv and pip
- For OpenFortranParser/OMNI+CLAW:
  - JDK 1.8+ with ant (can be installed using the install script or ecbuild)
  - libxml2 (with headers)
- For graphical output from the scheduler: graphviz

## Installation using install script

To install Loki with selected dependencies and using a local virtual environment `loki_env` use the provided `install` script.
Call it with `-h` to display usage information:

```text
$ ./install -h
Loki install script. This installs Loki and selected dependencies.

Usage: ./install [-v] [--ecmwf] [--*-certificate[=]<path>] [--use-venv[=]<path>] [--with-*] [...]

Available options:
  -h                           Display this help message
  -v                           Enable verbose output
  --ecmwf                      Load ECMWF workstation specific modules and settings
  --hpc2020                    Load HPC2020 (Atos) specific modules and settings
  --proxy-certificate[=]<path> Provide https proxy certificate, disable cert verification for JDK/ant/OFP downloads
  --jdk-certificate[=]<path>   Provide trusted certificate for JDK runtime
  --use-venv[=]<path>          Use existing virtual environment at <path>
  --with[out]-jdk              Install JDK instead of using system version (default: use system version)
  --with[out]-ant              Install ant instead of using system version (default: use system version)
  --with[out]-claw             Install CLAW and OMNI Compiler (default: disabled)
  --with[out]-ofp              Install Open Fortran Parser (default: disabled)
  --with[out]-dace             Install DaCe (default: enabled)
  --with[out]-tests            Install dependencies to run tests (default: enabled)
  --with[out]-docs             Install dependencies to generate documentation (default: disabled)
  --with[out]-max              Enable experimental use of Maxeler simulator, requires --ecmwf (default: disabled)
  --claw-install-env[=]<...>   Specify environmental variables for CLAW build and install
```

On an ECMWF workstation the `--ecmwf` flag is recommended as it loads required modules and makes sure the proxy settings are correct.
On the Atos HPC facility, the `--hpc2020` flag is recommended as it loads required modules.
Omitting (other) options (i.e., any of the `--with-*`) will install only the Fparser2 frontend.

### Examples:

The default command on an ECMWF workstation for a full stack installation is

```bash
./install --ecmwf --with-ant --with-claw --with-ofp
```

The equivalent command on the Atos HPC facility for a full stack installation is

```bash
./install --hpc2020 --with-ant --with-claw --with-ofp
```

On the `volta` host it requires a local installation of JDK (note that it will mention missing modules but that does not cause problems because system-versions are sufficiently up-to-date)

```bash
./install --ecmwf --with-jdk --with-ant --with-claw --with-ofp
```

On standard Linux hosts with up-to-date JDK and ant, it is as easy as

```bash
./install --with-claw --with-ofp
```

To update the installation (e.g., to add JDK), the existing virtual environment can be provided, e.g.,

```bash
./install --with-claw --with-jdk --with-ant --use-venv=loki_env --with-ofp
```

## Installation using CMake/ecbuild

Loki and dependencies (excluding OpenFortranParser) can be installed using [ecbuild](https://github.com/ecmwf/ecbuild) (a set of CMake macros and a wrapper around CMake).
This requires ecbuild 3.4+ and CMake 3.17+.

```bash
ecbuild <path-to-loki>
```

It creates a virtual environment and downloads OpenJDK and Ant on-demand if no up-to-date versions have been found.
This installation method is particularly handy when used as a subproject of a larger CMake build.
When used this way, it exports a number of CMake functions that can then be used elsewhere:

- ``loki_transform_convert``: A wrapper for calls to ``loki-transform.py`` in ``convert`` mode that takes care of automatically setting path and environment.
- ``loki_transform_transpile``: A wrapper for calls to ``loki-transform.py`` in ``transpile`` mode that takes care of automatically setting path and environment.
- ``claw_compile``: A wrapper for calls to ``clawfc`` that takes care of automatically setting path and environment.

## Installation as part of a bundle

Loki being installable by CMake/ecbuild makes it easy to integrate with [ecbundle](https://github.com/ecmwf/ecbundle).
Simply add the following to your ``bundle.yml``:

```yaml
projects :

  # ...other projects ...

  - loki :
    git     : ${BITBUCKET}/rdx/loki
    version : master

```

## Manual installation

The following uses a virtual environment to install Loki on your local machine. You can create an empty directory and copy-paste the following steps to obtain a working version:

### 1. Clone the Loki repository

```bash
git clone ssh://git@git.ecmwf.int/rdx/loki.git
```

### 2. Create and activate virtual environment

```bash
python3 -m venv loki_env
source loki_env/bin/activate
```

### 3.  Install Loki and Python dependencies

```bash
pushd loki
pip install -e .[tests]
popd
```

### 4.  Install CLAW with OMNI compiler

```bash
git clone --recursive --single-branch --branch=mlange-dev https://github.com/mlange05/claw-compiler.git claw-compiler
pushd claw-compiler
mkdir build
pushd build
# Now build and install CLAW in the venv:
cmake -DCMAKE_INSTALL_PREFIX=../../loki_env ..
make
make install
popd
popd
```

### 5.  Install OpenFortranParser (OFP)

```bash
pip install -e git+https://github.com/mlange05/open-fortran-parser-xml@mlange05-dev#egg=open-fortran-parser
# Fix version number in OFP to enable download of dependencies:
echo "VERSION = '0.5.3'" > loki_env/src/open-fortran-parser/open_fortran_parser/_version.py
python3 -m open_fortran_parser --deps
pushd loki_env/src/open-fortran-parser
mkdir -p lib
cp open_fortran_parser/*.jar lib
# Rebuild OFP binaries to include custom changes:
ant
popd
```

### 6.  Verify everything is working

```bash
pushd loki
py.test
popd
```
