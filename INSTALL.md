# Installation

There are multiple different ways of installing Loki, tailored towards various use-cases:

- via `pip install` as a pure Python package
- via the provided install script to ease the setup of optional dependencies
- via CMake/ecbuild to enable installation as part of a CMake project
- via ecbundle
- manually

Loki is a pure Python package that depends on a range of upstream packages. We recommend to use a
[virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
to avoid conflicts with versions of system-wide installed packages. The CMake/ecbuild installation method
as well as the provided [install script](install) will do this automatically.

## Requirements

- Python 3.8+ with virtualenv and pip
- For OpenFortranParser/OMNI+CLAW:
  - JDK 1.8+ with ant (can be installed using the install script or ecbuild)
  - libxml2 (with headers)
- For graphical output from the scheduler: graphviz

## Installation via `pip install`

The easiest way to obtain a useable installation of Loki with the fparser frontend:

```bash
python3 -m venv loki_env  # Create a virtual environment
source loki_env/bin/activate  # Activate the virtual environment

# Installation of the Loki core library
pip install "loki @ git+https://github.com/ecmwf-ifs/loki.git"

# Optional: Installation of additional transformations
pip install "transformations @ git+https://github.com/ecmwf-ifs/loki.git#subdirectory=scripts/transformations"

# Optional: Installation of rules for the use as a linter
pip install "lint_rules @ git+https://github.com/ecmwf-ifs/loki.git#subdirectory=scripts/lint_rules"
```

This makes the Python package available and installs the scripts `loki-transform.py` and `loki-lint.py` on the PATH.


## Installation using install script

The provided `install` script can be used to install Loki with selected dependencies inside a local virtual environment `loki_env`.
This is the recommended way when additional optional dependencies, such as CLAW, the OMNI frontend, or Open Fortran Parser are required.

After downloading Loki, call the script with `-h` to display usage information:

```text
$ ./install -h
Loki install script. This installs Loki and selected dependencies.

Usage: ./install [-v] [--hpc2020] [--*-certificate[=]<path>] [--use-venv[=]<path>] [--with-*] [...]

Available options:
  -h / --help                  Display this help message
  -v                           Enable verbose output
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
  --with[out]-examples         Install dependencies to run the example notebooks (default: enabled)
  --claw-install-env[=]<...>   Specify environmental variables for CLAW build and install
```

On the Atos HPC facility, the `--hpc2020` flag is recommended as it loads required modules.
Omitting all (other) options (i.e., any of the `--with-*`) will install only the Fparser2 frontend.

After completion, this script writes a `loki-activate` file that can be sourced to bring up the virtual environment and set paths for the external dependencies.

### Examples:

The default command on ECMWF's Atos HPC facility for a full stack installation is

```bash
./install --hpc2020 --with-ant --with-claw --with-ofp
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
make
```

The following options are available and can be enabled/disabled by providing `-DENABLE_<feature>=<ON|OFF>`:

- `NO_INSTALL` (default: `OFF`): Do not install Loki but make the CMake functions below available. This is useful if Loki is available on the path from elsewhere and only the CMake integration is required
- `EDITABLE` (default: `OFF`): Install Loki in editable mode, i.e. without copying any files
- `CLAW` (default: `OFF`): Install the CLAW and OMNI Compiler as well as its Java dependencies as required. Note that this is an experimental setup and comes with no support or guarantees.

This method is also suitable to create a system-wide installation of Loki:

```bash
mkdir build && cd build
ecbuild --prefix=<install-path> <path-to-loki>
make install
```

*Note: Using this to install Loki system-wide does currently not install OMNI Compiler and CLAW Compiler with it, even if the relevant ecbuild option is activated. It is recommended to install them separately, if required.*

The ecbuild installation method creates a virtual environment in the build directory and downloads OpenJDK and Ant on-demand if no up-to-date versions have been found.
This installation method is particularly handy when used as a subproject of a larger CMake build.

When used this way, it exports a number of CMake functions that can then be used elsewhere:

- `loki_transform_convert`: A wrapper for calls to `loki-transform.py` in `convert` mode that takes care of automatically setting path and environment.
- `loki_transform_transpile`: A wrapper for calls to `loki-transform.py` in `transpile` mode that takes care of automatically setting path and environment.
- `claw_compile`: A wrapper for calls to `clawfc` that takes care of automatically setting path and environment.
- `generate_xmod`: A wrapper for calls to OMNI's `F_Front` frontend to generate xmod dependency files.
- `loki_transform_plan`: A wrapper for calls to `loki-transform.py` in `plan` mode to generate CMake plan files.
- `loki_transform_ecphys`: A wrapper for calls to `loki-transform.py` in `ecphys` mode to apply bulk transformations across the ec_phys call tree
- `loki_transform_target`: A wrapper that takes care of calling the plan mode during configuration and applying bulk transformations at build time to a CMake target. This includes updates to the target's source file list as determined during the planning stage.

This allows to apply transformations as part of the build process without the need to take care of PATH handling on the user side. See the [CLOUDSC dwarf](https://github.com/ecmwf-ifs/dwarf-p-cloudsc) for an example how this can be used.

## Installation as part of an ecbundle bundle

Loki being installable by CMake/ecbuild makes it easy to integrate with [ecbundle](https://github.com/ecmwf/ecbundle).
Simply add the following to your ``bundle.yml``:

```yaml
projects :

  # ...other projects ...

  - loki :
    git     : https://github.com/ecmwf-ifs/loki
    version : main

```

See the [CLOUDSC dwarf](https://github.com/ecmwf-ifs/dwarf-p-cloudsc) for an example how this can be used.

## Manual installation

The following outlines the manual steps for installing Loki using a virtual environment. This installation method is not recommended but may be used when maximum control over all steps is required or all of the above are not working. You can create an empty directory and copy-paste the following steps to obtain a working version:

### 1. Clone the Loki repository

```bash
git clone https://github.com/ecmwf-ifs/loki
```

### 2. Create and activate virtual environment

```bash
python3 -m venv loki_env
source loki_env/bin/activate
```

### 3.  Install Loki and Python dependencies

```bash
pushd loki
pip install -e .[tests,examples]
pip install -e scripts/transformations
pip install -e scripts/lint_rules
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
