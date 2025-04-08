# Installation

There are multiple different ways of installing Loki, tailored towards various
use-cases:

- via `pip install` as a pure Python package
- via the provided install script to ease the setup of optional dependencies
- via CMake/ecbuild to enable installation as part of a CMake project
- via ecbundle
- manually

Loki is a pure Python package that depends on a range of upstream packages. We
recommend to use a
[virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
to avoid conflicts with versions of system-wide installed packages. The
CMake/ecbuild installation method as well as the provided script `install` will
do this automatically.

## Requirements

- Python 3.9+ with virtualenv and pip
- For graphical output of Scheduler dependency graphs: graphviz

### Optional requirements

The following is required to use the OMNI frontend:

- JDK 1.8+ (can be installed using the install script or ecbuild)
- libxml2 (with headers)

## Installation without prior download

The easiest way to obtain a useable installation of Loki with the fparser frontend does not
require downloading the source code. Simply run the following commands:

```bash
python3 -m venv loki_env  # Create a virtual environment
source loki_env/bin/activate  # Activate the virtual environment

# Installation of the Loki core library
pip install "loki @ git+https://github.com/ecmwf-ifs/loki.git"

# Optional: Installation of IFS lint rules for the use as a linter
pip install "lint_rules @ git+https://github.com/ecmwf-ifs/loki.git#subdirectory=lint_rules"
```

This makes the Python package available and installs the scripts `loki-transform.py` and `loki-lint.py` on the PATH.


## Installation from source

After downloading the source code, e.g., via

```bash
git clone https://github.com/ecmwf-ifs/loki.git
```

enter the created source directory and choose one of the following installation methods.

### Installation with pip

This yields an installation that uses the fparser frontend and is suitable for
development of transformations and working with the example notebooks:

```bash
python3 -m venv loki_env  # Create a virtual environment
source loki_env/bin/activate  # Activate the virtual environment

# Installation of the Loki core library
# Optional:
#   * Add `-e` to obtain an editable install that allows modifying the
#     source files without having to re-install the package
#   * Enable the following options by providing them as a comma-separated
#     list in square brackets behind the `.`:
#     * tests    - allows running the Loki test suite
#     * examples - installs dependencies to run the example notebooks
#     * docs     - installs dependencies required to generate the Sphinx documentation
#     * dace     - installs DaCe
pip install .

# Optional: Installation of IFS lint rules for the use as a linter
#           (again optionally with `-e` for an editable install)
pip install ./lint_rules
```

### Installation using install script

The provided `install` script can be used to install Loki with selected
dependencies inside a local virtual environment `loki_env`. This is the
recommended way when additional optional dependencies, such as the OMNI
frontend, are required.

After downloading Loki, call the script with `-h` to display usage information:

```text
$ ./install -h
Loki install script. This installs Loki and selected dependencies.

Usage: ./install [-v] [--hpc2020] [--use-venv[=]<path>] [--with-*] [...]

Available options:
  -h / --help                  Display this help message
  -v                           Enable verbose output
  --hpc2020                    Load HPC2020 (Atos) specific modules and settings
  --use-venv[=]<path>          Use existing virtual environment at <path>
  --with[out]-jdk              Install JDK instead of using system version (default: use system version)
  --with[out]-omni             Install OMNI Compiler (default: disabled)
  --with[out]-dace             Install DaCe (default: enabled)
  --with[out]-tests            Install dependencies to run tests (default: enabled)
  --with[out]-docs             Install dependencies to generate documentation (default: disabled)
  --with[out]-examples         Install dependencies to run the example notebooks (default: enabled)
```

On the ECMWF Atos HPC facility, the `--hpc2020` flag is recommended as it loads
required modules.  Omitting all (other) options (i.e., any of the `--with-*`)
will install only the Fparser2 frontend.

After completion, this script writes a `loki-activate` file that can be sourced
to bring up the virtual environment and set paths for the external dependencies.

#### Examples:

The default command on ECMWF's Atos HPC facility for a full stack installation is

```bash
./install --hpc2020 --with-omni
```

On standard Linux hosts with up-to-date JDK and ant, it is as easy as

```bash
./install --with-omni
```

To update the installation (e.g., to add JDK), the existing virtual environment can be provided, e.g.,

```bash
./install --with-omni --with-jdk --use-venv=loki_env
```

### Installation using CMake/ecbuild

Loki and dependencies (excluding OpenFortranParser) can be installed using
[ecbuild](https://github.com/ecmwf/ecbuild) (a set of CMake macros and a wrapper
around CMake). This requires ecbuild 3.7+ and CMake 3.19+.

```bash
cmake -DCMAKE_MODULE_PATH=</path-to-ecbuild>/cmake -S <path-to-loki> -B <build-dir>
cmake --build <build-dir>
```

The following options are available and can be enabled/disabled by providing `-DENABLE_<feature>=<ON|OFF>`:

- `NO_INSTALL` (default: `OFF`): Do not install Loki but make the CMake
  functions below available. This is useful if Loki is available on the path from
  elsewhere and only the CMake integration is required
- `EDITABLE` (default: `OFF`): Install Loki in editable mode, i.e. without
  copying any files
- `OMNI` (default: `OFF`): Install the OMNI compiler as well as its
  Java dependencies as required. Note that this is an experimental setup and comes
  with no support or guarantees.

This method is also suitable to create a system-wide installation of Loki.
After running the above steps, install Loki to a chosen prefix using

```bash
cmake --install <build-dir> --prefix <install-path>
```

*Note: Using this to install Loki system-wide does currently not install the OMNI frontend with it, even if the relevant ecbuild option is activated. It is recommended to install them separately, if required.*

The ecbuild installation method creates a virtual environment in the build
directory and downloads OpenJDK and Ant on-demand if no up-to-date versions have
been found. This installation method is particularly handy when used as a
subproject of a larger CMake build.

When used this way, it exports a number of CMake functions that can then be used
elsewhere:

- `loki_transform`: A wrapper for calls to `loki-transform.py` that takes care
  of automatically setting path and environment.
- `loki_transform_plan`: A wrapper for calls to `loki-transform.py` in `plan`
  mode to generate CMake plan files.
- `loki_transform_target`: A wrapper that takes care of calling the plan mode
  during configuration and applying bulk transformations at build time to a CMake
  target. This includes updates to the target's source file list as determined
  during the planning stage.
- `generate_xmod`: A wrapper for calls to OMNI's `F_Front` frontend to generate
  xmod dependency files.

This allows to apply transformations as part of the build process without the
need to take care of PATH handling on the user side. See the [CLOUDSC
dwarf](https://github.com/ecmwf-ifs/dwarf-p-cloudsc) for an example how this can
be used.

### Offline installation using CMake/ecbuild

When the CMake/ecbuild installation procedure is required on a system without
internet access, then the required Python wheels can be downloaded and transferred
to the target system. To do so, run the [`populate`](populate) script from the Loki
main directory.

This will download all required Python weels into a directory `artifacts`.
Transfer this wheelhouse directory to the target system and provide
`-DARTIFACTS_DIR=<path-to-artifacts>` to the CMake command when installing Loki.

The behaviour of this script can be customized using the following environment variables:

- `ARTIFACTS_DIR`: Choose a different target directory (default: `artifacts` in the current
  working directory)
- `LOKI_INSTALL_OPTIONS`: Add additional PIP install options to ensure dependencies for this
  are included in the wheelhouse. Most commonly required is `[tests]`.
- `LOKI_WHEEL_PYTHON_VERSION`: When using a different Python version to download the wheels
  than on the target system, specify the version here (e.g., `LOKI_WHEEL_PYTHON_VERSION=312`
  to request wheels for Python 3.12). See the
  [PIP documentation](https://pip.pypa.io/en/stable/cli/pip_download/#cmdoption-python-version)
  for more details.
- `LOKI_WHEEL_ARCH`: When the system that downloads the wheels uses a different architecture
  than the target system (e.g., an ARM-based MacBook is used to download wheels for a Linux
  x86_64 system), specify the target architecture here (e.g.,
  `LOKI_WHEEL_ARCH=manylinux_2_17_x86_64`). A list of typical platform tags are available
  [here](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/#platform-tag).


## Installation on MacOS

Although tailored to the Linux environment commonly found on HPC systems, Loki
can also be installed on MacOS.

This requires installing some additional dependencies using
[Brew](https://brew.sh) to allow running the Loki test suite:


```bash
# Install dependencies with brew
brew install gcc@13 graphviz python@3.11

# Install Loki using the install script
# NB: we explicitly select Python 3.11 (in case a newer version is the default)
#     by adding it in first place to the search path
PATH="$(brew --prefix)/opt/python@3.11/libexec/bin:$PATH" \
  CC=gcc-13 CXX=g++-13 FC=gfortran-13 \
  ./install --with-examples --with-tests --with-dace

# Amend the Loki environment with correct compiler variables
echo "export PATH=$(brew --prefix)/opt/python@3.11/libexec/bin:$(brew --prefix)/bin:${PATH}" | cat - loki-activate > loki-activate.tmp
mv loki-activate.tmp loki-activate
echo "export CC=gcc-13" >> loki-activate
echo "export CXX=g++-13" >> loki-activate
echo "export FC=gfortran-13" >> loki-activate
echo "export F90=gfortran-13" >> loki-activate
echo "export LD=gfortran-13" >> loki-activate

# Activate the virtual environment to run the tests
source loki-activate
pytest --pyargs loki
```

## Installation as part of an ecbundle bundle

Loki being installable by CMake/ecbuild makes it easy to integrate with
[ecbundle](https://github.com/ecmwf/ecbundle). Simply add the following to your
`bundle.yml`:

```yaml
projects :

  # ...other projects ...

  - loki :
    git     : https://github.com/ecmwf-ifs/loki
    version : main

```

See the [CLOUDSC dwarf](https://github.com/ecmwf-ifs/dwarf-p-cloudsc) for an
example how this can be used.

## Manual installation

The following outlines the manual steps for installing Loki using a virtual
environment. This installation method is not recommended but may be used when
maximum control over all steps is required or all of the above are not working.
You can create an empty directory and copy-paste the following steps to obtain a
working version:

### 1. Clone the Loki repository

```bash
git clone https://github.com/ecmwf-ifs/loki
```

### 2. Create and activate virtual environment

```bash
python3 -m venv loki_env
source loki_env/bin/activate
pip install --upgrade pip
```

Note that we need to make sure that we use a recent pip version (21.3 or newer)
that has support for editable installs using `pyproject.toml`.

### 3.  Install Loki and Python dependencies

```bash
pushd loki
pip install -e .[tests,examples]
pip install -e ./lint_rules
popd
```

### 4.  Install OMNI frontend -- optional

#### Option a: install latest xcodeml-tools

```bash
git clone --recursive --single-branch https://github.com/omni-compiler/xcodeml-tools.git xcodeml-tools
pushd xcodeml-tools
# Now build and install OMNI in the venv:
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=../loki_env
cmake --build build
cmake --install build
popd
```

#### Option b: install (older) OMNI version and CLAW

```bash
git clone --recursive --single-branch --branch=mlange-dev https://github.com/mlange05/claw-compiler.git claw-compiler
pushd claw-compiler
# Now build and install CLAW in the venv:
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=../loki_env
cmake --build build
cmake --install build
popd
```

### 5.  Verify everything is working

```bash
pushd loki
py.test transformations lint_rules .
popd
```

Note that the order is important to avoid clashes with conftest utilities.
