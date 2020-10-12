## Installation

Loki is a pure Python package that depends on a range of upstream packages,
including some dependencies on dev branches. It is therefore recommended
to create a Loki-specific virtual environment. The provided install script will do this automatically.

### Requirements

- Python 3.6+ with virtualenv and pip
- For OpenFortranParser/OMNI+CLAW: JDK 1.8+ with ant (can be installed using the install script), libxml2(with headers)
- For graphical output from the scheduler: graphviz

### Installation using install script

To install Loki with selected dependencies and using a local virtual environment `loki_env` use the provided [install script](install).
Call it with `-h` to display usage information:

```
$ ./install -h
Loki install script. This installs Loki and selected dependencies.

Usage: ./install [-v] [--ecmwf] [--use-venv[=]<path>] [--with-*]

Available options:
  -h                    Display this help message
  -v                    Enable verbose output
  --ecmwf               Load ECMWF workstation specific modules and settings
  --use-venv[=]<path>   Use existing virtual environment at <path>
  --with-jdk            Install JDK instead of using system version
  --with-ant            Install ant instead of using system version
  --with-claw           Install CLAW and OMNI Compiler
```

On an ECMWF machine the `--ecmwf` flag is recommended as it makes sure the proxy settings are correct.

The default command on an ECMWF workstation is

```
./install --ecmwf --with-ant --with-claw
```

On the `volta` host it requires a local installation of JDK (note that it will mention missing modules but it doesn't cause problems because system-versions are sufficiently up-to-date)

```
./install --ecmwf --with-jdk --with-ant --with-claw
```

On standard Linux hosts with up-to-date JDK and ant, it is as easy as

```
./install --with-claw
```

To update the installation (e.g., to add JDK), the existing virtual environment can be provided, e.g.,

```
./install --with-claw --with-jdk --with-ant --use-venv=loki_env
```

### Manual installation

The following uses a virtual environment to install Loki on your local machine. You can create an empty directory and copy-paste the following steps to obtain a working version:

1. Clone the Loki repository:
```
git clone ssh://git@git.ecmwf.int/~naml/loki.git
```
2. Create and activate virtual environment:
```
python3 -m venv loki_env
source loki_env/bin/activate
pip install --upgrade pip
```
3.  Install Loki and Python dependencies:
```
pushd loki
pip install numpy
pip install -r requirements.txt
pip install -e .
popd
```
4.  Install CLAW with OMNI compiler:
```
git clone --recursive https://github.com/claw-project/claw-compiler.git
pushd claw-compiler
# Fix ant-file with working version of ivy:
sed -i.bak 's/src="https:\/\/repo1.maven.org\/maven2\/org\/apache\/ivy\/ivy\/2.3.0\/ivy-2.3.0.jar"/src="https:\/\/repo1.maven.org\/maven2\/org\/apache\/ivy\/ivy\/2.5.0\/ivy-2.5.0.jar"/g' cx2t/common-targets.xml 
mkdir build
pushd build
# Now build and install CLAW in the venv:
cmake -DCMAKE_INSTALL_PREFIX=../../loki_env ..
make
make install
popd
popd
```
5.  Install OpenFortranParser (OFP):
```
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
6.  Verify everything is working:
```
pushd loki
py.test
popd
```
