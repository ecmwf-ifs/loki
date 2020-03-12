## Installation

Loki is pure Python package that depends on a range of upstream packages,
including some dependencies on dev branches. It is therefore recommended
to create a Loki-specific virtual environment.

### Requirements
- Python 3.6+
- For OpenFortranParser/OMNI: JDK 1.8+ with ant

### Installation with OFP and OMNI using virtual environment
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
py.test -v tests
popd
```
