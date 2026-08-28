# Loki: Freely programmable source-to-source translation

[![license](https://img.shields.io/github/license/ecmwf-ifs/loki)](https://www.apache.org/licenses/LICENSE-2.0.html)
[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/incubating_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#incubating)
[![code-checks](https://github.com/ecmwf-ifs/loki/actions/workflows/code_checks.yml/badge.svg)](https://github.com/ecmwf-ifs/loki/actions/workflows/code_checks.yml)
[![tests](https://github.com/ecmwf-ifs/loki/actions/workflows/tests.yml/badge.svg)](https://github.com/ecmwf-ifs/loki/actions/workflows/tests.yml)
[![regression-tests](https://github.com/ecmwf-ifs/loki/actions/workflows/regression_tests.yml/badge.svg)](https://github.com/ecmwf-ifs/loki/actions/workflows/regression_tests.yml)
[![codecov](https://codecov.io/gh/ecmwf-ifs/loki/branch/main/graph/badge.svg?token=9ZDS95SFWI)](https://codecov.io/gh/ecmwf-ifs/loki)

> [!IMPORTANT]
> This software is **Incubating** and subject to ECMWF's guidelines on
  [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

The Loki package is intended to explore the possible use of
source-to-source translation for ECMWF's Integrated Forecasting System
(IFS) and associated Fortran software packages. It consists of a general
code transformation API and a set of general transformations that may
be applied to appropriate Fortran codes at build time via Python
scripts or CMake macros.

Loki is based on compiler technology (visitor patterns and ASTs) and aims to
provide an abstract, language-agnostic representation of a kernel, as well as a
programmable (pythonic) interface that allows developers to experiment with
different kernel implementations and optimizations. The aim is to allow changes
to programming models and coding styles to be encoded and automated instead of
hand-applying them, enabling advanced experimentation with large kernels as well
as bulk processing of large numbers of source files to evaluate different kernel
implementations and programming models.

## Contact

Michael Lange (michael.lange@ecmwf.int),
Ahmad Nawab (ahmad.nawab@ecmwf.int)
Michael Staneker (michael.staneker@ecmwf.int)

## License

Loki is distributed under the [Apache License 2.0](LICENSE). In applying this
licence, ECMWF does not waive the privileges and immunities granted to it by
virtue of its status as an intergovernmental organisation nor does it submit to
any jurisdiction.

## Installation

See [INSTALL.md](INSTALL.md).

## Documentation

Loki has a comprehensive [documentation](https://sites.ecmwf.int/docs/loki) that
describes the API and how to use it to write custom transformations.  There are
also a number of Jupyter notebooks available in the
[example directory](https://github.com/ecmwf-ifs/loki/blob/main/example) that help
getting up to speed with the core functionality of the package.

## Contributing

Contributions to Loki are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.
All participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).
