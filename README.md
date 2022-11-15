# Loki: Freely programmable source-to-source translation

[![license](https://img.shields.io/github/license/ecmwf-ifs/loki)](https://www.apache.org/licenses/LICENSE-2.0.html)

**Loki is an experimental tool** to explore the possible use of
source-to-source translation for ECMWF's Integrated Forecasting System (IFS) and
associated Fortran software packages.

Loki is based on compiler technology (visitor patterns and ASTs) and aims to
provide an abstract, language-agnostic representation of a kernel, as well as a
programmable (pythonic) interface that allows developers to experiment with
different kernel implementations and optimizations.  The aim is to allow changes
to programming models and coding styles to be encoded and automated instead of
hand-applying them, enabling advanced experimentation with large kernels as well
as bulk processing of large numbers of source files to evaluate different kernel
implementations and programming models.

*This package is made available to support research collaborations and is not
officially supported by ECMWF.*

## Contact

Michael Lange (michael.lange@ecmwf.int),
Balthasar Reuter (balthasar.reuter@ecmwf.int)

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

Contributions to Loki are welcome. In order to do so, please open an issue where
a feature request or bug can be discussed. Then create a pull request with your
contribution and sign the
[contributors license agreement (CLA)](http://claassistant.ecmwf.int/ecmwf-ifs/loki).
