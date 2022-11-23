# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import os
import setuptools

ca_bundle = (os.environ.get('SSL_CERT_FILE') or
             os.environ.get('REQUESTS_CA_BUNDLE') or
             os.environ.get('CURL_CA_BUNDLE'))
if ca_bundle:
    # Provide custom CA paths to setuptools when behind a proxy server that
    # uses a self-signed certificate for SSL connections.
    # Note: setuptools>=57.1.0 dropped ssl_support and relies entirely on urllib,
    #       thus proxy certificates passed to pip should already be used.
    from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
    if parse_version(setuptools.__version__) < parse_version("57.1.0"):
        import setuptools.ssl_support  # pylint: disable=import-outside-toplevel,ungrouped-imports
        setuptools.ssl_support.cert_paths = [ca_bundle]

setuptools.setup()
