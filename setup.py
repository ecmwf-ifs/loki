import os
import setuptools

ca_bundle = (os.environ.get('SSL_CERT_FILE') or
             os.environ.get('REQUESTS_CA_BUNDLE') or
             os.environ.get('CURL_CA_BUNDLE'))
if ca_bundle:
    setuptools.ssl_support.cert_paths = [ca_bundle]

setuptools.setup()
