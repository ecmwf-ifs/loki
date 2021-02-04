try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import os
import setuptools.ssl_support

ca_bundle = (os.environ.get('SSL_CERT_FILE') or
             os.environ.get('REQUESTS_CA_BUNDLE') or
             os.environ.get('CURL_CA_BUNDLE'))
if ca_bundle:
    setuptools.ssl_support.cert_paths = [ca_bundle]

setup(name='loki',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Experimental Fortran IR to facilitate source-to-source transformations",
      author="Michael Lange",
      packages=find_packages(exclude=['tests']),
      install_requires=['open-fortran-parser'],
      test_requires=['pytest', 'flake8'],
      scripts=['scripts/loki-transform.py', 'scripts/loki-lint.py'],
     )
