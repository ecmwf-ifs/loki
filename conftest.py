# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Local pytest plugin to add bespoke pytest extensions for Loki

See the pytest documentation for more details:
https://docs.pytest.org/en/stable/how-to/writing_plugins.html#local-conftest-plugins
"""

import os

from loki.config import config as loki_config


XFAIL_DERIVED_TYPE_JIT_TESTS = os.environ.get('XFAIL_DERIVED_TYPE_JIT_TESTS', '1') == '1'
"""
Flag to mark tests that use JIT compilation for derived type argument procedures
as expected to fail with :any:`pytest.mark.xfail`

The derived type wrapping support provided by f90wrap has stopped working in the
Loki JIT backend that is used for tests. Because this is not a crucial feature, we
are temporarily allowing these tests to fail to fix this separately in the future.

By setting the environment variable ``XFAIL_DERIVED_TYPE_JIT_TESTS`` to ``0``, tests
will no longer be marked as xfail.
"""


def pytest_addoption(parser, pluginmanager):  # pylint: disable=unused-argument
    """
    Add options to the pytest CLI

    Additional options can be specified via ``parser.addoption`` using the same signature as
    :any:`argparse.ArgumentParser.add_argument`.

    For Loki, we add ``--loki-log-level`` to overwrite the log level in :any:`loki.logging`.
    """
    parser.addoption('--loki-log-level', dest='LOKI_LOG_LEVEL', default='INFO',
                     help='Change the Loki log level (ERROR, WARNING, INFO, PERF, DETAIL, DEBUG)')


def pytest_configure(config):
    """
    Apply configuration changes

    This function is invoked after all command line options have been processed
    """
    loki_config['log-level'] = config.option.LOKI_LOG_LEVEL
