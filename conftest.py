# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Local pytest plugin to add bespoke pytest extensions for Loki

See the pytest documentation for more details:
https://docs.pytest.org/en/stable/how-to/writing_plugins.html#local-conftest-plugins
"""

from loki.config import config as loki_config


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
