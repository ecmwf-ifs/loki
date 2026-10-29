# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
Fixtures for the Loki command-line interface tests.
"""

import pytest

from loki.logging import logger


@pytest.fixture(autouse=True)
def restore_loki_logger():
    """
    Restore the level and handlers of the Loki logger after each test.

    The CLI entry points modify the global Loki logger (for example, ``--debug``
    lowers its level and ``--log`` attaches a file handler), which would
    otherwise leak into unrelated tests that run later in the same process.
    """
    level = logger.level
    handlers = list(logger.handlers)
    yield
    logger.setLevel(level)
    logger.handlers[:] = handlers
