# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import config


@pytest.fixture(name='reset_frontend_mode')
def fixture_reset_frontend_mode():
    original_frontend_mode = config['frontend-strict-mode']
    yield
    config['frontend-strict-mode'] = original_frontend_mode
