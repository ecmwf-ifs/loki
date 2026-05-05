# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from pathlib import Path

import pytest

from loki.frontend import FP


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


@pytest.fixture(name='config')
def fixture_config():
    """
    Default configuration dict with basic options.
    """
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'disable': ['abort'],
            'enable_imports': True,
        },
        'routines': {}
    }


@pytest.fixture(name='frontend')
def fixture_frontend():
    """
    Frontend to use.

    Not parametrizing the tests as the scheduler functionality should be
    independent from the specific frontend used. Cannot use OMNI for this
    as not all tests have dependencies fully resolved.
    """
    return FP
