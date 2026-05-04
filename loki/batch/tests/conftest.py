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


@pytest.fixture(name='driverA_dependencies')
def fixture_drivera_dependencies():
    return {
        'driverA_mod#driverA': ('kernelA_mod#kernelA', 'header_mod', 'header_mod#header_type'),
        'kernelA_mod#kernelA': ('compute_l1_mod#compute_l1', '#another_l1'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        '#another_l1': ('#another_l2', 'header_mod'),
        '#another_l2': ('header_mod',),
        'header_mod': (),
        'header_mod#header_type': (),
    }


@pytest.fixture(name='driverB_dependencies')
def fixture_driverb_dependencies():
    return {
        'driverB_mod#driverB': (
            'kernelB_mod#kernelB',
            'header_mod#header_type',
            'header_mod'
        ),
        'kernelB_mod#kernelB': ('compute_l1_mod#compute_l1', 'ext_driver_mod#ext_driver'),
        'compute_l1_mod#compute_l1': ('compute_l2_mod#compute_l2',),
        'compute_l2_mod#compute_l2': (),
        'ext_driver_mod#ext_driver': ('ext_kernel_mod', 'ext_kernel_mod#ext_kernel',),
        'ext_kernel_mod': (),
        'ext_kernel_mod#ext_kernel': (),
        'header_mod#header_type': (),
        'header_mod': (),
    }


@pytest.fixture(name='proj_typebound_dependencies')
def fixture_proj_typebound_dependencies():
    return {
        '#driver': (
            'typebound_item',
            'typebound_item#some_type%other_routine',
            'typebound_item#some_type%some_routine',
            'typebound_header',
            'typebound_header#header_type%member_routine',
            'typebound_header#header_type%routine',
            'typebound_other#other_type',
            'typebound_other#other_type%member',
            'typebound_other#other_type%var%member_routine',
        ),
        'typebound_item': ('typebound_header',),
        'typebound_item#some_type': (),
        'typebound_item#some_type%some_routine': ('typebound_item#some_routine',),
        'typebound_item#some_type%other_routine': ('typebound_item#other_routine',),
        'typebound_item#some_type%routine': ('typebound_item#module_routine',),
        'typebound_item#some_type%routine1': ('typebound_item#routine1',),
        'typebound_item#some_type%routine2': ('typebound_item#routine',),
        'typebound_item#routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%some_routine',
        ),
        'typebound_item#routine1': (
            'typebound_item#module_routine',
            'typebound_item#some_type'
        ),
        'typebound_item#some_routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%routine',
        ),
        'typebound_item#other_routine': (
            'typebound_item#some_type',
            'typebound_item#some_type%routine1',
            'typebound_item#some_type%routine2',
            'typebound_header#abor1'
        ),
        'typebound_item#module_routine': (),
        'typebound_header': (),
        'typebound_header#header_type': (),
        'typebound_header#header_type%member_routine': ('typebound_header#header_member_routine',),
        'typebound_header#header_member_routine': ('typebound_header#header_type',),
        'typebound_header#header_type%routine': (
            'typebound_header#header_type%routine_real',
            'typebound_header#header_type%routine_integer'
        ),
        'typebound_header#header_type%routine_real': ('typebound_header#header_routine_real',),
        'typebound_header#header_routine_real': ('typebound_header#header_type',),
        'typebound_header#header_type%routine_integer': ('typebound_header#routine_integer',),
        'typebound_header#routine_integer': ('typebound_header#header_type',),
        'typebound_header#abor1': (),
        'typebound_other#other_type': ('typebound_header#header_type',),
        'typebound_other#other_type%member': ('typebound_other#other_member',),
        'typebound_other#other_member': (
            'typebound_header#header_member_routine',
            'typebound_other#other_type',
            'typebound_other#other_type%var%member_routine'
        ),
        'typebound_other#other_type%var%member_routine': ('typebound_header#header_type%member_routine',)
    }
