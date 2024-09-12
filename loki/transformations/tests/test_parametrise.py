# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A selection of tests for the parametrisation functionality.
"""
from pathlib import Path
import pytest
import numpy as np

from loki import Scheduler, fgen
from loki.build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes

from loki.transformations.parametrise import ParametriseTransformation


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
        },
        'routines': {
            'driver': {
                'role': 'driver',
                'expand': True,
            },
            'another_driver': {
                'role': 'driver',
                'expand': True,
            },
        }
    }


def compile_and_test(scheduler, tmp_path, a=5, b=1):
    """
    Compile the source code and call the driver function in order to test the results for correctness.
    """
    # Pick out the source files to compile
    driver_path_map = {item: item.source.path.stem for item in scheduler.items if 'driver' in item.name}
    path_source_map = {item.source.path.stem: item.source for item in driver_path_map}

    # Compile each file only once
    path_module_map = {
        stem: jit_compile(source, filepath=tmp_path/f'{stem}.F90', objname=stem)
        for stem, source in path_source_map.items()
    }

    # Run and validate each driver
    for item, stem in driver_path_map.items():
        c = np.zeros((a, b), dtype=np.int32, order='F')
        d = np.zeros((b,), dtype=np.int32, order='F')
        if item.local_name == 'driver':
            path_module_map[stem].driver(a, b, c, d)
            assert (c == 11).all()
            assert (d == 42).all()
        elif item.local_name == 'another_driver':
            path_module_map[stem].another_driver(a, b, c)
            assert (c == 11).all()
        else:
            assert False, f'Unknown driver name {item.local_name}'


def check_arguments_and_parameter(scheduler, subroutine_arguments, call_arguments, parameter_variables):
    """
    Check the parameters, subroutine and call arguments of each subroutine.
    """
    item = scheduler['parametrise#driver']
    routine_parameters = [var for var in item.ir.variables if var.type.parameter]
    assert routine_parameters == parameter_variables["driver"]
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["driver"]
    for call in FindNodes(ir.CallStatement).visit(item.ir.body):
        if "kernel1" in call.name:
            assert call.arguments == call_arguments["kernel1"]
        elif "kernel2" in call.name:
            assert call.arguments == call_arguments["kernel2"]
    item = scheduler['parametrise#another_driver']
    routine_parameters = [var for var in item.ir.variables if var.type.parameter]
    assert routine_parameters == parameter_variables["another_driver"]
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["another_driver"]
    for call in FindNodes(ir.CallStatement).visit(item.ir.body):
        if "kernel1" in call.name:
            assert call.arguments == call_arguments["kernel1"]
    item = scheduler['parametrise#kernel1']
    routine_parameters = [var for var in item.ir.variables if var.type.parameter]
    assert routine_parameters == parameter_variables["kernel1"]
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["kernel1"]
    item = scheduler['parametrise#kernel2']
    routine_parameters = [var for var in item.ir.variables if var.type.parameter]
    assert routine_parameters == parameter_variables["kernel2"]
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["kernel2"]
    for call in FindNodes(ir.CallStatement).visit(item.ir.body):
        if "device1" in call.name:
            assert call.arguments == call_arguments["device1"]
        elif "device2" in call.name:
            assert call.arguments == call_arguments["device2"]
    item = scheduler['parametrise#device1']
    routine_parameters = [var for var in item.ir.variables if var.type.parameter]
    assert routine_parameters == parameter_variables["device1"]
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["device1"]
    for call in FindNodes(ir.CallStatement).visit(item.ir.body):
        if "device2" in call.name:
            assert call.arguments == call_arguments["device2"]
    item = scheduler['parametrise#device2']
    routine_parameters = [var for var in item.ir.variables if var.type.parameter]
    assert routine_parameters == parameter_variables["device2"]
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["device2"]


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_source(tmp_path, testdir, frontend, config):
    """
    Test the actual source code without any transformations applied.
    """

    proj = testdir/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'],
        frontend=frontend, xmods=[tmp_path]
    )

    # check generated source code
    subroutine_arguments = {
        "driver": ['a', 'b', 'c', 'd'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c'],
        "kernel2": ['a_new', 'b', 'd'],
        "device1": ['a', 'b', 'd', 'x', 'y'],
        "device2": ['a', 'b', 'd', 'x'],
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b', 'd'),
        "device1": ('a_new', 'b', 'd', 'x', 'k2_tmp'),
        "device2": ('a', 'b', 'd', 'x')
    }

    parameter_variables = {
        "driver": [],
        "another_driver": [],
        "kernel1": [],
        "kernel2": [],
        "device1": [],
        "device2": [],
    }

    check_arguments_and_parameter(scheduler=scheduler, subroutine_arguments=subroutine_arguments,
                                  call_arguments=call_arguments, parameter_variables=parameter_variables)

    compile_and_test(scheduler=scheduler, tmp_path=tmp_path, a=a, b=b)


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_simple(tmp_path, testdir, frontend, config):
    """
    Basic testing of parametrisation functionality.
    """

    proj = testdir/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'],
        frontend=frontend, xmods=[tmp_path]
    )

    transformation = ParametriseTransformation(dic2p=dic2p)
    scheduler.process(transformation=transformation)

    subroutine_arguments = {
        "driver": ['parametrised_a', 'parametrised_b', 'c', 'd'],
        "another_driver": ['parametrised_a', 'parametrised_b', 'c'],
        "kernel1": ['c'],
        "kernel2": ['d'],
        "device1": ['d', 'x', 'y'],
        "device2": ['d', 'x'],
    }

    call_arguments = {
        "kernel1": ('c',),
        "kernel2": ('d',),
        "device1": ('d', 'x', 'k2_tmp'),
        "device2": ('d', 'x')
    }

    parameter_variables = {
        "driver": ['a', 'b'],
        "another_driver": ['a', 'b'],
        "kernel1": ['a', 'b'],
        "kernel2": ['a_new', 'b'],
        "device1": ['a', 'b'],
        "device2": ['a', 'b'],
    }

    check_arguments_and_parameter(scheduler=scheduler, subroutine_arguments=subroutine_arguments,
                                  call_arguments=call_arguments, parameter_variables=parameter_variables)

    compile_and_test(scheduler=scheduler, tmp_path=tmp_path, a=a, b=b)


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_simple_replace_by_value(tmp_path, testdir, frontend, config):
    """
    Basic testing of parametrisation functionality including replacing of the variables with the actual values.
    """

    proj = testdir/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'],
        frontend=frontend, xmods=[tmp_path]
    )

    transformation = ParametriseTransformation(dic2p=dic2p, replace_by_value=True)
    scheduler.process(transformation=transformation)

    subroutine_arguments = {
        "driver": ['parametrised_a', 'parametrised_b', 'c', 'd'],
        "another_driver": ['parametrised_a', 'parametrised_b', 'c'],
        "kernel1": ['c'],
        "kernel2": ['d'],
        "device1": ['d', 'x', 'y'],
        "device2": ['d', 'x'],
    }

    call_arguments = {
        "kernel1": ('c',),
        "kernel2": ('d',),
        "device1": ('d', 'x', 'k2_tmp'),
        "device2": ('d', 'x')
    }

    parameter_variables = {
        "driver": [],
        "another_driver": [],
        "kernel1": [],
        "kernel2": [],
        "device1": [],
        "device2": [],
    }

    check_arguments_and_parameter(scheduler=scheduler, subroutine_arguments=subroutine_arguments,
                                  call_arguments=call_arguments, parameter_variables=parameter_variables)

    routine_spec_str = fgen(scheduler['parametrise#driver'].ir.spec)
    assert f'c({a}, {b})' in routine_spec_str
    assert f'd({b})' in routine_spec_str
    routine_spec_str = fgen(scheduler['parametrise#another_driver'].ir.spec)
    assert f'c({a}, {b})' in routine_spec_str
    assert f'x({a})' in routine_spec_str
    routine_spec_str = fgen(scheduler['parametrise#kernel1'].ir.spec)
    assert f'c({a}, {b})' in routine_spec_str
    assert f'x({a})' in routine_spec_str
    assert f'y({a}, {b})' in routine_spec_str
    assert f'k1_tmp({a}, {b})' in routine_spec_str
    routine_spec_str = fgen(scheduler['parametrise#kernel2'].ir.spec)
    assert f'd({b})' in routine_spec_str
    assert f'x({a})' in routine_spec_str
    assert f'k2_tmp({a}, {a})' in routine_spec_str
    routine_spec_str = fgen(scheduler['parametrise#device1'].ir.spec)
    assert f'd({b})' in routine_spec_str
    assert f'x({a})' in routine_spec_str
    assert f'y({a}, {a})' in routine_spec_str
    routine_spec_str = fgen(scheduler['parametrise#device2'].ir.spec)
    assert f'd({b})' in routine_spec_str
    assert f'x({a})' in routine_spec_str
    assert f'z({b})' in routine_spec_str
    assert f'd2_tmp({b})' in routine_spec_str

    compile_and_test(scheduler=scheduler, tmp_path=tmp_path, a=a, b=b)


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_modified_callback(tmp_path, testdir, frontend, config):
    """
    Testing of the parametrisation functionality with modified callbacks for failed sanity checks.
    """

    proj = testdir/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    subroutine_arguments = {
        "driver": ['parametrised_a', 'parametrised_b', 'c', 'd'],
        "another_driver": ['parametrised_a', 'parametrised_b', 'c'],
        "kernel1": ['c'],
        "kernel2": ['d'],
        "device1": ['d', 'x', 'y'],
        "device2": ['d', 'x'],
    }

    call_arguments = {
        "kernel1": ('c',),
        "kernel2": ('d',),
        "device1": ('d', 'x', 'k2_tmp'),
        "device2": ('d', 'x')
    }

    parameter_variables = {
        "driver": ['a', 'b'],
        "another_driver": ['a', 'b'],
        "kernel1": ['a', 'b'],
        "kernel2": ['a_new', 'b'],
        "device1": ['a', 'b'],
        "device2": ['a', 'b'],
    }

    def error_stop(**kwargs):
        msg = kwargs.get("msg")
        abort = (ir.Intrinsic(text=f'error stop "{msg}"'),)
        return abort

    def stop_execution(**kwargs):
        msg = kwargs.get("msg")
        abort = (ir.CallStatement(name=sym.Variable(name="stop_execution"), arguments=(sym.StringLiteral(f'{msg}'),)),)
        return abort

    abort_callbacks = (error_stop, stop_execution)

    for _, abort_callback in enumerate(abort_callbacks):
        scheduler = Scheduler(
            paths=[proj], config=config, seed_routines=['driver', 'another_driver'],
            frontend=frontend, xmods=[tmp_path]
        )
        transformation = ParametriseTransformation(dic2p=dic2p, abort_callback=abort_callback)
        scheduler.process(transformation=transformation)

        check_arguments_and_parameter(scheduler=scheduler, subroutine_arguments=subroutine_arguments,
                                      call_arguments=call_arguments, parameter_variables=parameter_variables)

        compile_and_test(scheduler=scheduler, tmp_path=tmp_path, a=a, b=b)


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_modified_callback_wrong_input(tmp_path, testdir, frontend, config):
    """
    Testing of the parametrisation functionality with modified callback for failed sanity checks including test of
    a failed sanity check.
    """

    proj = testdir/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}

    def only_warn(**kwargs):
        msg = kwargs.get("msg")
        abort = (ir.Intrinsic(text=f'print *, "This is just a warning: {msg}"'),)
        return abort

    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'],
        frontend=frontend, xmods=[tmp_path]
    )
    transformation = ParametriseTransformation(dic2p=dic2p, abort_callback=only_warn)
    scheduler.process(transformation=transformation)

    subroutine_arguments = {
        "driver": ['parametrised_a', 'parametrised_b', 'c', 'd'],
        "another_driver": ['parametrised_a', 'parametrised_b', 'c'],
        "kernel1": ['c'],
        "kernel2": ['d'],
        "device1": ['d', 'x', 'y'],
        "device2": ['d', 'x'],
    }

    call_arguments = {
        "kernel1": ('c',),
        "kernel2": ('d',),
        "device1": ('d', 'x', 'k2_tmp'),
        "device2": ('d', 'x')
    }

    parameter_variables = {
        "driver": ['a', 'b'],
        "another_driver": ['a', 'b'],
        "kernel1": ['a', 'b'],
        "kernel2": ['a_new', 'b'],
        "device1": ['a', 'b'],
        "device2": ['a', 'b'],
    }

    check_arguments_and_parameter(scheduler=scheduler, subroutine_arguments=subroutine_arguments,
                                  call_arguments=call_arguments, parameter_variables=parameter_variables)

    compile_and_test(scheduler=scheduler, tmp_path=tmp_path, a=5, b=1)


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_non_driver_entry_points(tmp_path, testdir, frontend, config):
    """
    Testing of parametrisation functionality with defined entry points/functions, thus not being the default (driver).
    """

    proj = testdir/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend,
        xmods=[tmp_path]
    )

    transformation = ParametriseTransformation(dic2p=dic2p, entry_points=("kernel1", "kernel2"))
    scheduler.process(transformation=transformation)

    subroutine_arguments = {
        "driver": ['a', 'b', 'c', 'd'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['parametrised_a', 'parametrised_b', 'c'],
        "kernel2": ['a_new', 'parametrised_b', 'd'],
        "device1": ['a', 'd', 'x', 'y'],
        "device2": ['a', 'd', 'x'],
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b', 'd'),
        "device1": ('a_new', 'd', 'x', 'k2_tmp'),
        "device2": ('a', 'd', 'x')
    }

    parameter_variables = {
        "driver": [],
        "another_driver": [],
        "kernel1": ['a', 'b'],
        "kernel2": ['b'],
        "device1": ['b'],
        "device2": ['b'],
    }

    check_arguments_and_parameter(scheduler=scheduler, subroutine_arguments=subroutine_arguments,
                                  call_arguments=call_arguments, parameter_variables=parameter_variables)

    compile_and_test(scheduler=scheduler, tmp_path=tmp_path, a=a, b=b)
