"""
A selection of tests for the parametrisation functionality.
"""
from pathlib import Path
import pytest
import numpy as np

from conftest import available_frontends, jit_compile, clean_test
from loki import ir, fgen, Scheduler, FindNodes, OMNI
from loki.transform import ParametriseTransformation
import loki.expression.symbols as sym


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


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


def compile_and_test(scheduler, here, a=5, b=1, test_name=""):
    """
    Compile the source code and call the driver function in order to test the results for correctness.
    """
    for item in scheduler.items:
        if "driver" in item.name:
            suffix = '.F90'
            if test_name != "":
                item.source.path = Path(f"{item.source.path.stem}_{test_name}")
            module = jit_compile(item.source, filepath=item.source.path.with_suffix(suffix).name, objname=None)
            c = np.zeros((a, b), dtype=np.int32, order='F')
            d = np.zeros((b,), dtype=np.int32, order='F')
            module.Parametrise.driver(a, b, c, d)
            assert (c == 11).all()
            assert (d == 42).all()
            clean_test(filepath=here.parent / item.source.path.with_suffix(suffix).name)


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
def test_parametrise_source(here, frontend, config):
    """
    Test the actual source code without any transformations applied.
    """

    proj = here/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

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

    compile_and_test(scheduler=scheduler, here=here, a=a, b=b, test_name="original_source")


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_simple(here, frontend, config):
    """
    Basic testing of parametrisation functionality.
    """

    proj = here/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

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

    compile_and_test(scheduler=scheduler, here=here, a=a, b=b, test_name="parametrised")


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_simple_replace_by_value(here, frontend, config):
    """
    Basic testing of parametrisation functionality including replacing of the variables with the actual values.
    """

    proj = here/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

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

    if frontend == OMNI:
        routine_spec_str = fgen(scheduler['parametrise#driver'].ir.spec)
        assert f'c(1:{a}, 1:{b})' in routine_spec_str
        assert f'd(1:{b})' in routine_spec_str
        routine_spec_str = fgen(scheduler['parametrise#another_driver'].ir.spec)
        assert f'c(1:{a}, 1:{b})' in routine_spec_str
        assert f'x(1:{a})' in routine_spec_str
        routine_spec_str = fgen(scheduler['parametrise#kernel1'].ir.spec)
        assert f'c(1:{a}, 1:{b})' in routine_spec_str
        assert f'x(1:{a})' in routine_spec_str
        assert f'y(1:{a}, 1:{b})' in routine_spec_str
        assert f'k1_tmp(1:{a}, 1:{b})' in routine_spec_str
        routine_spec_str = fgen(scheduler['parametrise#kernel2'].ir.spec)
        assert f'd(1:{b})' in routine_spec_str
        assert f'x(1:{a})' in routine_spec_str
        assert f'k2_tmp(1:{a}, 1:{a})' in routine_spec_str
        routine_spec_str = fgen(scheduler['parametrise#device1'].ir.spec)
        assert f'd(1:{b})' in routine_spec_str
        assert f'x(1:{a})' in routine_spec_str
        assert f'y(1:{a}, 1:{a})' in routine_spec_str
        routine_spec_str = fgen(scheduler['parametrise#device2'].ir.spec)
        assert f'd(1:{b})' in routine_spec_str
        assert f'x(1:{a})' in routine_spec_str
        assert f'z(1:{b})' in routine_spec_str
        assert f'd2_tmp(1:{b})' in routine_spec_str
    else:
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

    compile_and_test(scheduler=scheduler, here=here, a=a, b=b, test_name="replaced")


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_modified_callback(here, frontend, config):
    """
    Testing of the parametrisation functionality with modified callbacks for failed sanity checks.
    """

    proj = here/'sources/projParametrise'

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

    for i, abort_callback in enumerate(abort_callbacks):
        scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'],
                              frontend=frontend)
        transformation = ParametriseTransformation(dic2p=dic2p, abort_callback=abort_callback)
        scheduler.process(transformation=transformation)

        check_arguments_and_parameter(scheduler=scheduler, subroutine_arguments=subroutine_arguments,
                                      call_arguments=call_arguments, parameter_variables=parameter_variables)

        compile_and_test(scheduler=scheduler, here=here, a=a, b=b, test_name=f"callback_{i+1}")


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_modified_callback_wrong_input(here, frontend, config):
    """
    Testing of the parametrisation functionality with modified callback for failed sanity checks including test of
    a failed sanity check.
    """

    proj = here/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    def only_warn(**kwargs):
        msg = kwargs.get("msg")
        abort = (ir.Intrinsic(text=f'print *, "This is just a warning: {msg}"'),)
        return abort

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)
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

    test_name = "parametrised_callback_wrong_input"
    a = a + 1
    b = b + 1
    for item in scheduler.items:
        if "driver" in item.name:
            suffix = '.F90'
            if test_name != "":
                item.source.path = Path(f"{item.source.path.stem}_{test_name}")
            module = jit_compile(item.source, filepath=item.source.path.with_suffix(suffix).name, objname=None)
            c = np.zeros((a, b), dtype=np.int32, order='F')
            d = np.zeros((b,), dtype=np.int32, order='F')
            module.Parametrise.driver(a, b, c, d)
            assert not (c == 11).all()
            assert not (d == 42).all()
            clean_test(filepath=here.parent / item.source.path.with_suffix(suffix).name)


@pytest.mark.parametrize('frontend', available_frontends())
def test_parametrise_non_driver_entry_points(here, frontend, config):
    """
    Testing of parametrisation functionality with defined entry points/functions, thus not being the default (driver).
    """

    proj = here/'sources/projParametrise'

    dic2p = {'a': 12, 'b': 11}
    a = dic2p['a']
    b = dic2p['b']

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

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

    compile_and_test(scheduler=scheduler, here=here, a=a, b=b, test_name="parametrised_entry_points")
