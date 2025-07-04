# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A selection of tests for the (generic) hoist variables functionalities.
"""
from pathlib import Path
import pytest
import numpy as np

from loki import (
    Scheduler, SchedulerConfig, is_iterable, FindInlineCalls
)
from loki.jit_build import jit_compile_lib, Builder
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes
from loki.transformations.temporaries.hoist_variables import (
    HoistVariablesAnalysis, HoistVariablesTransformation,
    HoistTemporaryArraysAnalysis, HoistTemporaryArraysTransformationAllocatable
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent.parent/'tests'


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
            'yet_another_driver': {
                'role': 'driver',
                'expand': True,
            },
            'inline_driver': {
                'role': 'driver',
                'expand': True,
            },
        }
    }


def compile_and_test(scheduler, path, a=(5,), frontend="",  test_name="", items=None, inline=False):
    """
    Compile the source code and call the driver function in order to test the results for correctness.
    """
    assert is_iterable(a) and all(isinstance(_a, int) for _a in a)
    path = Path(path)
    if not items:
        items = [scheduler["transformation_module_hoist#driver"], scheduler["subroutines_mod#kernel1"]]
    for item in items:
        suffix = '.F90'
        item.source.path = (path/f"{item.source.path.stem}").with_suffix(suffix=suffix)
    libname = f'lib_{test_name}_{frontend}'
    builder = Builder(source_dirs=path, build_dir=path)
    lib = jit_compile_lib([item.source for item in items], path=path, name=libname, builder=builder)
    item = items[0]
    for _a in a:
        parameter_length = 3
        b = np.zeros((_a,), dtype=np.int32, order='F')
        c = np.zeros((_a, parameter_length), dtype=np.int32, order='F')
        if inline:
            lib.Transformation_Module_Hoist_Inline.inline_driver(_a, b, c)
        else:
            lib.Transformation_Module_Hoist.driver(_a, b, c)
        assert (b == 42).all()
        assert (c == 11).all()
    builder.clean()


def check_arguments(scheduler, subroutine_arguments, call_arguments, call_kwarguments, driver_item=None,
                    driver_name=None, include_device_functions=False, include_another_driver=True,
                    subroutine_mod=None):
    """
    Check the subroutine and call arguments of each subroutine.
    """
    # driver
    if not driver_item:
        driver_item = scheduler['transformation_module_hoist#driver']
    if not driver_name:
        driver_name = "driver"

    assert [arg.name for arg in driver_item.ir.arguments] == subroutine_arguments[driver_name]
    for call in FindNodes(ir.CallStatement).visit(driver_item.ir.body):
        if "kernel1" in call.name:
            assert call.arguments == call_arguments["kernel1"]
            assert call.kwarguments == call_kwarguments["kernel1"]
        elif "kernel2" in call.name:
            assert call.arguments == call_arguments["kernel2"]
            assert call.kwarguments == call_kwarguments["kernel2"]
    # another driver
    if include_another_driver:
        item = scheduler['transformation_module_hoist#another_driver']
        assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["another_driver"]
        for call in FindNodes(ir.CallStatement).visit(item.ir.body):
            if "kernel1" in call.name:
                assert call.arguments == call_arguments["kernel1"]
                assert call.kwarguments == call_kwarguments["kernel1"]
    # kernel 1
    if not subroutine_mod:
        subroutine_mod = 'subroutines_mod'

    item = scheduler[subroutine_mod + '#kernel1']
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["kernel1"]

    for call in FindInlineCalls().visit(item.ir.body):
        if 'func1' in call.name:
            assert call.arguments == call_arguments["func1"]
            assert call.kwarguments == call_kwarguments["func1"]

    # kernel 2
    item = scheduler[subroutine_mod + '#kernel2']
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["kernel2"]
    for call in FindNodes(ir.CallStatement).visit(item.ir.body):
        if "device1" in call.name:
            assert call.arguments == call_arguments["device1"]
            assert call.kwarguments == call_kwarguments["device1"]
        elif "device2" in call.name:
            assert call.arguments == call_arguments["device2"]
            assert call.kwarguments == call_kwarguments["device2"]
    if include_device_functions:
        # device 1
        item = scheduler[subroutine_mod + '#device1']
        assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["device1"]
        for call in FindNodes(ir.CallStatement).visit(item.ir.body):
            if "device2" in call.name:
                assert call.arguments == call_arguments["device2"]
                assert call.kwarguments == call_kwarguments["device2"]
        # device 2
        item = scheduler[subroutine_mod + '#device2']
        assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["device2"]

        for call in FindInlineCalls().visit(item.ir.body):
            if 'init_int' in call.name:
                assert call.arguments == call_arguments["init_int"]
                assert call.kwarguments == call_kwarguments["init_int"]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', [False, True])
def test_hoist(tmp_path, testdir, frontend, config, as_kwarguments):
    """
    Basic testing of the non-modified Hoist functionality, thus hoisting all (non-parameter) local variables.
    """

    proj = testdir/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend, xmods=[tmp_path]
    )

    # check correctness of original source code
    compile_and_test(scheduler=scheduler, path=tmp_path, frontend=frontend, a=(5, 10, 100), test_name="source")

    # Transformation: Analysis
    scheduler.process(transformation=HoistVariablesAnalysis())
    # Transformation: Synthesis
    scheduler.process(transformation=HoistVariablesTransformation(as_kwarguments=as_kwarguments))

    # check generated source code
    subroutine_arguments = {
        "driver": ['a', 'b', 'c'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c', 'x', 'y', 'k1_tmp'],
        "kernel2": ['a1', 'b', 'x', 'y', 'z', 'k2_tmp', 'device1_z', 'device1_d1_tmp', 'device2_z', 'device2_d2_tmp'],
        "device1": ['a1', 'b', 'x', 'y', 'z', 'd1_tmp', 'device2_z', 'device2_d2_tmp'],
        "device2": ['a2', 'b', 'x', 'z', 'd2_tmp'],
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b'),
        "device1": ('a1', 'b', 'x', 'k2_tmp'),
        "device2": ('a1', 'b', 'x')
    }
    if not as_kwarguments:
        call_arguments["kernel1"] += ('kernel1_x', 'kernel1_y', 'kernel1_k1_tmp')
        call_arguments["kernel2"] += ('kernel2_x', 'kernel2_y', 'kernel2_z', 'kernel2_k2_tmp',
                'device1_z', 'device1_d1_tmp', 'device2_z', 'device2_d2_tmp')
        call_arguments["device1"] += ('device1_z', 'device1_d1_tmp', 'device2_z', 'device2_d2_tmp')
        call_arguments["device2"] += ('device2_z', 'device2_d2_tmp')

    call_kwarguments = {
        "kernel1": (('x', 'kernel1_x'), ('y', 'kernel1_y'), ('k1_tmp', 'kernel1_k1_tmp')) if as_kwarguments else (),
        "kernel2": (('x', 'kernel2_x'), ('y', 'kernel2_y'), ('z', 'kernel2_z'), ('k2_tmp', 'kernel2_k2_tmp'),
            ('device1_z', 'device1_z'), ('device1_d1_tmp', 'device1_d1_tmp'),
            ('device2_z', 'device2_z'), ('device2_d2_tmp', 'device2_d2_tmp')) if as_kwarguments else (),
        "device1": (('z', 'device1_z'), ('d1_tmp', 'device1_d1_tmp'), ('device2_z', 'device2_z'),
            ('device2_d2_tmp', 'device2_d2_tmp')) if as_kwarguments else (),
        "device2": (('z', 'device2_z'), ('d2_tmp', 'device2_d2_tmp')) if as_kwarguments else ()
    }

    check_arguments(scheduler=scheduler, subroutine_arguments=subroutine_arguments, call_arguments=call_arguments,
            call_kwarguments=call_kwarguments)
    compile_and_test(scheduler=scheduler, path=tmp_path, a=(5, 10, 100), frontend=frontend, test_name="all_hoisted")


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', [False, True])
def test_hoist_disable(tmp_path, testdir, frontend, config, as_kwarguments):
    """
    Basic testing of the non-modified Hoist functionality excluding/disabling some subroutines,
    thus hoisting all (non-parameter) local variables for the non-disabled subroutines.
    """

    disable = ("device1", "device2")
    config['routines']['kernel2'] = {'role': 'kernel', 'block': disable}
    proj = testdir/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend, xmods=[tmp_path]
    )

    # Transformation: Analysis
    scheduler.process(transformation=HoistVariablesAnalysis())
    # Transformation: Synthesis
    scheduler.process(transformation=HoistVariablesTransformation(as_kwarguments=as_kwarguments))

    # check generated source code
    subroutine_arguments = {
        "driver": ['a', 'b', 'c'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c', 'x', 'y', 'k1_tmp'],
        "kernel2": ['a1', 'b', 'x', 'y', 'z', 'k2_tmp'],
        "device1": ['a1', 'b', 'x', 'y'],
        "device2": ['a2', 'b', 'x'],
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c', 'kernel1_x', 'kernel1_y', 'kernel1_k1_tmp'),
        "kernel2": ('a', 'b', 'kernel2_x', 'kernel2_y', 'kernel2_z', 'kernel2_k2_tmp'),
        "device1": ('a1', 'b', 'x', 'k2_tmp'),
        "device2": ('a1', 'b', 'x')
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b'),
        "device1": ('a1', 'b', 'x', 'k2_tmp'),
        "device2": ('a1', 'b', 'x')
    }
    if not as_kwarguments:
        call_arguments["kernel1"] += ('kernel1_x', 'kernel1_y', 'kernel1_k1_tmp')
        call_arguments["kernel2"] += ('kernel2_x', 'kernel2_y', 'kernel2_z', 'kernel2_k2_tmp')

    call_kwarguments = {
        "kernel1": (('x', 'kernel1_x'), ('y', 'kernel1_y'), ('k1_tmp', 'kernel1_k1_tmp')) if as_kwarguments else (),
        "kernel2": (('x', 'kernel2_x'), ('y', 'kernel2_y'), ('z', 'kernel2_z'),
            ('k2_tmp', 'kernel2_k2_tmp')) if as_kwarguments else (),
        "device1": (),
        "device2": ()
    }

    check_arguments(
        scheduler=scheduler, subroutine_arguments=subroutine_arguments,
        call_arguments=call_arguments, call_kwarguments=call_kwarguments,
        include_device_functions=False
    )
    compile_and_test(
        scheduler=scheduler, path=tmp_path, a=(5, 10, 100),
        frontend=frontend, test_name="all_hoisted_disable"
    )

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', [False, True])
def test_hoist_arrays_inline(tmp_path, testdir, frontend, config, as_kwarguments):
    """
    Testing hoist functionality for local arrays using the :class:`HoistTemporaryArraysAnalysis` for the *Analysis*
    part. The hoisted kernel contains inline function calls.
    """

    proj = testdir/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['inline_driver',], frontend=frontend, xmods=[tmp_path]
    )

    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis())
    # Transformation: Synthesis
    scheduler.process(transformation=HoistVariablesTransformation(as_kwarguments=as_kwarguments))

    # check generated source code
    subroutine_arguments = {
        "inline_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c', 'x', 'y', 'k1_tmp'],
        "kernel2": ['a1', 'b', 'x', 'k2_tmp', 'device2_z', 'init_int_tmp0'],
        "device1": ['a1', 'b', 'x', 'y', 'device2_z', 'init_int_tmp0'],
        "device2": ['a2', 'b', 'x', 'z', 'init_int_tmp0'],
        "init_int": ['a2', 'tmp0'],
        "func1": ['a']
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b'),
        "device1": ('a1', 'b', 'x', 'k2_tmp'),
        "device2": ('a1', 'b', 'x'),
        "init_int": ('a2',),
        "func1": ('a',)
    }
    if not as_kwarguments:
        call_arguments["kernel1"] += ('kernel1_x', 'kernel1_y', 'kernel1_k1_tmp')
        call_arguments["kernel2"] += ('kernel2_x', 'kernel2_k2_tmp', 'device2_z', 'init_int_tmp0')
        call_arguments["device1"] += ('device2_z', 'init_int_tmp0')
        call_arguments["device2"] += ('device2_z', 'init_int_tmp0')
        call_arguments["init_int"] += ('init_int_tmp0',)

    call_kwarguments = {
        "kernel1": (('x', 'kernel1_x'), ('y', 'kernel1_y'), ('k1_tmp', 'kernel1_k1_tmp')) if as_kwarguments else (),
        "kernel2": (('x', 'kernel2_x'), ('k2_tmp', 'kernel2_k2_tmp'),
            ('device2_z', 'device2_z'), ('init_int_tmp0', 'init_int_tmp0')) if as_kwarguments else (),
        "device1": (('device2_z', 'device2_z'), ('init_int_tmp0', 'init_int_tmp0')) if as_kwarguments else (),
        "device2": (('z', 'device2_z'), ('init_int_tmp0', 'init_int_tmp0')) if as_kwarguments else (),
        "init_int": (('tmp0', 'init_int_tmp0'),) if as_kwarguments else (),
        "func1": ()
    }

    check_arguments(scheduler=scheduler, subroutine_arguments=subroutine_arguments, call_arguments=call_arguments,
           call_kwarguments=call_kwarguments, driver_item=scheduler['transformation_module_hoist_inline#inline_driver'],
           driver_name='inline_driver', include_another_driver=False, subroutine_mod='subroutines_inline_mod',
           include_device_functions=True)
    compile_and_test(scheduler=scheduler, path=tmp_path, a=(5, 10, 100), frontend=frontend,
                     test_name="hoisted_arrays_inline",
                     items=[scheduler["transformation_module_hoist_inline#inline_driver"],
                            scheduler["subroutines_inline_mod#kernel1"]], inline=True)

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', [False, True])
def test_hoist_arrays(tmp_path, testdir, frontend, config, as_kwarguments):
    """
    Testing hoist functionality for local arrays using the :class:`HoistTemporaryArraysAnalysis` for the *Analysis*
    part.
    """

    proj = testdir/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend, xmods=[tmp_path]
    )

    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis())
    # Transformation: Synthesis
    scheduler.process(transformation=HoistVariablesTransformation(as_kwarguments=as_kwarguments))

    # check generated source code
    subroutine_arguments = {
        "driver": ['a', 'b', 'c'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c', 'x', 'y', 'k1_tmp'],
        "kernel2": ['a1', 'b', 'x', 'k2_tmp', 'device2_z'],
        "device1": ['a1', 'b', 'x', 'y', 'device2_z'],
        "device2": ['a2', 'b', 'x', 'z'],
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b'),
        "device1": ('a1', 'b', 'x', 'k2_tmp'),
        "device2": ('a1', 'b', 'x')
    }
    if not as_kwarguments:
        call_arguments["kernel1"] += ('kernel1_x', 'kernel1_y', 'kernel1_k1_tmp')
        call_arguments["kernel2"] += ('kernel2_x', 'kernel2_k2_tmp', 'device2_z')
        call_arguments["device1"] += ('device2_z',)
        call_arguments["device2"] += ('device2_z',)

    call_kwarguments = {
        "kernel1": (('x', 'kernel1_x'), ('y', 'kernel1_y'), ('k1_tmp', 'kernel1_k1_tmp')) if as_kwarguments else (),
        "kernel2": (('x', 'kernel2_x'), ('k2_tmp', 'kernel2_k2_tmp'),
            ('device2_z', 'device2_z')) if as_kwarguments else (),
        "device1": (('device2_z', 'device2_z'), ) if as_kwarguments else (),
        "device2": (('z', 'device2_z'), ) if as_kwarguments else ()
    }

    check_arguments(scheduler=scheduler, subroutine_arguments=subroutine_arguments, call_arguments=call_arguments,
            call_kwarguments=call_kwarguments)
    compile_and_test(scheduler=scheduler, path=tmp_path, a=(5, 10, 100), frontend=frontend, test_name="hoisted_arrays")


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', [False, True])
def test_hoist_specific_variables(tmp_path, testdir, frontend, config, as_kwarguments):
    """
    Testing hoist functionality for local arrays with variable ``a`` in the array dimensions using the
    :class:`HoistTemporaryArraysAnalysis` for the *Analysis* part.
    """

    proj = testdir/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend, xmods=[tmp_path]
    )

    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('a', 'a1', 'a2')))
    # Transformation: Synthesis
    scheduler.process(transformation=HoistVariablesTransformation(as_kwarguments=as_kwarguments))

    # check generated source code
    subroutine_arguments = {
        "driver": ['a', 'b', 'c'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c', 'x', 'y', 'k1_tmp'],
        "kernel2": ['a1', 'b', 'x', 'k2_tmp', 'device2_z'],
        "device1": ['a1', 'b', 'x', 'y', 'device2_z'],
        "device2": ['a2', 'b', 'x', 'z'],
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b'),
        "device1": ('a1', 'b', 'x', 'k2_tmp'),
        "device2": ('a1', 'b', 'x')
    }
    if not as_kwarguments:
        call_arguments["kernel1"] += ('kernel1_x', 'kernel1_y', 'kernel1_k1_tmp')
        call_arguments["kernel2"] += ('kernel2_x', 'kernel2_k2_tmp', 'device2_z')
        call_arguments["device1"] += ('device2_z',)
        call_arguments["device2"] += ('device2_z',)

    call_kwarguments = {
        "kernel1": (('x', 'kernel1_x'), ('y', 'kernel1_y'), ('k1_tmp', 'kernel1_k1_tmp')) if as_kwarguments else (),
        "kernel2": (('x', 'kernel2_x'), ('k2_tmp', 'kernel2_k2_tmp'),
            ('device2_z', 'device2_z')) if as_kwarguments else (),
        "device1": (('device2_z', 'device2_z'),) if as_kwarguments else (),
        "device2": (('z', 'device2_z'),) if as_kwarguments else ()
    }

    check_arguments(scheduler=scheduler, subroutine_arguments=subroutine_arguments, call_arguments=call_arguments,
            call_kwarguments=call_kwarguments)

    compile_and_test(scheduler=scheduler, path=tmp_path, a=(5, 10, 100), frontend=frontend,
                     test_name="hoisted_specific_arrays")


def check_variable_declaration(item, key):
    declarations = [_.symbols[0].name for _ in FindNodes(ir.VariableDeclaration).visit(item.ir.spec)]
    allocations = [_.variables[0].name for _ in FindNodes(ir.Allocation).visit(item.ir.body)]
    de_allocations = [_.variables[0].name for _ in FindNodes(ir.Deallocation).visit(item.ir.body)]
    assert allocations
    assert de_allocations
    to_hoist_vars = [var.name for var in item.trafo_data[key]["to_hoist"]]
    assert all(_ in declarations for _ in to_hoist_vars)
    assert all(_ in to_hoist_vars for _ in allocations)
    assert all(_ in to_hoist_vars for _ in de_allocations)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', [False, True])
def test_hoist_allocatable(tmp_path, testdir, frontend, config, as_kwarguments):
    """
    Testing hoist functionality for local arrays with variable ``a`` in the array dimensions using the
    :class:`HoistTemporaryArraysAnalysis` for the *Analysis* part **and** a *Synthesis* implementation using declaring
    hoisted arrays as *allocatable*, including allocation and de-allocation using
    :class:`HoistTemporaryArraysTransformationAllocatable`.
    """

    proj = testdir/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend, xmods=[tmp_path]
    )

    key = "HoistVariablesTransformation"
    # Transformation: Analysis
    scheduler.process(
        transformation=HoistTemporaryArraysAnalysis(dim_vars=('a', 'a1', 'a2'))
    )
    # Transformation: Synthesis
    scheduler.process(
        transformation=HoistTemporaryArraysTransformationAllocatable(
            as_kwarguments=as_kwarguments
        )
    )

    # check generated source code
    for item in scheduler.items:
        if "driver" in item.name and "another" not in item.name:
            check_variable_declaration(item, key)
        elif "another_driver" in item.name:
            check_variable_declaration(item, key)

    subroutine_arguments = {
        "driver": ['a', 'b', 'c'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c', 'x', 'y', 'k1_tmp'],
        "kernel2": ['a1', 'b', 'x', 'k2_tmp', 'device2_z'],
        "device1": ['a1', 'b', 'x', 'y', 'device2_z'],
        "device2": ['a2', 'b', 'x', 'z'],
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c', 'kernel1_x', 'kernel1_y', 'kernel1_k1_tmp'),
        "kernel2": ('a', 'b', 'kernel2_x', 'kernel2_k2_tmp', 'device2_z'),
        "device1": ('a1', 'b', 'x', 'k2_tmp', 'device2_z'),
        "device2": ('a1', 'b', 'x', 'device2_z')
    }

    call_arguments = {
        "kernel1": ('a', 'b', 'c'),
        "kernel2": ('a', 'b'),
        "device1": ('a1', 'b', 'x', 'k2_tmp'),
        "device2": ('a1', 'b', 'x')
    }
    if not as_kwarguments:
        call_arguments["kernel1"] += ('kernel1_x', 'kernel1_y', 'kernel1_k1_tmp')
        call_arguments["kernel2"] += ('kernel2_x', 'kernel2_k2_tmp', 'device2_z')
        call_arguments["device1"] += ('device2_z',)
        call_arguments["device2"] += ('device2_z',)

    call_kwarguments = {
        "kernel1": (('x', 'kernel1_x'), ('y', 'kernel1_y'), ('k1_tmp', 'kernel1_k1_tmp')) if as_kwarguments else (),
        "kernel2": (('x', 'kernel2_x'), ('k2_tmp', 'kernel2_k2_tmp'),
            ('device2_z', 'device2_z')) if as_kwarguments else (),
        "device1": (('device2_z', 'device2_z'),) if as_kwarguments else (),
        "device2": (('z', 'device2_z'),) if as_kwarguments else ()
    }

    check_arguments(scheduler=scheduler, subroutine_arguments=subroutine_arguments, call_arguments=call_arguments,
            call_kwarguments=call_kwarguments)
    compile_and_test(scheduler=scheduler, path=tmp_path, a=(5, 10, 100), frontend=frontend, test_name="allocatable")


@pytest.mark.parametrize('frontend', available_frontends())
def test_hoist_mixed_variable_declarations(tmp_path, frontend, config):

    fcode_driver = """
subroutine driver(NLON, NZ, NB, FIELD1, FIELD2)
    use kernel_mod, only: kernel
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    INTEGER, INTENT(IN) :: NLON, NZ, NB
    integer :: b
    real(kind=jprb), intent(inout) :: field1(nlon, nb)
    real(kind=jprb), intent(inout) :: field2(nlon, nz, nb)
    do b=1,nb
        call KERNEL(1, nlon, nlon, nz, 2, field1(:,b), field2(:,:,b))
    end do
end subroutine driver
    """.strip()
    fcode_kernel = """
module kernel_mod
    implicit none
contains
    subroutine kernel(start, end, klon, klev, nclv, field1, field2)
        use, intrinsic :: iso_c_binding, only : c_size_t
        implicit none
        interface
           subroutine another_kernel(klev)
               integer, intent(in) :: klev
           end subroutine another_kernel
        end interface
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: nclv
        integer, intent(in) :: start, end, klon, klev
        real(kind=jprb), intent(inout) :: field1(klon)
        real(kind=jprb), intent(inout) :: field2(klon,klev)
        real(kind=jprb) :: tmp1(klon)
        real(kind=jprb) :: tmp2(klon, klev), tmp3(nclv)
        real(kind=jprb) :: tmp4(2), tmp5(klon, nclv, klev)
        integer :: jk, jl, jm

        do jk=1,klev
            tmp1(jl) = 0.0_jprb
            do jl=start,end
                tmp2(jl, jk) = field2(jl, jk)
                tmp1(jl) = field2(jl, jk)
            end do
            field1(jl) = tmp1(jl)
        end do

        do jm=1,nclv
           tmp3(jm) = 0._jprb
           do jl=start,end
             tmp5(jl, jm, :) = field1(jl)
           enddo
        enddo

        call another_kernel(klev)
    end subroutine kernel
end module kernel_mod
    """.strip()
    fcode_mod = """
module size_mod
   implicit none
   integer :: n
end module size_mod
""".strip()
    fcode_another_kernel = """
subroutine another_kernel(klev)
    use size_mod, only : n
    implicit none
    integer, intent(in) :: klev
    real :: another_tmp(klev,n)
end subroutine another_kernel
""".strip()

    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel_mod.F90').write_text(fcode_kernel)
    (tmp_path/'size_mod.F90').write_text(fcode_mod)
    (tmp_path/'another_kernel.F90').write_text(fcode_another_kernel)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': True
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config), frontend=frontend, xmods=[tmp_path]
    )
    scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('klev',)))
    scheduler.process(transformation=HoistTemporaryArraysTransformationAllocatable())

    driver_variables = (
        'jprb', 'nlon', 'nz', 'nb', 'b',
        'field1(nlon, nb)', 'field2(nlon, nz, nb)',
        'kernel_tmp2(:,:)', 'kernel_tmp5(:,:,:)', 'another_kernel_another_tmp(:,:)'
    )
    kernel_arguments = (
       'start', 'end', 'klon', 'klev', 'nclv',
        'field1(klon)', 'field2(klon,klev)', 'tmp2(klon,klev)', 'tmp5(klon,nclv,klev)',
        'another_kernel_another_tmp(klev,n)'
    )

    # Check hoisting and declaration in driver
    assert scheduler['#driver'].ir.variables == driver_variables
    assert scheduler['kernel_mod#kernel'].ir.arguments == kernel_arguments

    # Check updated call signature
    calls = FindNodes(ir.CallStatement).visit(scheduler['#driver'].ir.body)
    assert len(calls) == 1
    assert calls[0].arguments == (
        '1', 'nlon', 'nlon', 'nz', '2', 'field1(:,b)', 'field2(:,:,b)',
        'kernel_tmp2', 'kernel_tmp5', 'another_kernel_another_tmp'
    )

    # Check that fgen works
    assert scheduler['kernel_mod#kernel'].source.to_fortran()

    # Check that imports were updated
    imports = FindNodes(ir.Import).visit(scheduler['kernel_mod#kernel'].ir.spec)
    assert len(imports) == 2
    assert 'n' in scheduler['kernel_mod#kernel'].ir.imported_symbols
    assert imports[0].module.lower() == 'size_mod'

    imports = FindNodes(ir.Import).visit(scheduler['#driver'].ir.spec)
    assert len(imports) == 2
    assert 'n' in scheduler['#driver'].ir.imported_symbols
    assert imports[0].module.lower() == 'size_mod'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('remap_dimensions', (False, True))
def test_hoist_dim_mapping(tmp_path, frontend, config, remap_dimensions):

    fcode_driver = """
subroutine driver(NLON, NB, FIELD1)
    use kernel_mod, only: kernel
    implicit none
    INTEGER, INTENT(IN) :: NLON, NB
    integer :: b
    integer, intent(inout) :: field1(nlon, nb)
    integer :: local_nlon
    local_nlon = nlon
    do b=1,nb
        call KERNEL(local_nlon, field1(:,b))
    end do
end subroutine driver
    """.strip()
    fcode_kernel = """
module kernel_mod
    implicit none
contains
    subroutine kernel(klon, field1)
        implicit none
        integer, intent(in) :: klon
        integer, intent(inout) :: field1(klon)
        integer :: tmp1(klon)
        integer :: jl

        do jl=1,klon
            tmp1(jl) = 0
            field1(jl) = tmp1(jl)
        end do

    end subroutine kernel
end module kernel_mod
    """.strip()

    (tmp_path/'driver.F90').write_text(fcode_driver)
    (tmp_path/'kernel_mod.F90').write_text(fcode_kernel)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': True
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }

    scheduler = Scheduler(
        paths=[tmp_path], config=SchedulerConfig.from_dict(config), frontend=frontend, xmods=[tmp_path]
    )

    scheduler.process(transformation=HoistTemporaryArraysAnalysis())
    scheduler.process(transformation=HoistVariablesTransformation(remap_dimensions=remap_dimensions))

    driver_var_map = scheduler['#driver'].ir.variable_map
    assert 'kernel_tmp1' in driver_var_map
    if remap_dimensions:
        assert driver_var_map['kernel_tmp1'].dimensions == ('nlon',)
    else:
        assert driver_var_map['kernel_tmp1'].dimensions == ('local_nlon',)


@pytest.mark.parametrize('frontend', available_frontends())
def test_hoist_dim_alias(tmp_path, testdir, frontend, config):
    """
    Test that temporaries declared with an aliased dimension aren't repeated.
    """

    proj = testdir/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['yet_another_driver'], frontend=frontend, xmods=[tmp_path]
    )

    scheduler.process(transformation=HoistTemporaryArraysAnalysis())
    scheduler.process(transformation=HoistVariablesTransformation())

    kernel3 = scheduler['subroutines_mod#kernel3'].ir
    device3 = scheduler['subroutines_mod#device3'].ir
    driver = scheduler['transformation_module_hoist#yet_another_driver'].ir

    # check temporary was hoisted
    assert 'x' in device3._dummies
    assert 'device3_x' in kernel3._dummies
    assert len(kernel3.arguments) == 3
    device3_temp = kernel3.variable_map['device3_x']
    assert 'a1' in device3_temp.shape

    # check call signatures for device3 were updated
    calls = FindNodes(ir.CallStatement).visit(kernel3.body)
    assert len(calls) == 2
    assert 'device3_x' in calls[0].arguments
    assert 'device3_x' in calls[1].arguments

    # check driver layer
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls[0].arguments) == 3
    assert 'device3_x' in calls[0].arguments
