"""
A selection of tests for the generic hoist functionality.
"""
from pathlib import Path
import pytest
import numpy as np

from conftest import available_frontends, jit_compile, clean_test
from loki import FindVariables
from loki.expression import symbols as sym
from loki import ir, as_tuple, is_iterable
from loki import Scheduler
from loki.transform import HoistAnalysis, HoistSynthesis

write_transformation_output = False


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
        'routines': []
    }


def compile_and_test(scheduler, here, a=(5,)):
    """
    Compile the source code and call the driver function in order to test the results for correctness.
    """
    assert is_iterable(a) and all(isinstance(_a, int) for _a in a)
    for item in scheduler.items:
        if "driver" in item.name:
            suffix = '.F90'
            module = jit_compile(item.source, filepath=item.source.path.with_suffix(suffix).name, objname=None)
            for _a in a:
                parameter_length = 3
                b = np.zeros((_a,), dtype=np.int32, order='F')
                c = np.zeros((_a, parameter_length), dtype=np.int32, order='F')
                module.Transformation_Module_Hoist.driver(_a, b, c)
                assert (b == 42).all()
                assert (c == 11).all()
            clean_test(filepath=here.parent / item.source.path.with_suffix(suffix).name)


@pytest.mark.parametrize('frontend', available_frontends())
def test_hoist(here, frontend, config):
    """
    Basic testing of the non-modified Hoist functionality, thus hoisting all (non-parameter) local variables.
    """

    proj = here/'sources/projHoist'

    config['routine'] = [
        {
            'name': 'driver',
            'role': 'driver',
            'expand': True,
        },
    ]

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))

    # Analysis
    scheduler.process(transformation=HoistAnalysis(), reverse=True)
    # Synthesis
    scheduler.process(transformation=HoistSynthesis())

    for item in scheduler.items:
        if write_transformation_output:
            suffix = '.out.F90'
            sourcefile = item.source
            sourcefile.write(path=sourcefile.path.with_suffix(suffix).name)
        if "driver" in item.name:
            assert len(item.routine.arguments) == 3
        elif "kernel1" in item.name:
            assert len(item.routine.arguments) == 6
        elif "kernel2" in item.name:
            assert len(item.routine.arguments) == 10
        elif "device1" in item.name:
            assert len(item.routine.arguments) == 8
        elif "device2" in item.name:
            assert len(item.routine.arguments) == 5

    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))


@pytest.mark.parametrize('frontend', available_frontends())
def test_hoist_find_arrays(here, frontend, config):
    """
    Testing hoist functionality with modified *Analysis* part using a `find_variables` method only searching for arrays.
    """

    class HoistAnalysisFindArrays(HoistAnalysis):

        @staticmethod
        def find_variables(routine):
            return [var for var in routine.variables if var not in routine.arguments and isinstance(var, sym.Array)]

    proj = here/'sources/projHoist'

    config['routine'] = [
        {
            'name': 'driver',
            'role': 'driver',
            'expand': True,
        },
    ]

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    # compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))

    # Analysis
    scheduler.process(transformation=HoistAnalysisFindArrays(), reverse=True)
    # Synthesis
    scheduler.process(transformation=HoistSynthesis())

    for item in scheduler.items:
        if write_transformation_output:
            suffix = '.find_arrays.out.F90'
            sourcefile = item.source
            sourcefile.write(path=sourcefile.path.with_suffix(suffix).name)
        if "driver" in item.name:
            assert len(item.routine.arguments) == 3
        elif "kernel1" in item.name:
            assert len(item.routine.arguments) == 6
        elif "kernel2" in item.name:
            assert len(item.routine.arguments) == 6
        elif "device1" in item.name:
            assert len(item.routine.arguments) == 6
        elif "device2" in item.name:
            assert len(item.routine.arguments) == 5

    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))


@pytest.mark.parametrize('frontend', available_frontends())
def test_hoist_find_variables(here, frontend, config):
    """
    Testing hoist functionality with modified *Analysis* part using a `find_variables` method only searching for arrays
    with variable `a` within the shape/dimensions.
    """

    class HoistAnalysisFindVariables(HoistAnalysis):

        @staticmethod
        def find_variables(routine):
            return [var for var in routine.variables if var not in routine.arguments and isinstance(var, sym.Array)
                    and "a" in FindVariables().visit(var.dimensions)]

    proj = here/'sources/projHoist'

    config['routine'] = [
        {
            'name': 'driver',
            'role': 'driver',
            'expand': True,
        },
    ]

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    # compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))

    # Analysis
    scheduler.process(transformation=HoistAnalysisFindVariables(), reverse=True)
    # Synthesis
    scheduler.process(transformation=HoistSynthesis())

    for item in scheduler.items:
        if write_transformation_output:
            suffix = '.find_variables.out.F90'
            sourcefile = item.source
            sourcefile.write(path=sourcefile.path.with_suffix(suffix).name)
        if "driver" in item.name:
            assert len(item.routine.arguments) == 3
        elif "kernel1" in item.name:
            assert len(item.routine.arguments) == 6
        elif "kernel2" in item.name:
            assert len(item.routine.arguments) == 5
        elif "device1" in item.name:
            assert len(item.routine.arguments) == 5
        elif "device2" in item.name:
            assert len(item.routine.arguments) == 4

    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))


@pytest.mark.parametrize('frontend', available_frontends())
def test_hoist_allocatable(here, frontend, config):
    """
    Testing hoist functionality with modified *Analysis* part using a `find_variables` method only searching for arrays
    with variable `a` within the shape/dimensions **and** modified *Synthesis* part using `driver_variable_declaration`
    implementation declaring hoisted arrays as *allocatable*, including allocation and de-allocation.
    """

    class HoistAnalysisFindVariables(HoistAnalysis):

        @staticmethod
        def find_variables(routine):
            return [var for var in routine.variables if var not in routine.arguments and isinstance(var, sym.Array)
                    and "a" in FindVariables().visit(var.dimensions)]

    class HoistSynthesisAllocatable(HoistSynthesis):

        @staticmethod
        def driver_variable_declaration(routine, var):
            dimensions = [sym.IntrinsicLiteral(":")] * (len(var.dimensions))
            routine.spec.append(ir.VariableDeclaration((var.clone(dimensions=as_tuple(dimensions),
                                                                  type=var.type.clone(allocatable=True)),)))
            routine.body.prepend(ir.Allocation((var.clone(),)))
            routine.body.append(ir.Deallocation((var.clone(dimensions=None),)))

    proj = here/'sources/projHoist'

    config['routine'] = [
        {
            'name': 'driver',
            'role': 'driver',
            'expand': True,
        },
    ]

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    # compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))

    # Analysis
    scheduler.process(transformation=HoistAnalysisFindVariables(), reverse=True)
    # Synthesis
    scheduler.process(transformation=HoistSynthesisAllocatable())

    for item in scheduler.items:
        if write_transformation_output:
            suffix = '.allocatable.F90'
            sourcefile = item.source
            sourcefile.write(path=sourcefile.path.with_suffix(suffix).name)
        if "driver" in item.name:
            assert len(item.routine.arguments) == 3
        elif "kernel1" in item.name:
            assert len(item.routine.arguments) == 6
        elif "kernel2" in item.name:
            assert len(item.routine.arguments) == 5
        elif "device1" in item.name:
            assert len(item.routine.arguments) == 5
        elif "device2" in item.name:
            assert len(item.routine.arguments) == 4

    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100))
