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

from conftest import available_frontends, jit_compile_lib, clean_test
from loki import FindNodes, Scheduler, Builder, SchedulerConfig, OMNI
from loki import ir, is_iterable, gettempdir, normalize_range_indexing
from loki.transform import (
    HoistVariablesAnalysis, HoistVariablesTransformation,
    HoistTemporaryArraysAnalysis, HoistTemporaryArraysTransformationAllocatable
)


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


def compile_and_test(scheduler, here, a=(5,), frontend="",  test_name=""):
    """
    Compile the source code and call the driver function in order to test the results for correctness.
    """
    assert is_iterable(a) and all(isinstance(_a, int) for _a in a)
    items = [scheduler["transformation_module_hoist#driver"], scheduler["subroutines_mod#kernel1"]]
    for item in items:
        suffix = '.F90'
        item.source.path = here / 'build' / Path(f"{item.source.path.stem}").with_suffix(suffix=suffix)
    libname = f'lib_{test_name}_{frontend}'
    builder = Builder(source_dirs=here/'build', build_dir=here/'build')
    lib = jit_compile_lib([item.source for item in items], path=here/'build', name=libname, builder=builder)
    item = scheduler["transformation_module_hoist#driver"]
    for _a in a:
        parameter_length = 3
        b = np.zeros((_a,), dtype=np.int32, order='F')
        c = np.zeros((_a, parameter_length), dtype=np.int32, order='F')
        lib.Transformation_Module_Hoist.driver(_a, b, c)
        assert (b == 42).all()
        assert (c == 11).all()
    builder.clean()
    for item in items:
        item.source.path.unlink()
    clean_test(filepath=here.parent / item.source.path.with_suffix(suffix).name)


def check_arguments(scheduler, subroutine_arguments, call_arguments, call_kwarguments, include_device_functions=False):
    """
    Check the subroutine and call arguments of each subroutine.
    """
    # driver
    item = scheduler['transformation_module_hoist#driver']
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["driver"]
    for call in FindNodes(ir.CallStatement).visit(item.ir.body):
        if "kernel1" in call.name:
            assert call.arguments == call_arguments["kernel1"]
            assert call.kwarguments == call_kwarguments["kernel1"]
        elif "kernel2" in call.name:
            assert call.arguments == call_arguments["kernel2"]
            assert call.kwarguments == call_kwarguments["kernel2"]
    # another driver
    item = scheduler['transformation_module_hoist#another_driver']
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["another_driver"]
    for call in FindNodes(ir.CallStatement).visit(item.ir.body):
        if "kernel1" in call.name:
            assert call.arguments == call_arguments["kernel1"]
            assert call.kwarguments == call_kwarguments["kernel1"]
    # kernel 1
    item = scheduler['subroutines_mod#kernel1']
    assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["kernel1"]
    # kernel 2
    item = scheduler['subroutines_mod#kernel2']
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
        item = scheduler['subroutines_mod#device1']
        assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["device1"]
        for call in FindNodes(ir.CallStatement).visit(item.ir.body):
            if "device2" in call.name:
                assert call.arguments == call_arguments["device2"]
                assert call.kwarguments == call_kwarguments["device2"]
        # device 2
        item = scheduler['subroutines_mod#device2']
        assert [arg.name for arg in item.ir.arguments] == subroutine_arguments["device2"]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', ['False', 'True'])
def test_hoist(here, frontend, config, as_kwarguments):
    """
    Basic testing of the non-modified Hoist functionality, thus hoisting all (non-parameter) local variables.
    """

    proj = here/'sources/projHoist'
    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

    # check correctness of original source code
    compile_and_test(scheduler=scheduler, here=here, frontend=frontend, a=(5, 10, 100), test_name="source")

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
    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100), frontend=frontend, test_name="all_hoisted")


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', ['False', 'True'])
def test_hoist_disable(here, frontend, config, as_kwarguments):
    """
    Basic testing of the non-modified Hoist functionality excluding/disabling some subroutines,
    thus hoisting all (non-parameter) local variables for the non-disabled subroutines.
    """

    disable = ("device1", "device2")
    config['routines']['kernel2'] = {'role': 'kernel', 'block': disable}
    proj = here/'sources/projHoist'
    scheduler = Scheduler(
        paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend
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
        scheduler=scheduler, here=here, a=(5, 10, 100),
        frontend=frontend, test_name="all_hoisted_disable"
    )


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', ['False', 'True'])
def test_hoist_arrays(here, frontend, config, as_kwarguments):
    """
    Testing hoist functionality for local arrays using the :class:`HoistTemporaryArraysAnalysis` for the *Analysis*
    part.
    """

    proj = here/'sources/projHoist'
    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis())
    # Transformation: Synthesis
    scheduler.process(transformation=HoistVariablesTransformation(as_kwarguments=as_kwarguments))

    # check generated source code
    subroutine_arguments = {
        "driver": ['a', 'b', 'c'],
        "another_driver": ['a', 'b', 'c'],
        "kernel1": ['a', 'b', 'c', 'x', 'y', 'k1_tmp'],
        "kernel2": ['a1', 'b', 'x', 'k2_tmp', 'device2_z', 'device2_d2_tmp'],
        "device1": ['a1', 'b', 'x', 'y', 'device2_z', 'device2_d2_tmp'],
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
        call_arguments["kernel2"] += ('kernel2_x', 'kernel2_k2_tmp', 'device2_z', 'device2_d2_tmp')
        call_arguments["device1"] += ('device2_z', 'device2_d2_tmp')
        call_arguments["device2"] += ('device2_z', 'device2_d2_tmp')

    call_kwarguments = {
        "kernel1": (('x', 'kernel1_x'), ('y', 'kernel1_y'), ('k1_tmp', 'kernel1_k1_tmp')) if as_kwarguments else (),
        "kernel2": (('x', 'kernel2_x'), ('k2_tmp', 'kernel2_k2_tmp'),
            ('device2_z', 'device2_z'), ('device2_d2_tmp', 'device2_d2_tmp')) if as_kwarguments else (),
        "device1": (('device2_z', 'device2_z'), ('device2_d2_tmp', 'device2_d2_tmp')) if as_kwarguments else (),
        "device2": (('z', 'device2_z'), ('d2_tmp', 'device2_d2_tmp')) if as_kwarguments else ()
    }

    check_arguments(scheduler=scheduler, subroutine_arguments=subroutine_arguments, call_arguments=call_arguments,
            call_kwarguments=call_kwarguments)
    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100), frontend=frontend, test_name="hoisted_arrays")


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('as_kwarguments', ['False', 'True'])
def test_hoist_specific_variables(here, frontend, config, as_kwarguments):
    """
    Testing hoist functionality for local arrays with variable ``a`` in the array dimensions using the
    :class:`HoistTemporaryArraysAnalysis` for the *Analysis* part.
    """

    proj = here/'sources/projHoist'
    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

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

    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100), frontend=frontend,
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
def test_hoist_allocatable(here, frontend, config, as_kwarguments):
    """
    Testing hoist functionality for local arrays with variable ``a`` in the array dimensions using the
    :class:`HoistTemporaryArraysAnalysis` for the *Analysis* part **and** a *Synthesis* implementation using declaring
    hoisted arrays as *allocatable*, including allocation and de-allocation using
    :class:`HoistTemporaryArraysTransformationAllocatable`.
    """

    proj = here/'sources/projHoist'
    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver', 'another_driver'], frontend=frontend)

    key = "HoistVariablesAllocatable"
    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('a', 'a1', 'a2'), key=key))
    # Transformation: Synthesis
    scheduler.process(transformation=HoistTemporaryArraysTransformationAllocatable(key=key,
        as_kwarguments=as_kwarguments))

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
    compile_and_test(scheduler=scheduler, here=here, a=(5, 10, 100), frontend=frontend, test_name="allocatable")


@pytest.mark.parametrize('frontend', available_frontends())
def test_hoist_mixed_variable_declarations(frontend, config):

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
    end subroutine kernel
end module kernel_mod
    """.strip()

    basedir = gettempdir()/'test_hoist_mixed_variable_declarations'
    basedir.mkdir(exist_ok=True)
    (basedir/'driver.F90').write_text(fcode_driver)
    (basedir/'kernel_mod.F90').write_text(fcode_kernel)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }

    scheduler = Scheduler(paths=[basedir], config=SchedulerConfig.from_dict(config), frontend=frontend)

    if frontend == OMNI:
        for item in scheduler.items:
            normalize_range_indexing(item.ir)

    scheduler.process(transformation=HoistTemporaryArraysAnalysis(dim_vars=('klev',)))
    scheduler.process(transformation=HoistTemporaryArraysTransformationAllocatable())

    driver_variables = (
        'jprb', 'nlon', 'nz', 'nb', 'b',
        'field1(nlon, nb)', 'field2(nlon, nz, nb)',
        'kernel_tmp2(:,:)', 'kernel_tmp5(:,:,:)'
    )
    kernel_arguments = (
       'start', 'end', 'klon', 'klev', 'nclv',
        'field1(klon)', 'field2(klon,klev)', 'tmp2(klon,klev)', 'tmp5(klon,nclv,klev)'
    )

    # Check hoisting and declaration in driver
    assert scheduler['#driver'].ir.variables == driver_variables
    assert scheduler['kernel_mod#kernel'].ir.arguments == kernel_arguments

    # Check updated call signature
    calls = FindNodes(ir.CallStatement).visit(scheduler['#driver'].ir.body)
    assert len(calls) == 1
    assert calls[0].arguments == (
        '1', 'nlon', 'nlon', 'nz', '2', 'field1(:,b)', 'field2(:,:,b)',
        'kernel_tmp2', 'kernel_tmp5'
    )

    # Check that fgen works
    assert scheduler['kernel_mod#kernel'].source.to_fortran()
