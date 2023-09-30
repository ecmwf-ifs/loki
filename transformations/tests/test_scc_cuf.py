# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest

from conftest import available_frontends
from loki import (
    Scheduler, Subroutine, Dimension, FindNodes, Loop, Assignment,
    CallStatement, Allocation, Deallocation, VariableDeclaration, Import, FindVariables
)
from loki.transform import HoistTemporaryArraysAnalysis, ParametriseTransformation
from loki.expression import symbols as sym
from transformations import SccCufTransformation, HoistTemporaryArraysDeviceAllocatableTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'))


@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk')


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


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
        'routine': [
            {
                'name': 'driver',
                'role': 'driver',
            },
        ]
    }


def check_subroutine_driver(routine, blocking, disable=()):
    # use of "use cudafor"
    imports = [_import.module.lower() for _import in FindNodes(Import).visit(routine.spec)]
    assert "cudafor" in imports
    # device arrays
    # device arrays: declaration
    arrays = [var for var in routine.variables if isinstance(var, sym.Array)]
    device_arrays = [array for array in arrays if "_d" in array.name[-2::]]
    array_map = {}
    for device_array in device_arrays:
        for array in arrays:
            if device_array.name.replace("_d", "") == array.name:
                array_map[device_array] = array
    assert len(arrays)
    assert len(device_arrays)
    _declarations = FindNodes(VariableDeclaration).visit(routine.spec)
    declarations = []
    for _decl in _declarations:
        declarations.extend(_decl.symbols)
    for array in arrays:
        assert array in declarations
        if "_d" in array.name[-2::]:
            assert array.type.allocatable
            assert array.type.device
    # device arrays: allocation and deallocation
    _allocations = FindNodes(Allocation).visit(routine.body)
    allocations = []
    for _allocation in _allocations:
        allocations.extend(_allocation.variables)
    _de_allocations = FindNodes(Deallocation).visit(routine.body)
    de_allocations = []
    for _de_allocation in _de_allocations:
        de_allocations.extend(_de_allocation.variables)
    for device_array in device_arrays:
        assert device_array.name in [_.name for _ in allocations]
        assert device_array.name in [_.name for _ in de_allocations]
    # device arrays: copy device to host and host to device
    assignments = FindNodes(Assignment).visit(routine.body)
    cuda_device_synchronize = sym.InlineCall(
        function=sym.ProcedureSymbol(name="cudaDeviceSynchronize", scope=routine),
        parameters=())
    assert cuda_device_synchronize in [assignment.rhs for assignment in assignments]
    for device_array in device_arrays:
        if array_map[device_array].type.intent == "inout":
            assert Assignment(lhs=device_array.clone(dimensions=None),
                              rhs=array_map[device_array].clone(dimensions=None)) in assignments
            assert Assignment(rhs=device_array.clone(dimensions=None),
                              lhs=array_map[device_array].clone(dimensions=None)) in assignments
        elif array_map[device_array].type.intent == "in":
            assert Assignment(lhs=device_array.clone(dimensions=None),
                              rhs=array_map[device_array].clone(dimensions=None)) in assignments
        elif array_map[device_array].type.intent == "out":
            assert Assignment(rhs=device_array.clone(dimensions=None),
                              lhs=array_map[device_array].clone(dimensions=None)) in assignments
    # definition of block and griddim
    assert "GRIDDIM" in routine.variables
    assert "BLOCKDIM" in routine.variables
    # kernel launch configuration
    calls = [call for call in FindNodes(CallStatement).visit(routine.body) if str(call.name) not in disable]
    for call in calls:
        assert call.chevron[0] == "GRIDDIM"
        assert call.chevron[1] == "BLOCKDIM"
        assert blocking.size in call.arguments


def _check_subroutine_kernel(routine, horizontal, vertical, blocking):
    # use of "use cudafor"
    imports = [_import.module for _import in FindNodes(Import).visit(routine.spec)]
    assert "cudafor" in imports
    # if statement around body
    assert blocking.size in routine.arguments
    # loop structure
    loops = FindNodes(Loop).visit(routine.body)
    loop_variables = [loop.variable for loop in loops]
    assert horizontal.index not in loop_variables
    assert vertical.index in loop_variables
    argument_arrays = [arg for arg in routine.arguments if isinstance(arg, sym.Array)]
    for argument_array in argument_arrays:
        dims = FindVariables().visit(argument_array.dimensions)
        assert blocking.index in dims or blocking.size in dims
    # TODO: assert for local arrays!
    # arrays = [var for var in routine.variables if isinstance(var, sym.Array)]


def check_subroutine_kernel(routine, horizontal, vertical, blocking):
    _check_subroutine_kernel(routine=routine, horizontal=horizontal, vertical=vertical, blocking=blocking)
    assert "ATTRIBUTES(GLOBAL)" in routine.prefix
    assignments = FindNodes(Assignment).visit(routine.body)
    assert "THREADIDX%X" in [_.rhs for _ in assignments]
    assert "BLOCKIDX%Z" in [_.rhs for _ in assignments]


def check_subroutine_device(routine, horizontal, vertical, blocking):
    _check_subroutine_kernel(routine=routine, horizontal=horizontal, vertical=vertical, blocking=blocking)
    assert "ATTRIBUTES(DEVICE)" in routine.prefix
    assert horizontal.index in routine.arguments
    assert blocking.index in routine.arguments


def check_subroutine_elemental_device(routine):
    assert "ATTRIBUTES(DEVICE)" in routine.prefix
    assert "ELEMENTAL" not in routine.prefix


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_simple(frontend, horizontal, vertical, blocking):

    fcode_driver = """
  SUBROUTINE driver(nlon, nz, nb, tot, q, t, z)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    INTEGER, INTENT(IN)   :: tot
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: z(nlon,nz+1,nb)
    INTEGER :: b, start, end, ibl, icend

    start = 1
    end = tot
    do b=1,end,nlon
      ibl = (b-1)/nlon+1
      icend = MIN(nlon,tot-b+1)
      call kernel(start, icend, nlon, nz, q(:,:,b), t(:,:,b), z(:,:,b))
    end do
  END SUBROUTINE driver
"""

    fcode_kernel = """
  SUBROUTINE kernel(start, end, nlon, nz, q, t, z)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    REAL, INTENT(INOUT) :: z(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    DO jk = 2, nz
      DO jl = start, end
        z(jl, jk) = 0.0
      END DO
    END DO

    ! DO JL = START, END
    !   Q(JL, NZ) = Q(JL, NZ) * C
    ! END DO
  END SUBROUTINE kernel
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=[kernel])

    cuf_transform = SccCufTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking
    )

    cuf_transform.apply(driver, role='driver', targets=['kernel'])
    cuf_transform.apply(kernel, role='kernel')

    check_subroutine_driver(routine=driver, blocking=blocking)
    check_subroutine_kernel(routine=kernel, horizontal=horizontal, vertical=vertical, blocking=blocking)


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_parametrise(here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 0, thus including parametrising (array dimension(s))
    """

    proj = here / 'sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuf_transform = SccCufTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='parametrise'
    )
    scheduler.process(transformation=cuf_transform)

    dic2p = {'nz': 137}
    scheduler.process(transformation=ParametriseTransformation(dic2p=dic2p))

    # check for correct CUF transformation
    check_subroutine_driver(routine=scheduler.item_map["driver_mod#driver"].routine, blocking=blocking)
    check_subroutine_kernel(routine=scheduler.item_map["kernel_mod#kernel"].routine, horizontal=horizontal,
                            vertical=vertical, blocking=blocking)
    check_subroutine_device(routine=scheduler.item_map["kernel_mod#device"].routine, horizontal=horizontal,
                            vertical=vertical, blocking=blocking)
    check_subroutine_elemental_device(routine=scheduler.item_map["kernel_mod#elemental_device"].routine)

    # check for parametrised variables
    vars2p = list(dic2p.keys())
    routine_parameters = [var for var in scheduler.item_map["driver_mod#driver"].routine.variables
                          if var.type.parameter]
    assert routine_parameters == vars2p
    routine_parameters = [var for var in scheduler.item_map["kernel_mod#kernel"].routine.variables
                          if var.type.parameter]
    assert routine_parameters == vars2p
    routine_parameters = [var for var in scheduler.item_map["kernel_mod#device"].routine.variables
                          if var.type.parameter]
    assert routine_parameters == vars2p

    # local arrays
    routine = scheduler.item_map["kernel_mod#kernel"].routine
    argument_arrays = [arg for arg in routine.arguments if isinstance(arg, sym.Array)]
    local_arrays = [var for var in routine.variables if isinstance(var, sym.Array) and var not in argument_arrays]
    for local_array in local_arrays:
        assert local_array.type.device
    routine = scheduler.item_map["kernel_mod#device"].routine
    argument_arrays = [arg for arg in routine.arguments if isinstance(arg, sym.Array)]
    local_arrays = [var for var in routine.variables if isinstance(var, sym.Array) and var not in argument_arrays]
    for local_array in local_arrays:
        assert local_array.type.device


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_hoist(here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 1, thus including host side hoisting
    """

    proj = here / 'sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuf_transform = SccCufTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='hoist'
    )
    scheduler.process(transformation=cuf_transform)

    # Transformation: Analysis
    scheduler.process(transformation=HoistTemporaryArraysAnalysis(), reverse=True)
    # Transformation: Synthesis
    scheduler.process(transformation=HoistTemporaryArraysDeviceAllocatableTransformation())

    check_subroutine_driver(routine=scheduler.item_map["driver_mod#driver"].routine, blocking=blocking)
    check_subroutine_kernel(routine=scheduler.item_map["kernel_mod#kernel"].routine, horizontal=horizontal,
                            vertical=vertical, blocking=blocking)
    check_subroutine_device(routine=scheduler.item_map["kernel_mod#device"].routine, horizontal=horizontal,
                            vertical=vertical, blocking=blocking)
    check_subroutine_elemental_device(routine=scheduler.item_map["kernel_mod#elemental_device"].routine)

    # check driver
    for call in FindNodes(CallStatement).visit(scheduler.item_map["driver_mod#driver"].routine.body):
        argnames = [arg.name.lower() for arg in call.arguments]
        assert 'kernel_local_z' in argnames
        assert 'device_local_x' in argnames
    # check kernel
    argnames = [arg.name.lower() for arg in scheduler.item_map["kernel_mod#kernel"].routine.arguments]
    assert 'local_z' in argnames
    assert 'device_local_x' in argnames
    calls = [call for call in FindNodes(CallStatement).visit(scheduler.item_map["kernel_mod#kernel"].routine.body)
             if str(call.name) == "DEVICE"]
    for call in calls:
        assert 'DEVICE_local_x' in call.arguments
    # check device
    assert all(_ in [arg.name for arg in scheduler.item_map["kernel_mod#device"].routine.arguments]
               for _ in ['local_x'])

    # local arrays
    routine = scheduler.item_map["kernel_mod#kernel"].routine
    local_arrays = [routine.variable_map["local_z"]]
    for local_array in local_arrays:
        assert local_array.type.intent == 'inout'
        dims = FindVariables().visit(local_array.dimensions)
        assert horizontal.size in dims
        assert vertical.size in dims
        assert blocking.size in dims
    routine = scheduler.item_map["kernel_mod#device"].routine
    local_arrays = [routine.variable_map["local_x"]]
    for local_array in local_arrays:
        assert local_array.type.intent == 'inout'
        dims = FindVariables().visit(local_array.dimensions)
        assert horizontal.size in dims
        assert vertical.size in dims
        assert blocking.size in dims


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuf_dynamic_memory(here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 2, thus including dynamic memory allocation on the device (for local arrays)
    """

    proj = here / 'sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuf_transform = SccCufTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='dynamic'
    )
    scheduler.process(transformation=cuf_transform)

    check_subroutine_driver(routine=scheduler.item_map["driver_mod#driver"].routine, blocking=blocking)
    check_subroutine_kernel(routine=scheduler.item_map["kernel_mod#kernel"].routine, horizontal=horizontal,
                            vertical=vertical, blocking=blocking)
    check_subroutine_device(routine=scheduler.item_map["kernel_mod#device"].routine, horizontal=horizontal,
                            vertical=vertical, blocking=blocking)
    check_subroutine_elemental_device(routine=scheduler.item_map["kernel_mod#elemental_device"].routine)

    # kernel
    routine = scheduler.item_map["kernel_mod#kernel"].routine
    _allocations = FindNodes(Allocation).visit(routine.body)
    allocations = []
    for allocation in _allocations:
        allocations.extend(allocation.variables)
    assert "local_z" in [_.name for _ in allocations]
    # device
    routine = scheduler.item_map["kernel_mod#device"].routine
    _allocations = FindNodes(Allocation).visit(routine.body)
    allocations = []
    for allocation in _allocations:
        allocations.extend(allocation.variables)
    assert "local_x" in [_.name for _ in allocations]

    # local arrays
    routine = scheduler.item_map["kernel_mod#kernel"].routine
    argument_arrays = [arg for arg in routine.arguments if isinstance(arg, sym.Array)]
    local_arrays = [var for var in routine.variables if isinstance(var, sym.Array) and var not in argument_arrays]
    for local_array in local_arrays:
        assert local_array.type.allocatable
        assert local_array.type.device
        assert len(local_array.dimensions) == 1
    routine = scheduler.item_map["kernel_mod#device"].routine
    argument_arrays = [arg for arg in routine.arguments if isinstance(arg, sym.Array)]
    local_arrays = [var for var in routine.variables if isinstance(var, sym.Array) and var not in argument_arrays]
    for local_array in local_arrays:
        assert local_array.type.allocatable
        assert local_array.type.device
        assert len(local_array.dimensions) == 1
