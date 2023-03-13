# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree

import pytest
from loki import (
    gettempdir, Scheduler, SchedulerConfig, Dimension,
    FindNodes,
    CallStatement, Assignment
)
from conftest import available_frontends

from transformations.pool_allocator import TemporariesPoolAllocatorTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'))


@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk')


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.mark.parametrize('generate_driver_stack', [False, True])
@pytest.mark.parametrize('frontend', available_frontends())
def test_pool_allocator_temporaries(frontend, generate_driver_stack, horizontal, vertical, blocking):
    fcode_stack_decl = """
    integer :: istsz
    REAL(KIND=JPRGB), ALLOCATABLE :: PSTACK(:, :)
    type(stack) :: ylstack

    istsz = (nz+1) * nlon
    ALLOCATE(PSTACK(ISTSZ, nb))
    """
    fcode_stack_assign = """
        ylstack%l = loc(pstack, (1, b))
        ylstack%u = ylstack%l + 8 * istsz
    """

    fcode_driver = f"""
subroutine driver(NLON, NZ, NGPBLK, FIELD1, FIELD2)
    use stack_mod, only: stack
    use kernel_mod, only: kernel
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    INTEGER, INTENT(IN) :: NLON, NZ, NGPBLK
    real(kind=jprb), intent(inout) :: field1(nlon, nb)
    real(kind=jprb), intent(inout) :: field2(nlon, nz, nb)
    integer :: b
    {fcode_stack_decl if not generate_driver_stack else ''}
    do b=1,nb
        {fcode_stack_assign if not generate_driver_stack else ''}
        call KERNEL(nlon, nz, field1(:,b), field2(:,:,b))
    end do
end subroutine driver
    """.strip()
    fcode_kernel = """
module kernel_mod
    implicit none
contains
    subroutine kernel(start, end, nlon, nz, field1, field2)
        implicit none
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: start, end, nlon, nz
        real(kind=jprb), intent(inout) :: field1(nlon)
        real(kind=jprb), intent(inout) :: field2(nlon,nz)
        real(kind=jprb) :: tmp1(nlon)
        real(kind=jprb) :: tmp2(nlon, nz)
        integer :: jk, jl

        do jk=1,nz
            tmp1(jl) = 0.0_jprb
            do jl=start,end
                tmp2(jl, jk) = field2(jl, jk)
                tmp1(jl) = field2(jl, jk)
            end do
            field1(jl) = tmp1(jl)
        end do
    end subroutine kernel
end module kernel_mod
    """.strip()

    basedir = gettempdir()/'test_pool_allocator_temporaries'
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
        'routine': [{
            'name': 'driver',
            'role': 'driver',
        }]
    }
    scheduler = Scheduler(paths=[basedir], config=SchedulerConfig.from_dict(config), frontend=frontend)
    transformation = TemporariesPoolAllocatorTransformation(
        horizontal=horizontal, vertical=vertical, block_dim=blocking
    )
    scheduler.process(transformation=transformation, reverse=True)

    #
    # A few checks on the driver
    #
    driver = scheduler['#driver'].routine

    # Has the stack been added to the call statement?
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('nlon', 'nz', 'field1(:,b)', 'field2(:,:,b)', 'ylstack')

    #
    # A few checks on the kernel
    #
    kernel = scheduler['kernel_mod#kernel'].routine

    # Has the stack module been imported?
    assert any(import_.module.lower() == 'stack_mod' for import_ in kernel.imports)
    assert 'stack' in kernel.imported_symbols

    # Has the stack been added to the arguments?
    assert 'ydstack' in kernel.arguments

    # Is it being assigned to a local variable?
    assert 'ylstack' in kernel.variables

    # Let's check for the relevant "allocations" happening in the right order
    assign_idx = {}
    for idx, assign in enumerate(FindNodes(Assignment).visit(kernel.body)):
        if assign.lhs == 'ylstack' and assign.rhs == 'ydstack':
            # Local copy of stack status
            assign_idx['stack_assign'] = idx
        elif assign.lhs == 'ip_tmp1' and assign.rhs == 'ylstack%l':
            # Assign Cray pointer for tmp1
            assign_idx['tmp1_ptr_assign'] = idx
        elif assign.lhs == 'ip_tmp2' and assign.rhs == 'ylstack%l':
            # Assign Cray pointer for tmp2
            assign_idx['tmp2_ptr_assign'] = idx
        elif assign.lhs == 'ylstack%l' and 'ylstack%l' in assign.rhs and 'size' in assign.rhs and 'tmp1' in assign.rhs:
            # Stack increment for tmp1
            assign_idx['tmp1_stack_incr'] = idx
        elif assign.lhs == 'ylstack%l' and 'ylstack%l' in assign.rhs and 'tmp2' in assign.rhs and 'tmp2' in assign.rhs:
            # Stack increment for tmp2
            assign_idx['tmp2_stack_incr'] = idx

    expected_assign_in_order = [
        'stack_assign', 'tmp1_ptr_assign', 'tmp1_stack_incr', 'tmp2_ptr_assign', 'tmp2_stack_incr'
    ]
    assert set(expected_assign_in_order) == set(assign_idx.keys())

    for assign1, assign2 in zip(expected_assign_in_order, expected_assign_in_order[1:]):
        assert assign_idx[assign2] > assign_idx[assign1]

    # Check for pointer declarations in generated code
    fcode = kernel.to_fortran()
    assert 'pointer(ip_tmp1, tmp1)' in fcode.lower()
    assert 'pointer(ip_tmp2, tmp2)' in fcode.lower()

    # Check for stack size safegurads in generated code
    assert fcode.lower().count('if (ylstack%l > ylstack%u)') == 2
    assert fcode.lower().count('stop') == 2

    if generate_driver_stack:
        breakpoint()

    rmtree(basedir)
