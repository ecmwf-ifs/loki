# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree

import pytest
from loki import (
    gettempdir, Scheduler, SchedulerConfig,
    FindNodes,
    CallStatement, Assignment
)
from conftest import available_frontends

from transformations.pool_allocator import TemporariesPoolAllocatorTransformation


@pytest.mark.parametrize('frontend', available_frontends())
def test_pool_allocator_temporaries(frontend):
    fcode_driver = """
subroutine driver(KLON, KLEV, NGPBLK, FIELD1, FIELD2)
    use stack_mod, only: stack
    use kernel_mod, only: kernel
    implicit none
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
    INTEGER, INTENT(IN) :: KLON, KLEV, NGPBLK
    real(kind=jprb), intent(inout) :: field1(klon, ngpblk)
    real(kind=jprb), intent(inout) :: field2(klon, klev, ngpblk)
    integer :: jblk
    integer :: istsz
    REAL(KIND=JPRGB), ALLOCATABLE :: PSTACK(:, :)
    type(stack) :: ylstack

    istsz = (klev+1) * klon
    ALLOCATE(PSTACK(ISTSZ, NGPBLK))
    do jblk=1,ngpblk
        ylstack%l = loc(pstack, (1, jblk))
        ylstack%u = ylstack%l + 8 * istsz
        call KERNEL(klon, klev, field1(:,jblk), field2(:,:,jblk))
    end do
end subroutine driver
    """.strip()
    fcode_kernel = """
module kernel_mod
    implicit none
contains
    subroutine kernel(klon, klev, field1, field2)
        implicit none
        integer, parameter :: jprb = selected_real_kind(13,300)
        integer, intent(in) :: klon, klev
        real(kind=jprb), intent(inout) :: field1(klon)
        real(kind=jprb), intent(inout) :: field2(klon,klev)
        real(kind=jprb) :: tmp1(klon)
        real(kind=jprb) :: tmp2(klon, klev)
        integer :: jk, jl

        do jk=1,klev
            tmp1(jl) = 0.0_jprb
            do jl=1,klon
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
    transformation = TemporariesPoolAllocatorTransformation()
    scheduler.process(transformation=transformation, reverse=True)

    #
    # A few checks on the driver
    #
    driver = scheduler['#driver'].routine

    # Has the stack been added to the call statement?
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('klon', 'klev', 'field1(:,jblk)', 'field2(:,:,jblk)', 'ylstack')

    #
    # A few checks on the kernel
    #
    kernel = scheduler['kernel_mod#kernel'].routine

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

    rmtree(basedir)
