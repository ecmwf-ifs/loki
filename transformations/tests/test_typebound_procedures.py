# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree
import pytest

from loki import Scheduler, gettempdir, FindNodes, CallStatement
from conftest import available_frontends

from transformations import TypeboundProcedureCallTransformation


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
        'routines': [
            {
                'name': 'driver',
                'role': 'driver',
                'expand': True,
            },
        ]
    }


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_typebound_procedure_calls(frontend, config):
    fcode1 = """
module typebound_procedure_calls_mod
    implicit none

    type my_type
        integer :: val
    contains
        procedure :: reset
        procedure :: add => add_my_type
    end type my_type

    type other_type
        type(my_type) :: arr(3)
    contains
        procedure :: add => add_other_type
        procedure :: total_sum
    end type other_type

contains

    subroutine reset(this)
        class(my_type), intent(inout) :: this
        this%val = 0
    end subroutine reset

    subroutine add_my_type(this, val)
        class(my_type), intent(inout) :: this
        integer, intent(in) :: val
        this%val = this%val + val
    end subroutine add_my_type

    subroutine add_other_type(this, other)
        class(other_type) :: this
        type(other_type) :: other
        integer :: i
        do i=1,3
            call this%arr(i)%add(other%arr(i)%val)
        end do
    end subroutine add_other_type

    function total_sum(this) result(result)
        class(other_type), intent(in) :: this
        integer :: result
        integer :: i
        result = 0
        do i=1,3
            result = result + this%arr(i)%val
        end do
    end function total_sum

end module typebound_procedure_calls_mod
    """.strip()

    fcode2 = """
module other_typebound_procedure_calls_mod
    use typebound_procedure_calls_mod, only: other_type
    implicit none

    type third_type
        type(other_type) :: stuff(2)
    contains
        procedure :: init
        procedure :: print => print_content
    end type third_type

contains

    subroutine init(this)
        class(third_type), intent(inout) :: this
        integer :: i, j
        do i=1,2
            do j=1,3
                call this%stuff(i)%arr(j)%reset()
                call this%stuff(i)%arr(j)%add(i+j)
            end do
        end do
    end subroutine init

    subroutine print_content(this)
        class(third_type), intent(inout) :: this
        call this%stuff(1)%add(this%stuff(2))
        print *, this%stuff(1)%total_sum()
    end subroutine print_content
end module other_typebound_procedure_calls_mod
    """.strip()

    fcode3 = """
subroutine driver
    use other_typebound_procedure_calls_mod, only: third_type
    implicit none
    type(third_type) :: data
    integer :: mysum

    call data%init()
    call data%stuff(1)%arr(1)%add(1)
    mysum = data%stuff(1)%total_sum() + data%stuff(2)%total_sum()
    call data%print
end subroutine driver
    """.strip()

    workdir = gettempdir()/'test_transform_typebound_procedure_calls'
    workdir.mkdir(exist_ok=True)
    (workdir/'typebound_procedure_calls_mod.F90').write_text(fcode1)
    (workdir/'other_typebound_procedure_calls_mod.F90').write_text(fcode2)
    (workdir/'driver.F90').write_text(fcode3)

    scheduler = Scheduler(paths=[workdir], config=config, seed_routines=['driver'], frontend=frontend)
    scheduler.process(transformation=TypeboundProcedureCallTransformation())

    driver = scheduler['#driver'].routine
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 3
    assert calls[0].name == 'init'
    assert calls[0].arguments == ('data',)
    assert calls[1].name == 'add_my_type'
    assert calls[1].arguments == ('data%stuff(1)%arr(1)', '1')
    assert calls[2].name == 'print_content'
    assert calls[2].arguments == ('data',)

    add_other_type = scheduler['typebound_procedure_calls_mod#add_other_type'].routine
    calls = FindNodes(CallStatement).visit(add_other_type.body)
    assert len(calls) == 1
    assert calls[0].name == 'add_my_type'
    assert calls[0].arguments == ('this%arr(i)', 'other%arr(i)%val')

    init = scheduler['other_typebound_procedure_calls_mod#init'].routine
    calls = FindNodes(CallStatement).visit(init.body)
    assert len(calls) == 2
    assert calls[0].name == 'reset'
    assert calls[0].arguments == ('this%stuff(i)%arr(j)',)
    assert calls[1].name == 'add_my_type'
    assert calls[1].arguments == ('this%stuff(i)%arr(j)', 'i + j')

    print_content = scheduler['other_typebound_procedure_calls_mod#print_content'].routine
    calls = FindNodes(CallStatement).visit(print_content.body)
    assert len(calls) == 1
    assert calls[0].name == 'add_other_type'
    assert calls[0].arguments == ('this%stuff(1)', 'this%stuff(2)')

    # TODO: resolve inline calls

    rmtree(workdir)
