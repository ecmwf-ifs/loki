# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree
import pytest

from loki import (
    OMNI, Sourcefile, Module, Scheduler,
    FindNodes, FindInlineCalls, CallStatement, gettempdir
)
from conftest import available_frontends
from transformations import (
    DerivedTypeArgumentsTransformation,
    TypeboundProcedureCallTransformation
)


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
def test_transform_derived_type_arguments(frontend):
    fcode = f"""
module transform_derived_type_arguments_mod

    implicit none

    type some_derived_type
{'!$loki dimension(n)' if frontend is not OMNI else ''}
        real, allocatable :: a(:)
{'!$loki dimension(m, n)' if frontend is not OMNI else ''}
        real, allocatable :: b(:,:)
    end type some_derived_type

contains

    subroutine caller(z)
        integer, intent(in) :: z
        type(some_derived_type) :: t_io
        type(some_derived_type), allocatable :: t_in(:), t_out(:)
        integer :: m, n
        integer :: i, j

        m = 100
        n = 10

        allocate(t_io%a(n))
        allocate(t_io%b(m, n))

        do j=1,n
            t_io%a(j) = real(i)
            t_io%b(:, j) = real(i)
        end do

        allocate(t_in(z), t_out(z))

        do i=1,z
            allocate(t_in(i)%a(n))
            allocate(t_in(i)%b(m, n))
            allocate(t_out(i)%a(n))
            allocate(t_out(i)%b(m, n))

            do j=1,n
                t_in(i)%a(j) = real(i-1)
                t_in(i)%b(:, j) = real(i-1)
            end do
        end do

        do i=1,z
            call kernel(m, n, t_io%a, t_io%b, t_in(i), t_out(i))
        end do

        deallocate(t_io%a)
        deallocate(t_io%b)

        do i=1,z
            deallocate(t_in(i)%a)
            deallocate(t_in(i)%b)
            deallocate(t_out(i)%a)
            deallocate(t_out(i)%b)
        end do

        deallocate(t_in, t_out)
    end subroutine caller

    subroutine kernel(m, n, P_a, P_b, Q, R)
        integer                , intent(in)    :: m, n
        real, intent(inout)                    :: P_a(n), P_b(m, n)
        type(some_derived_type), intent(in)    :: Q
        type(some_derived_type), intent(out)   :: R
        integer :: j, k

        do j=1,n
            R%a(j) = P_a(j) + Q%a(j)
            do k=1,m
                R%b(k, j) = P_b(k, j) - Q%b(k, j)
            end do
        end do
    end subroutine kernel
end module transform_derived_type_arguments_mod
    """.strip()

    # Parse the Fortran code and enrich call statements
    module = Module.from_source(fcode, frontend=frontend)
    for routine in module.subroutines:
        routine.enrich_calls(routines=module.subroutines)

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for name, role in [('caller', 'driver'), ('kernel', 'kernel')]:
        module[name].apply(transformation, role=role)

    # Make sure derived type arguments are flattened
    call_args = [
        'm', 'n', 't_io%a', 't_io%b', 't_in(i)%a(:)', 't_in(i)%b(:, :)',
        't_out(i)%a(:)', 't_out(i)%b(:, :)'
    ]
    if frontend == OMNI:
        kernel_args = ['m', 'n', 'p_a(1:n)', 'p_b(1:m, 1:n)', 'q_a(:)', 'q_b(:, :)', 'r_a(:)', 'r_b(:, :)']
    else:
        kernel_args = ['m', 'n', 'P_a(n)', 'P_b(m, n)', 'Q_a(n)', 'Q_b(m, n)', 'R_a(n)', 'R_b(m, n)']

    call = FindNodes(CallStatement).visit(module['caller'].ir)[0]
    assert call.name == 'kernel'
    assert [str(v) for v in call.arguments] == call_args
    assert [str(v) in kernel_args for v in module['kernel'].arguments]

    # Make sure rescoping hasn't accidentally overwritten the
    # type information for local variables that have the same name
    # as the shape of another variable
    assert module['caller'].variable_map['m'].type.intent is None
    assert module['caller'].variable_map['n'].type.intent is None


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
    use function_mod, only: some_type
    implicit none

    type third_type
        type(other_type) :: stuff(2)
        type(some_type) :: some
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
        integer :: val
        call this%stuff(1)%add(this%stuff(2))
        val = this%stuff(1)%total_sum()
        print *, val
    end subroutine print_content
end module other_typebound_procedure_calls_mod
    """.strip()

    fcode3 = """
module function_mod
    implicit none
    type some_type
    contains
        procedure :: some_func
    end type some_type
contains
    function some_func(this)
        class(some_type) :: this
        integer some_func
        some_func = 1
    end function some_func
end module function_mod
    """.strip()

    fcode4 = """
subroutine driver
    use other_typebound_procedure_calls_mod, only: third_type
    implicit none
    type(third_type) :: data
    integer :: mysum

    call data%init()
    call data%stuff(1)%arr(1)%add(1)
    mysum = data%stuff(1)%total_sum() + data%stuff(2)%total_sum()
    associate (some => data%some)
        mysum = mysum + some%some_func()
    end associate
    call data%print
end subroutine driver
    """.strip()

    workdir = gettempdir()/'test_transform_typebound_procedure_calls'
    workdir.mkdir(exist_ok=True)
    (workdir/'typebound_procedure_calls_mod.F90').write_text(fcode1)
    (workdir/'other_typebound_procedure_calls_mod.F90').write_text(fcode2)
    (workdir/'function_mod.F90').write_text(fcode3)
    (workdir/'driver.F90').write_text(fcode4)

    # As long as the scheduler isn't able to find the dependency for inline calls,
    # we have to provide it manually as a definition
    function_mod = Sourcefile.from_file(workdir/'function_mod.F90', frontend=frontend)

    scheduler = Scheduler(
        paths=[workdir], config=config, seed_routines=['driver'],
        definitions=function_mod.definitions, frontend=frontend
    )

    transformation = TypeboundProcedureCallTransformation()
    scheduler.process(transformation=transformation)

    # Verify that new dependencies have been identified correctly...
    assert transformation.inline_call_dependencies == {
        '#driver': {'typebound_procedure_calls_mod#total_sum', 'function_mod#some_func'},
        'other_typebound_procedure_calls_mod#print_content': {'typebound_procedure_calls_mod#total_sum'}
    }

    # ...which are not yet in the scheduler
    assert 'typebound_procedure_calls_mod#total_sum' not in scheduler.item_graph.nodes

    # Make sure that we can add them successfully to the scheduler
    scheduler.add_dependencies(transformation.inline_call_dependencies)
    assert 'typebound_procedure_calls_mod#total_sum' in scheduler.item_graph.nodes

    driver = scheduler['#driver'].routine
    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 3
    assert calls[0].name == 'init'
    assert calls[0].arguments == ('data',)
    assert calls[1].name == 'add_my_type'
    assert calls[1].arguments == ('data%stuff(1)%arr(1)', '1')
    assert calls[2].name == 'print_content'
    assert calls[2].arguments == ('data',)

    calls = FindInlineCalls().visit(driver.body)
    assert len(calls) == 3
    assert {str(call).lower() for call in calls} == {
        'total_sum(data%stuff(1))', 'total_sum(data%stuff(2))', 'some_func(some)'
    }

    assert 'init' in driver.imported_symbols
    assert 'add_my_type' in driver.imported_symbols
    assert 'print_content' in driver.imported_symbols
    assert 'total_sum' in driver.imported_symbols

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

    calls = list(FindInlineCalls().visit(print_content.body))
    assert len(calls) == 1
    assert str(calls[0]).lower() == 'total_sum(this%stuff(1))'

    rmtree(workdir)
