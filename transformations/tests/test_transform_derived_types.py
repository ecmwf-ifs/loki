import pytest

from loki import OMNI, Module, FindNodes, CallStatement
from conftest import available_frontends
from transformations import DerivedTypeArgumentsTransformation


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
