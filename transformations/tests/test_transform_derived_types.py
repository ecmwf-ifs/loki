# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import OMNI, Module, FindNodes, CallStatement
from conftest import available_frontends
from transformations import DerivedTypeArgumentsAnalysis, DerivedTypeArgumentsTransformation


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_analysis(frontend):
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

    module = Module.from_source(fcode, frontend=frontend)
    kernel = module['kernel']
    kernel.apply(DerivedTypeArgumentsAnalysis(), role='kernel')

    assert kernel.symbol_map['Q'].type.expansion_names == ('a', 'b')
    assert kernel.symbol_map['R'].type.expansion_names == ('a', 'b')


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

    module = Module.from_source(fcode, frontend=frontend)

    call_tree = [('caller', 'driver'), ('kernel', 'kernel')]

    # Apply analysis
    analysis = DerivedTypeArgumentsAnalysis()
    for name, role in reversed(call_tree):
        module[name].apply(analysis, role=role)

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for name, role in call_tree:
        module[name].apply(transformation, role=role)

    # Make sure derived type arguments are flattened
    call_args = (
        'm', 'n', 't_io%a', 't_io%b', 't_in(i)%a(:)', 't_in(i)%b(:, :)',
        't_out(i)%a(:)', 't_out(i)%b(:, :)'
    )
    if frontend == OMNI:
        kernel_args = ['m', 'n', 'p_a(1:n)', 'p_b(1:m, 1:n)', 'q_a(:)', 'q_b(:, :)', 'r_a(:)', 'r_b(:, :)']
    else:
        kernel_args = ['m', 'n', 'P_a(n)', 'P_b(m, n)', 'Q_a(n)', 'Q_b(m, n)', 'R_a(n)', 'R_b(m, n)']

    call = FindNodes(CallStatement).visit(module['caller'].ir)[0]
    assert call.name == 'kernel'
    assert call.arguments == call_args
    assert [v in kernel_args for v in module['kernel'].arguments]

    # Make sure rescoping hasn't accidentally overwritten the
    # type information for local variables that have the same name
    # as the shape of another variable
    assert module['caller'].variable_map['m'].type.intent is None
    assert module['caller'].variable_map['n'].type.intent is None


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_nested(frontend):
    """
    Verify correct behaviour of the derived type argument flattening when
    used in nested call trees. There it is mandatory to traverse the tree from
    the leaf upwards to make sure every use of derived type members is seen by
    the calling subroutine.
    """
    fcode = """
module transform_derived_type_arguments_nested
    implicit none

    type some_type
        real, allocatable :: a(:), b(:), c(:)
    end type some_type

contains

    subroutine caller(n, obj)
        integer, intent(in) :: n
        type(some_type), intent(inout) :: obj

        call setup_obj(obj, n)
    end subroutine caller

    subroutine setup_obj(obj, n)
        type(some_type), intent(inout) :: obj
        integer, intent(in) :: n

        call deallocate_obj(obj)

        allocate(obj%a(n))
        allocate(obj%b(n))
    end subroutine setup_obj

    subroutine deallocate_obj(obj)
        type(some_type), intent(inout) :: obj

        if(allocated(obj%a)) deallocate(obj%a)
        if(allocated(obj%b)) deallocate(obj%b)
        if(allocated(obj%c)) deallocate(obj%c)
    end subroutine deallocate_obj

end module transform_derived_type_arguments_nested
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    orig_args = {
        'caller': ('n', 'obj'),
        'setup_obj': ('obj', 'n'),
        'deallocate_obj': ('obj',),
    }

    transformed_args = {
        'caller': ('n', 'obj'),
        'setup_obj': ('obj_a(:)', 'obj_b(:)', 'obj_c(:)', 'n'),
        'deallocate_obj': ('obj_a(:)', 'obj_b(:)', 'obj_c(:)'),
    }

    for routine in module.subroutines:
        assert routine.arguments == orig_args[routine.name.lower()]

    call_tree = [('caller', 'driver'), ('setup_obj', 'kernel'), ('deallocate_obj', 'kernel')]

    analysis = DerivedTypeArgumentsAnalysis()
    for name, role in reversed(call_tree):
        module[name].apply(analysis, role=role)

    transformation = DerivedTypeArgumentsTransformation()
    for name, role in call_tree:
        module[name].apply(transformation, role=role)

    for routine in module.subroutines:
        assert routine.arguments == transformed_args[routine.name.lower()]
