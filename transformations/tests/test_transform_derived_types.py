# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from itertools import zip_longest
import pytest

from loki import (
    OMNI, Sourcefile, FindNodes, CallStatement, SubroutineItem, as_tuple,
    ProcedureDeclaration, Scalar, Array, FindVariables, Assignment, fgen, BasicType
)
from conftest import available_frontends
from transformations import DerivedTypeArgumentsTransformation #DerivedTypeArgumentsExpansionAnalysis, DerivedTypeArgumentsExpansionTransformation


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
{'!$loki dimension(m, n)' if frontend is not OMNI else ''}
        real, allocatable :: c(:,:)
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
                R%b(k, j) = P_b(k, j) - Q%b(k, j) - Q%c(k, j)
            end do
        end do
    end subroutine kernel
end module transform_derived_type_arguments_mod
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    item = SubroutineItem(name='transform_derived_type_arguments_mod#kernel', source=source)
    source.apply(DerivedTypeArgumentsTransformation(), role='kernel', item=item)

    # Make sure the trafo data contains the right information
    assert item.trafo_data[DerivedTypeArgumentsTransformation._key] == {
        'expansion_map': {
            'q': ('q%a', 'q%b', 'q%c'),
            'r': ('r%a', 'r%b'),
        },
        'orig_argnames': ('m', 'n', 'p_a', 'p_b', 'q', 'r')
    }

    # Make sure the trafo data is actual variable nodes with proper type information
    # but not attached to any scope
    for members in item.trafo_data[DerivedTypeArgumentsTransformation._key]['expansion_map'].values():
        for member in members:
            assert isinstance(member, (Scalar, Array))
            assert member.scope is None
            assert member.type.dtype != BasicType.DEFERRED


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_expansion(frontend):
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
            t_io%a(j) = real(j)
            t_io%b(:, j) = real(j)
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

    source = Sourcefile.from_source(fcode, frontend=frontend)

    call_tree = [
        SubroutineItem(name='transform_derived_type_arguments_mod#caller', source=source, config={'role': 'driver'}),
        SubroutineItem(name='transform_derived_type_arguments_mod#kernel', source=source, config={'role': 'kernel'}),
    ]

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for item, successor in reversed(list(zip_longest(call_tree, call_tree[1:]))):
        transformation.apply(item.source, role=item.role, item=item, successors=as_tuple(successor))

    # Make sure derived type arguments are flattened
    call_args = (
        'm', 'n', 't_io%a', 't_io%b', 't_in(i)%a', 't_in(i)%b',
        't_out(i)%a', 't_out(i)%b'
    )
    if frontend == OMNI:
        kernel_args = ('m', 'n', 'p_a(1:n)', 'p_b(1:m, 1:n)', 'q_a(:)', 'q_b(:, :)', 'r_a(:)', 'r_b(:, :)')
    else:
        kernel_args = ('m', 'n', 'P_a(n)', 'P_b(m, n)', 'Q_a(:)', 'Q_b(:, :)', 'R_a(:)', 'R_b(:, :)')

    call = FindNodes(CallStatement).visit(source['caller'].ir)[0]
    assert call.name == 'kernel'
    assert call.arguments == call_args
    assert source['kernel'].arguments == kernel_args
    assert all(v.type.intent for v in source['kernel'].arguments)

    # Make sure rescoping hasn't accidentally overwritten the
    # type information for local variables that have the same name
    # as the shape of another variable
    assert source['caller'].variable_map['m'].type.intent is None
    assert source['caller'].variable_map['n'].type.intent is None


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_multilevel(frontend):
    """
    Verify correct behaviour of the derived type argument flattening when
    used in multi-level call trees. There it is mandatory to traverse the tree from
    the leaf upwards to make sure every use of derived type members is seen by
    the calling subroutine.
    """
    fcode = """
module transform_derived_type_arguments_multilevel
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

end module transform_derived_type_arguments_multilevel
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)

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

    for routine in source.subroutines:
        assert routine.arguments == orig_args[routine.name.lower()]

    call_tree = [
        SubroutineItem(
            name='transform_derived_type_arguments_multilevel#caller',
            source=source, config={'role': 'driver'}
        ),
        SubroutineItem(
            name='transform_derived_type_arguments_multilevel#setup_obj',
            source=source, config={'role': 'kernel'}
        ),
        SubroutineItem(
            name='transform_derived_type_arguments_multilevel#deallocate_obj',
            source=source, config={'role': 'kernel'}
        ),
    ]

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for item, successor in reversed(list(zip_longest(call_tree, call_tree[1:]))):
        transformation.apply(item.source, role=item.role, item=item, successors=as_tuple(successor))

    for item in call_tree:
        if item.role == 'driver':
            assert not item.trafo_data[transformation._key]
        else:
            assert item.trafo_data[transformation._key]['orig_argnames'] == orig_args[item.routine.name.lower()]

    for routine in source.subroutines:
        assert routine.arguments == transformed_args[routine.name.lower()]
        assert all(a.type.intent for a in routine.arguments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_expansion_nested(frontend):
    fcode = f"""
module transform_derived_type_arguments_mod

    implicit none

    type some_derived_type
{'!$loki dimension(n)' if frontend is not OMNI else ''}
        real, allocatable :: a(:)
{'!$loki dimension(m, n)' if frontend is not OMNI else ''}
        real, allocatable :: b(:,:)
    end type some_derived_type

    type bucket_type
        type(some_derived_type) :: a
        type(some_derived_type) :: b(5)
    end type bucket_type

contains

    subroutine caller(z)
        integer, intent(in) :: z
        type(bucket_type) :: t_io
        type(bucket_type), allocatable :: t_in(:), t_out(:)
        integer :: m, n
        integer :: i, j, k

        m = 100
        n = 10

        call setup(t_io%a, m, n)
        call init(t_io%a, m, n)
        do k=1,5
            call setup(t_io%b(k), m, n)
            call init(t_io%b(k), m, n)
        end do

        allocate(t_in(z), t_out(z))

        do i=1,z
            call setup(t_in(i)%a, m, n)
            call setup(t_out(i)%a, m, n)

            do j=1,n
                t_in(i)%a%a(j) = real(i-1)
                t_in(i)%a%b(:, j) = real(i-1)
            end do

            do k=1,5
                call setup(t_in(i)%b(k), m, n)
                call setup(t_out(i)%b(k), m, n)

                do j=1,n
                    t_in(i)%b(k)%a(j) = real(i-1)
                    t_in(i)%b(k)%b(:, j) = real(i-1)
                end do
            end do
        end do

        do i=1,z
            call layer(m, n, t_io%a, t_io%b, t_in(i), t_out(i))
        end do

        do i=1,z
            call teardown(t_in(i)%a)
            call teardown(t_out(i)%a)

            do k=1,5
                call teardown(t_in(i)%b(k))
                call teardown(t_out(i)%b(k))
            end do
        end do

        deallocate(t_in)
        deallocate(t_out)

        do k=1,5
            call teardown(t_io%b(k))
            call teardown(t_io%b(k))
        end do
        call teardown(t_io%a)
    end subroutine caller

    subroutine setup(t, m, n)
        type(some_derived_type), intent(inout) :: t
        integer, intent(in) :: m, n

        allocate(t%a(n))
        allocate(t%b(m, n))
    end subroutine setup

    subroutine teardown(t)
        type(some_derived_type), intent(inout) :: t
        deallocate(t%a)
        deallocate(t%b)
    end subroutine teardown

    subroutine init(t, m, n)
        type(some_derived_type), intent(inout) :: t
        integer, intent(in) :: m, n
        integer j

        do j=1,n
            t%a(j) = real(j)
            t%b(:, j) = real(j)
        end do
    end subroutine init

    subroutine layer(m, n, P_a, P_b, Q, R)
        integer                , intent(in) :: m, n
        type(some_derived_type), intent(in) :: P_a, P_b(5)
        type(bucket_type), intent(in)       :: Q
        type(bucket_type), intent(out)      :: R
        integer :: k

        call kernel(m, n, P_a%a, P_a%b, Q%a, R%a)
        do k=1,5
            call kernel(m, n, P_b(k)%a, P_b(k)%b, Q%b(k), R%b(k))
        end do
    end subroutine layer

    subroutine kernel(m, n, P_a, P_b, Q, R)
        integer                , intent(in)    :: m, n
        real, intent(in)                       :: P_a(n), P_b(m, n)
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

    source = Sourcefile.from_source(fcode, frontend=frontend)

    items = {
        'caller': SubroutineItem(
            name='transform_derived_type_arguments_mod#caller', source=source, config={'role': 'driver'}
        ),
        'setup': SubroutineItem(
            name='transform_derived_type_arguments_mod#setup', source=source, config={'role': 'kernel'}
        ),
        'init': SubroutineItem(
            name='transform_derived_type_arguments_mod#init', source=source, config={'role': 'kernel'}
        ),
        'layer': SubroutineItem(
            name='transform_derived_type_arguments_mod#layer', source=source, config={'role': 'kernel'}
        ),
        'kernel': SubroutineItem(
            name='transform_derived_type_arguments_mod#kernel', source=source, config={'role': 'kernel'}
        ),
        'teardown': SubroutineItem(
            name='transform_derived_type_arguments_mod#teardown', source=source, config={'role': 'kernel'}
        ),
    }

    call_tree = [
        ('caller', ['setup', 'init', 'layer', 'teardown']),
        ('setup', []),
        ('init', []),
        ('layer', ['kernel']),
        ('kernel', []),
        ('teardown', [])
    ]

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for name, successors in reversed(call_tree):
        item = items[name]
        children = [items[c] for c in successors]
        transformation.apply(item.source, role=item.role, item=item, successors=as_tuple(children))

    key = DerivedTypeArgumentsTransformation._key

    # Check analysis result in kernel
    assert key in items['kernel'].trafo_data
    assert items['kernel'].trafo_data[key]['expansion_map'] == {
        'q': ('q%a', 'q%b'),
        'r': ('r%a', 'r%b'),
    }
    assert items['kernel'].trafo_data[key]['orig_argnames'] == (
        'm', 'n', 'p_a', 'p_b', 'q', 'r'
    )

    # Check analysis result in layer
    assert key in items['layer'].trafo_data
    assert items['layer'].trafo_data[key]['expansion_map'] == {
        'p_a': ('p_a%a', 'p_a%b'),
        'q': ('q%a%a', 'q%a%b', 'q%b'),
        'r': ('r%a%a', 'r%a%b', 'r%b')
    }
    assert items['layer'].trafo_data[key]['orig_argnames'] == (
        'm', 'n', 'p_a', 'p_b', 'q', 'r'
    )

    # Check arguments of setup
    assert items['setup'].routine.arguments == (
        't_a(:)', 't_b(:, :)', 'm', 'n'
    )

    # Check arguments of init
    assert items['init'].routine.arguments == (
        't_a(:)', 't_b(:, :)', 'm', 'n'
    )

    # Check arguments of teardown
    assert items['teardown'].routine.arguments == (
        't_a(:)', 't_b(:, :)'
    )

    # Check arguments of kernel
    if frontend == OMNI:
        assert items['kernel'].routine.arguments == (
            'm', 'n', 'p_a(1:n)', 'p_b(1:m, 1:n)', 'q_a(:)', 'q_b(:, :)', 'r_a(:)', 'r_b(:, :)'
        )
    else:
        assert items['kernel'].routine.arguments == (
            'm', 'n', 'P_a(n)', 'P_b(m, n)', 'Q_a(:)', 'Q_b(:, :)', 'R_a(:)', 'R_b(:, :)'
        )

    # Check call arguments in layer
    calls = FindNodes(CallStatement).visit(items['layer'].routine.ir)
    assert len(calls) == 2

    assert calls[0].arguments == (
        'm', 'n', 'p_a_a', 'p_a_b', 'q_a_a', 'q_a_b', 'r_a_a', 'r_a_b'
    )
    assert calls[1].arguments == (
        'm', 'n', 'p_b(k)%a', 'p_b(k)%b', 'q_b(k)%a', 'q_b(k)%b', 'r_b(k)%a', 'r_b(k)%b'
    )

    # Check arguments of layer
    if frontend == OMNI:
        assert items['layer'].routine.arguments == (
            'm', 'n', 'p_a_a(:)', 'p_a_b(:, :)', 'p_b(1:5)', 'q_a_a(:)', 'q_a_b(:, :)', 'q_b(:)',
            'r_a_a(:)', 'r_a_b(:, :)', 'r_b(:)'
        )
    else:
        assert items['layer'].routine.arguments == (
            'm', 'n', 'p_a_a(:)', 'p_a_b(:, :)', 'p_b(5)', 'q_a_a(:)', 'q_a_b(:, :)', 'q_b(:)',
            'r_a_a(:)', 'r_a_b(:, :)', 'r_b(:)'
        )

    # Check call arguments in caller
    for call in FindNodes(CallStatement).visit(items['caller'].routine.body):
        if call.name in ('setup', 'init'):
            assert len(call.arguments) == 4
        elif call.name == 'teardown':
            assert len(call.arguments) == 2
        elif call.name == 'layer':
            assert call.arguments == (
                'm', 'n', 't_io%a%a', 't_io%a%b', 't_io%b',
                't_in(i)%a%a', 't_in(i)%a%b', 't_in(i)%b',
                't_out(i)%a%a', 't_out(i)%a%b', 't_out(i)%b'
            )
        else:
            pytest.xfail('Unknown call name')


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_typebound_proc(frontend):
    fcode = f"""
module transform_derived_type_arguments_mod

    implicit none

    type some_derived_type
{'!$loki dimension(n)' if frontend is not OMNI else ''}
        real, allocatable :: a(:)
{'!$loki dimension(m, n)' if frontend is not OMNI else ''}
        real, allocatable :: b(:,:)
{'!$loki dimension(m, n)' if frontend is not OMNI else ''}
        real, allocatable :: c(:,:)
    contains
        procedure, pass :: kernel_a
        procedure :: kernel_b_c => kernel
        procedure, pass(this) :: reduce
    end type some_derived_type

contains

    subroutine kernel_a(this, out, n)
        class(some_derived_type), intent(inout) :: this
        real, allocatable, intent(inout)        :: out(:)
        integer                , intent(in)     :: n
        integer :: j

        do j=1,n
            out(j) = this%a(j) + 1.
        end do
    end subroutine kernel_a

    subroutine kernel(this, other, m, n)
        class(some_derived_type), intent(in)   :: this
        type(some_derived_type), intent(inout) :: other
        integer                , intent(in)    :: m, n
        integer :: j, k

        do j=1,n
            do k=1,m
                other%b(k, j) = 1.e3 - this%b(k, j) - this%c(k, j)
            end do
        end do
    end subroutine kernel

    function reduce(start, this) result(val)
        real, intent(in) :: start
        class(some_derived_type), intent(in) :: this
        real :: val
        val = sum(this%a + sum(this%b + this%c, 1))
    end function reduce
end module transform_derived_type_arguments_mod
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    kernel_a = SubroutineItem(name='transform_derived_type_arguments_mod#kernel_a', source=source)
    kernel = SubroutineItem(name='transform_derived_type_arguments_mod#kernel', source=source)
    reduce = SubroutineItem(name='transform_derived_type_arguments_mod#reduce', source=source)

    # Check procedure bindings before the transformation
    typedef = source['some_derived_type']
    assert typedef.variable_map['kernel_a'].type.pass_attr is True
    assert typedef.variable_map['kernel_b_c'].type.pass_attr in (None, True)
    assert typedef.variable_map['reduce'].type.pass_attr == 'this'
    proc_decls = [decl for decl in typedef.declarations if isinstance(decl, ProcedureDeclaration)]
    assert len(proc_decls) == 3
    assert proc_decls[0].symbols[0] == 'kernel_a'
    assert proc_decls[1].symbols[0] == 'kernel_b_c'
    assert proc_decls[2].symbols[0] == 'reduce'

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation(key='some_key')
    source.apply(transformation, role='kernel', item=kernel_a)
    source.apply(transformation, role='kernel', item=kernel)
    source.apply(transformation, role='kernel', item=reduce)

    # Check analysis outcome
    assert 'some_key' in kernel_a.trafo_data
    assert 'some_key' in kernel.trafo_data
    assert 'some_key' in reduce.trafo_data

    assert kernel_a.trafo_data['some_key']['expansion_map'] == {
        'this': ('this%a',),
    }
    assert kernel_a.trafo_data['some_key']['orig_argnames'] == ('this', 'out', 'n')
    assert kernel.trafo_data['some_key']['expansion_map'] == {
        'this': ('this%b', 'this%c'),
        'other': ('other%b',)
    }
    assert kernel.trafo_data['some_key']['orig_argnames'] == ('this', 'other', 'm', 'n')
    assert reduce.trafo_data['some_key']['expansion_map'] == {
        'this': ('this%a', 'this%b', 'this%c'),
    }
    assert reduce.trafo_data['some_key']['orig_argnames'] == ('start', 'this')

    # Check transformation outcome
    assert kernel_a.routine.arguments == ('this_a(:)', 'out(:)', 'n')
    assert kernel.routine.arguments == ('this_b(:, :)', 'this_c(:, :)', 'other_b(:, :)', 'm', 'n')

    # Check updated procedure bindings
    typedef = source['some_derived_type']
    assert typedef.variable_map['kernel_a'].type.pass_attr is False
    assert typedef.variable_map['kernel_b_c'].type.pass_attr is False
    assert typedef.variable_map['reduce'].type.pass_attr is False
    proc_decls = [decl for decl in typedef.declarations if isinstance(decl, ProcedureDeclaration)]
    assert len(proc_decls) == 3
    assert proc_decls[0].symbols[0] == 'kernel_a'
    assert proc_decls[1].symbols[0] == 'kernel_b_c'
    assert proc_decls[2].symbols[0] == 'reduce'

    # Check output of fgen
    assert source.to_fortran().count(' NOPASS ') == 3


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_elemental(frontend):
    fcode_header = """
module header_mod
    implicit none
    integer, parameter :: jpia = selected_int_kind(9)
end module header_mod
    """.strip()

    fcode_mod = """
module elemental_mod
    implicit none
    integer, parameter :: jpim = selected_int_kind(9)
    type some_type
        integer :: a
        integer, allocatable :: b(:)
        integer, pointer :: vals(:)
    end type some_type
contains
    elemental subroutine callee(o, idx, v)
        use header_mod, only: jpia
        type(some_type), intent(in) :: o
        integer, intent(in) :: idx
        integer, intent(out) :: v
        v = o%a + O%b(idx) + o%vALs(MIN(iDX, 1_jpim)) + o%vals(min(idx, 1_JPIM)) + o%b(idx + 1_jpia) + o%b(idx + 1_jpia)
        v = v + o%vals(o%a + 1)
    end subroutine callee
end module elemental_mod
    """.strip()

    fcode_caller = """
subroutine caller(obj, arr)
    use elemental_mod, only: some_type, callee
    implicit none
    type(some_type), intent(in) :: obj
    integer, intent(inout) :: arr(4)
    integer :: idx = (/2, 3, 4/)

    call callee(obj, 1, arr(1))
    call callee(obj, idx, arr(idx))
end subroutine caller
    """.strip()

    source_header = Sourcefile.from_source(fcode_header, frontend=frontend)
    source_mod = Sourcefile.from_source(
        fcode_mod, frontend=frontend,
        definitions=source_header.definitions
    )
    source_caller = Sourcefile.from_source(
        fcode_caller, frontend=frontend,
        definitions=source_mod.definitions + source_header.definitions
    )
    caller = SubroutineItem(name='#caller', source=source_caller)
    callee = SubroutineItem(name='elemental_mod#callee', source=source_mod)

    assert 'jpia' not in caller.qualified_imports

    transformation = DerivedTypeArgumentsTransformation()
    source_mod.apply(transformation, item=callee, role='kernel', successors=())
    source_caller.apply(transformation, item=caller, role='driver', successors=(callee,))

    # Check analysis outcome
    assert caller.trafo_data[transformation._key] == {}
    assert callee.trafo_data[transformation._key]['expansion_map'] == {
        'o': ('o%a', 'o%b(idx + 1)', 'o%b(idx)', 'o%vals(min(idx, 1))', 'o%vals(o%a + 1)')
    }
    assert callee.trafo_data[transformation._key]['orig_argnames'] == ('o', 'idx', 'v')

    # Check arguments
    if frontend == OMNI:
        assert caller.routine.arguments == ('obj', 'arr(1:4)')
    else:
        assert caller.routine.arguments == ('obj', 'arr(4)')

    # Note: The local naming of arguments with running counters
    # is not stable because of random hash sorting
    # in the transformation. This is not a problem because the order
    # of the
    argument_name_permutations = [
        ('o_a', 'o_b_1', 'o_b_2', 'o_vals_1', 'o_vals_2', 'idx', 'v'),
        ('o_a', 'o_b_2', 'o_b_1', 'o_vals_1', 'o_vals_2', 'idx', 'v'),
        ('o_a', 'o_b_1', 'o_b_2', 'o_vals_2', 'o_vals_1', 'idx', 'v'),
        ('o_a', 'o_b_2', 'o_b_1', 'o_vals_2', 'o_vals_1', 'idx', 'v'),
    ]

    assert callee.routine.arguments in argument_name_permutations
    permutation_index = argument_name_permutations.index(callee.routine.arguments)

    for arg in callee.routine.arguments:
        assert isinstance(arg, Scalar)
        for attr in ('allocatable', 'target', 'pointer', 'shape'):
            # Check attributes are removed from declarations
            assert getattr(arg.type, attr) in (None, False)

    # Check there are only scalar variables in the callee
    for var in FindVariables().visit(callee.routine.body):
        assert isinstance(var, Scalar)

    # Check that substitution happened in expressions
    assignments = FindNodes(Assignment).visit(callee.routine.body)
    assert len(assignments) == 2
    assert assignments[0].rhs == [
        'o_a + o_b_2 + o_vals_1 + o_vals_1 + o_b_1 + o_b_1',
        'o_a + o_b_1 + o_vals_1 + o_vals_1 + o_b_2 + o_b_2',
        'o_a + o_b_2 + o_vals_2 + o_vals_2 + o_b_1 + o_b_1',
        'o_a + o_b_1 + o_vals_2 + o_vals_2 + o_b_2 + o_b_2',
    ][permutation_index]

    assert assignments[1].rhs == [
        'v + o_vals_2', 'v + o_vals_2',
        'v + o_vals_1', 'v + o_vals_1'
    ][permutation_index]

    # Check that calls on caller side have been updated
    calls = FindNodes(CallStatement).visit(caller.routine.body)
    assert calls[0].arguments == (
        'obj%a', 'obj%b(1 + 1)', 'obj%b(1)', 'obj%vals(min(1, 1))', 'obj%vals(obj%a + 1)', '1', 'arr(1)'
    )
    assert calls[1].arguments == (
        'obj%a', 'obj%b(idx + 1)', 'obj%b(idx)', 'obj%vals(min(idx, 1))', 'obj%vals(obj%a + 1)', 'idx', 'arr(idx)'
    )

    # Check that the global symbols (parameters) have been carried over accordingly
    if frontend == OMNI:
        # OMNI inlines parameters
        assert fgen(calls[0]) == (
            'CALL callee(obj%a, obj%b(1 + 1_4), obj%b(1), obj%vals(min(1, 1_4)), '
            'obj%vals(obj%a + 1), 1, arr(1))'
        )
        assert fgen(calls[1]) == (
            'CALL callee(obj%a, obj%b(idx + 1_4), obj%b(idx), obj%vals(min(idx, 1_4)), '
            'obj%vals(obj%a + 1), idx, arr(idx))'
        )
    else:
        assert fgen(calls[0]).lower() == (
            'call callee(obj%a, obj%b(1 + 1_jpia), obj%b(1), obj%vals(min(1, 1_jpim)), '
            'obj%vals(obj%a + 1), 1, arr(1))'
        )
        assert fgen(calls[1]).lower() == (
            'call callee(obj%a, obj%b(idx + 1_jpia), obj%b(idx), obj%vals(min(idx, 1_jpim)), '
            'obj%vals(obj%a + 1), idx, arr(idx))'
        )

        assert 'jpia' in caller.routine.imported_symbols
        assert 'jpim' in caller.routine.variables
        assert 'jpia' in caller.qualified_imports


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_import_rename(frontend):
    fcode1 = """
module some_mod
    implicit none
    type some_type
        integer, allocatable :: a(:)
    end type some_type
contains
    subroutine some_routine(t)
        type(some_type), intent(inout) :: t
        t%a = 1.
    end subroutine some_routine
end module some_mod
    """.strip()
    fcode2 = """
subroutine some_routine(t)
    use some_mod, only: some_type, routine => some_routine
    type(some_type), intent(inout) :: t
    call routine(t)
end subroutine some_routine
    """.strip()

    source1 = Sourcefile.from_source(fcode1, frontend=frontend)
    source2 = Sourcefile.from_source(fcode2, frontend=frontend, definitions=source1.definitions)

    callee = SubroutineItem(name='some_mod#some_routine', source=source1)
    caller = SubroutineItem(name='#some_routine', source=source2)

    transformation = DerivedTypeArgumentsTransformation()
    source1.apply(transformation, item=callee, role='kernel', successors=())
    source2.apply(transformation, item=caller, role='kernel', successors=(callee,))

    assert caller.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%a',),
    }
    assert callee.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%a',),
    }

    assert caller.routine.arguments == ('t_a(:)',)
    assert callee.routine.arguments == ('t_a(:)',)

    calls = FindNodes(CallStatement).visit(caller.routine.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('t_a',)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_optional_named_arg(frontend):
    fcode = """
module some_mod
    implicit none
    type some_type
        integer, allocatable :: arr(:)
    end type some_type
contains
    subroutine callee(t, val, opt1, opt2)
        type(some_type), intent(inout) :: t
        integer, intent(in) :: val
        integer, intent(in), optional :: opt1
        integer, intent(in), optional :: opt2

        t%arr(:) = val

        if (present(opt1)) then
            t%arr(:) = t%arr(:) + opt1
        endif
        if (present(opt2)) then
            t%arr(:) = t%arr(:) + opt2
        endif
    end subroutine callee

    subroutine caller(t)
        type(some_type), intent(inout) :: t
        call callee(t, 1, opt2=2)
        call callee(t, 1, 1)
    end subroutine caller
end module some_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=frontend)

    callee = SubroutineItem(name='some_mod#callee', source=source)
    caller = SubroutineItem(name='some_mod#caller', source=source)

    transformation = DerivedTypeArgumentsTransformation()
    source.apply(transformation, item=callee, role='kernel', successors=())
    source.apply(transformation, item=caller, role='driver', successors=(callee,))

    assert not caller.trafo_data[transformation._key]
    assert callee.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%arr',)
    }

    calls = FindNodes(CallStatement).visit(caller.routine.body)
    assert len(calls) == 2
    assert calls[0].arguments == ('t%arr', '1')
    assert calls[0].kwarguments == (('opt2', '2'),)
    assert calls[1].arguments == ('t%arr', '1', '1')
    assert not calls[1].kwarguments
