# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from itertools import zip_longest
from shutil import rmtree
import pytest

from loki import (
    OMNI, OFP, Sourcefile, Scheduler, ProcedureItem, as_tuple, gettempdir,
    CallStatement, ProcedureDeclaration, Scalar, Array,
    FindNodes, FindVariables, FindInlineCalls, BasicType,
    CaseInsensitiveDict, resolve_associates
)
from conftest import available_frontends
from transformations import (
    DerivedTypeArgumentsTransformation,
    TypeboundProcedureCallTransformation
)
#pylint: disable=too-many-lines

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
            'enable_imports': True,
        },
        'routines': {
            'driver': {
                'role': 'driver',
                'expand': True,
            },
        }
    }


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
    item = ProcedureItem(name='transform_derived_type_arguments_mod#kernel', source=source)
    source['kernel'].apply(DerivedTypeArgumentsTransformation(), role='kernel', item=item)

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


@pytest.mark.parametrize('all_derived_types', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_expansion_trivial_derived_type(frontend, all_derived_types):
    fcode = """
module transform_derived_type_arguments_mod

    implicit none

    type some_derived_type
        real :: a
        real :: b
    end type some_derived_type

contains

    subroutine caller(z)
        integer, intent(in) :: z
        type(some_derived_type) :: t_io
        type(some_derived_type) :: t_in, t_out
        integer :: m, n
        integer :: i, j

        m = 100
        n = 10

        t_in%a = real(m-1)
        t_in%b = real(n-1)

        call kernel(m, n, t_io%a, t_io%b, t_in, t_out)

    end subroutine caller

    subroutine kernel(m, n, P_a, P_b, Q, R)
        integer                , intent(in)    :: m, n
        real, intent(inout)                    :: P_a, P_b
        type(some_derived_type), intent(in)    :: Q
        type(some_derived_type), intent(out)   :: R
        integer :: j, k

        R%a = P_a + Q%a
        R%b = P_b - Q%b
    end subroutine kernel
end module transform_derived_type_arguments_mod
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)

    call_tree = [
        ProcedureItem(name='transform_derived_type_arguments_mod#caller', source=source, config={'role': 'driver'}),
        ProcedureItem(name='transform_derived_type_arguments_mod#kernel', source=source, config={'role': 'kernel'}),
    ]

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation(all_derived_types=all_derived_types)
    for item, successor in reversed(list(zip_longest(call_tree, call_tree[1:]))):
        transformation.apply(item.scope_ir, role=item.role, item=item, successors=as_tuple(successor))

    # all derived types, disregarding whether the derived type has pointer/allocatable/derived type members or not
    if all_derived_types:
        call_args = ('m', 'n', 't_io%a', 't_io%b', 't_in%a', 't_in%b', 't_out%a', 't_out%b')
        kernel_args = ('m', 'n', 'P_a', 'P_b', 'Q_a', 'Q_b', 'R_a', 'R_b')
    # only the derived type(s) with pointer/allocatable/derived type members, thus no changes expected!
    else:
        call_args = ('m', 'n', 't_io%a', 't_io%b', 't_in', 't_out')
        kernel_args = ('m', 'n', 'P_a', 'P_b', 'Q', 'R')

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


@pytest.mark.parametrize('all_derived_types', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_expansion_trivial_derived_type_scheduler(frontend, all_derived_types,
        config, here):

    proj = here / 'sources/projDerivedTypes'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation(all_derived_types=all_derived_types)
    scheduler.process(transformation=transformation)

    # all derived types, disregarding whether the derived type has pointer/allocatable/derived type members or not
    if all_derived_types:
        call_args = ('m', 'n', 't_io%a', 't_io%b', 't_in%a', 't_in%b', 't_out%a', 't_out%b')
        kernel_args = ('m', 'n', 'P_a', 'P_b', 'Q_a', 'Q_b', 'R_a', 'R_b')
    # only the derived type(s) with pointer/allocatable/derived type members, thus no changes expected!
    else:
        call_args = ('m', 'n', 't_io%a', 't_io%b', 't_in', 't_out')
        kernel_args = ('m', 'n', 'P_a', 'P_b', 'Q', 'R')

    driver = scheduler["driver_mod#driver"].ir
    kernel = scheduler["kernel_mod#kernel"].ir
    calls = FindNodes(CallStatement).visit(driver.body)
    call = calls[0]
    assert call.name == 'kernel'
    assert call.arguments == call_args
    assert kernel.arguments == kernel_args
    assert all(v.type.intent for v in kernel.arguments)

    # Make sure rescoping hasn't accidentally overwritten the
    # type information for local variables that have the same name
    # as the shape of another variable
    assert driver.variable_map['m'].type.intent is None
    assert driver.variable_map['n'].type.intent is None


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
        ProcedureItem(name='transform_derived_type_arguments_mod#caller', source=source, config={'role': 'driver'}),
        ProcedureItem(name='transform_derived_type_arguments_mod#kernel', source=source, config={'role': 'kernel'}),
    ]

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for item, successor in reversed(list(zip_longest(call_tree, call_tree[1:]))):
        transformation.apply(item.ir, role=item.role, item=item, successors=as_tuple(successor))

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
def test_transform_derived_type_arguments_inline_call(frontend):
    """
    Verify correct expansion of inline calls to functions
    """
    fcode_my_mod = """
module my_mod
    implicit none
    type my_type
        integer, allocatable :: a(:)
        integer :: b
    end type my_type
contains
    function kernel(r, s) result(t)
        type(my_type), intent(in) :: r, s
        real :: t
        t = sum(r%a + s%a) + r%b + s%b
    end function kernel
end module my_mod
    """.strip()

    fcode_driver = """
subroutine driver(arr, n, s, t)
    use my_mod, only: my_type, kernel
    implicit none
    type(my_type), intent(in) :: arr(n), s
    integer, intent(in) :: n
    real, intent(inout) :: t(n)
    integer :: j
    do j=1,n
        t(j) = kernel(arr(j), s)
    end do
end subroutine driver
    """.strip()

    source_my_mod = Sourcefile.from_source(fcode_my_mod, frontend=frontend)
    source_driver = Sourcefile.from_source(fcode_driver, frontend=frontend, definitions=source_my_mod.definitions)

    kernel = ProcedureItem('my_mod#kernel', config={'role': 'kernel'}, source=source_my_mod)
    driver = ProcedureItem('#driver', config={'role': 'driver'}, source=source_driver)

    transformation = DerivedTypeArgumentsTransformation()
    transformation.apply(kernel.ir, item=kernel, role=kernel.role)
    transformation.apply(driver.ir, item=driver, role=driver.role, successors=[kernel])

    assert kernel.trafo_data[transformation._key] == {
        'orig_argnames': ('r', 's'),
        'expansion_map': {'r': ('r%a', 'r%b'), 's': ('s%a', 's%b')}
    }

    assert kernel.ir.arguments == ('r_a(:)', 'r_b', 's_a(:)', 's_b')

    inline_calls = list(FindInlineCalls().visit(driver.ir.body))
    assert len(inline_calls) == 1
    assert inline_calls[0].parameters == ('arr(j)%a', 'arr(j)%b', 's%a', 's%b')


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
        ProcedureItem(
            name='transform_derived_type_arguments_multilevel#caller',
            source=source, config={'role': 'driver'}
        ),
        ProcedureItem(
            name='transform_derived_type_arguments_multilevel#setup_obj',
            source=source, config={'role': 'kernel'}
        ),
        ProcedureItem(
            name='transform_derived_type_arguments_multilevel#deallocate_obj',
            source=source, config={'role': 'kernel'}
        ),
    ]

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for item, successor in reversed(list(zip_longest(call_tree, call_tree[1:]))):
        transformation.apply(item.ir, role=item.role, item=item, successors=as_tuple(successor))

    for item in call_tree:
        if item.role == 'driver':
            assert not item.trafo_data[transformation._key]
        else:
            assert item.trafo_data[transformation._key]['orig_argnames'] == orig_args[item.ir.name.lower()]

    for routine in source.subroutines:
        assert routine.arguments == transformed_args[routine.name.lower()]
        assert all(a.type.intent for a in routine.arguments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_expansion_nested(frontend):
    fcode_header = f"""
module header_mod
    implicit none
    integer, parameter :: jprb = selected_real_kind(13, 300)
    integer, parameter :: NUMBER_TWO = 2

    type some_derived_type
{'!$loki dimension(n)' if frontend is not OMNI else ''}
        real(kind=jprb), allocatable :: a(:)
{'!$loki dimension(m, n)' if frontend is not OMNI else ''}
        real(kind=jprb), allocatable :: b(:,:)
    end type some_derived_type

    type constants_type
        real(kind=jprb) :: c
        real(kind=jprb), allocatable :: other(:)
    end type constants_type
end module header_mod
    """.strip()

    fcode_bucket = """
module bucket_mod
    use header_mod, only: some_derived_type, constants_type, number_two
    implicit none
    integer, parameter :: NUMBER_FIVE = 5

    type bucket_type
        type(some_derived_type) :: a
        type(some_derived_type) :: b(NUMBER_FIVE)
        type(constants_type) :: constants(number_two)
    end type bucket_type

contains

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
        use header_mod, only: jprb
        type(some_derived_type), intent(inout) :: t
        integer, intent(in) :: m, n
        integer j

        do j=1,n
            t%a(j) = real(j, kind=jprb)
            t%b(:, j) = real(j, kind=jprb)
        end do
    end subroutine init

    subroutine kernel(m, n, P_a, P_b, Q, R, c)
        use header_mod, only: jprb
        integer                , intent(in)    :: m, n
        real(kind=jprb), intent(in)            :: P_a(n), P_b(m, n), c
        type(some_derived_type), intent(in)    :: Q
        type(some_derived_type), intent(out)   :: R
        integer :: j, k

        do j=1,n
            R%a(j) = P_a(j) + Q%a(j) + c
            do k=1,m
                R%b(k, j) = P_b(k, j) - Q%b(k, j) + c
            end do
        end do
    end subroutine kernel
end module bucket_mod
    """.strip()

    fcode_layer = """
module layer_mod
contains
    subroutine layer(m, n, P_a, P_b, Q, R)
        use bucket_mod, only: bucket_type
        use header_mod, only: some_derived_type
        implicit none
        integer                , intent(in) :: m, n
        type(some_derived_type), intent(in) :: P_a, P_b(5)
        type(bucket_type), intent(in)       :: Q
        type(bucket_type), intent(out)      :: R
        integer :: k

        call kernel(m, n, P_a%a, P_a%b, Q%a, R%a, Q%constants(1)%c)
        do k=1,5
            call kernel(m, n, P_b(k)%a, P_b(k)%b, Q%b(k), R%b(k), Q%constants(2)%c)
        end do
    end subroutine layer
end module layer_mod
    """.strip()

    fcode_caller = """
subroutine caller(z)
    use bucket_mod, only: bucket_type, setup, init, teardown
    use layer_mod, only: layer
    implicit none

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
    """.strip()

    header = Sourcefile.from_source(fcode_header, frontend=frontend)
    bucket = Sourcefile.from_source(fcode_bucket, frontend=frontend, definitions=header.definitions)
    layer = Sourcefile.from_source(fcode_layer, frontend=frontend, definitions=header.definitions + bucket.definitions)
    source = Sourcefile.from_source(fcode_caller, frontend=frontend,
                                    definitions=header.definitions + bucket.definitions + layer.definitions)

    items = {
        'caller': ProcedureItem(
            name='#caller', source=source, config={'role': 'driver'}
        ),
        'layer': ProcedureItem(
            name='layer_mod#layer', source=layer, config={'role': 'kernel'}
        ),
        'setup': ProcedureItem(
            name='bucket_mod#setup', source=bucket, config={'role': 'kernel'}
        ),
        'init': ProcedureItem(
            name='bucket_mod#init', source=bucket, config={'role': 'kernel'}
        ),
        'kernel': ProcedureItem(
            name='bucket_mod#kernel', source=bucket, config={'role': 'kernel'}
        ),
        'teardown': ProcedureItem(
            name='bucket_mod#teardown', source=bucket, config={'role': 'kernel'}
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

    assert len(items['layer'].ir.imports) == 2

    # Apply transformation
    transformation = DerivedTypeArgumentsTransformation()
    for name, successors in reversed(call_tree):
        item = items[name]
        children = [items[c] for c in successors]
        transformation.apply(item.ir, role=item.role, item=item, successors=as_tuple(children))

    key = DerivedTypeArgumentsTransformation._key

    # Check analysis result in kernel
    assert key in items['kernel'].trafo_data
    assert items['kernel'].trafo_data[key]['expansion_map'] == {
        'q': ('q%a', 'q%b'),
        'r': ('r%a', 'r%b'),
    }
    assert items['kernel'].trafo_data[key]['orig_argnames'] == (
        'm', 'n', 'p_a', 'p_b', 'q', 'r', 'c'
    )

    # Check analysis result in layer
    assert key in items['layer'].trafo_data
    assert items['layer'].trafo_data[key]['expansion_map'] == {
        'p_a': ('p_a%a', 'p_a%b'),
        'q': ('q%a%a', 'q%a%b', 'q%b', 'q%constants'),
        'r': ('r%a%a', 'r%a%b', 'r%b')
    }
    assert items['layer'].trafo_data[key]['orig_argnames'] == (
        'm', 'n', 'p_a', 'p_b', 'q', 'r'
    )

    # Check arguments of setup
    assert items['setup'].ir.arguments == (
        't_a(:)', 't_b(:, :)', 'm', 'n'
    )

    # Check arguments of init
    assert items['init'].ir.arguments == (
        't_a(:)', 't_b(:, :)', 'm', 'n'
    )

    # Check arguments of teardown
    assert items['teardown'].ir.arguments == (
        't_a(:)', 't_b(:, :)'
    )

    # Check arguments of kernel
    if frontend == OMNI:
        assert items['kernel'].ir.arguments == (
            'm', 'n', 'p_a(1:n)', 'p_b(1:m, 1:n)', 'q_a(:)', 'q_b(:, :)', 'r_a(:)', 'r_b(:, :)', 'c'
        )
    else:
        assert items['kernel'].ir.arguments == (
            'm', 'n', 'P_a(n)', 'P_b(m, n)', 'Q_a(:)', 'Q_b(:, :)', 'R_a(:)', 'R_b(:, :)', 'c'
        )

    # Check call arguments in layer
    calls = FindNodes(CallStatement).visit(items['layer'].ir.ir)
    assert len(calls) == 2

    assert calls[0].arguments == (
        'm', 'n', 'p_a_a', 'p_a_b', 'q_a_a', 'q_a_b', 'r_a_a', 'r_a_b', 'q_constants(1)%c'
    )
    assert calls[1].arguments == (
        'm', 'n', 'p_b(k)%a', 'p_b(k)%b', 'q_b(k)%a', 'q_b(k)%b', 'r_b(k)%a', 'r_b(k)%b', 'q_constants(2)%c'
    )

    # Check arguments of layer
    if frontend == OMNI:
        assert items['layer'].ir.arguments == (
            'm', 'n', 'p_a_a(:)', 'p_a_b(:, :)', 'p_b(1:5)', 'q_a_a(:)', 'q_a_b(:, :)', 'q_b(:)', 'q_constants(:)',
            'r_a_a(:)', 'r_a_b(:, :)', 'r_b(:)'
        )
    else:
        assert items['layer'].ir.arguments == (
            'm', 'n', 'p_a_a(:)', 'p_a_b(:, :)', 'p_b(5)', 'q_a_a(:)', 'q_a_b(:, :)', 'q_b(:)', 'q_constants(:)',
            'r_a_a(:)', 'r_a_b(:, :)', 'r_b(:)'
        )

    # Check imports
    assert 'constants_type' in items['layer'].ir.imported_symbols
    if frontend != OMNI:
        # OMNI inlines parameters
        assert 'jprb' in items['layer'].ir.imported_symbols
        assert 'jprb' in items['setup'].ir.imported_symbols
        assert 'jprb' in items['teardown'].ir.imported_symbols

    # No additional imports added for init and kernel
    assert len(items['init'].ir.imports) == 1
    assert len(items['kernel'].ir.imports) == 1

    # Cached property updated?
    assert len(items['layer'].ir.imports) == 3

    # Check call arguments in caller
    for call in FindNodes(CallStatement).visit(items['caller'].ir.body):
        if call.name in ('setup', 'init'):
            assert len(call.arguments) == 4
        elif call.name == 'teardown':
            assert len(call.arguments) == 2
        elif call.name == 'layer':
            assert call.arguments == (
                'm', 'n', 't_io%a%a', 't_io%a%b', 't_io%b',
                't_in(i)%a%a', 't_in(i)%a%b', 't_in(i)%b', 't_in(i)%constants',
                't_out(i)%a%a', 't_out(i)%a%b', 't_out(i)%b',
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
        val = start + sum(this%a + sum(this%b + this%c, 1))
    end function reduce
end module transform_derived_type_arguments_mod
    """.strip()

    fcode_driver = """
subroutine driver(some, result)
    use transform_derived_type_arguments_mod, only: some_derived_type, reduce
    implicit none
    type(some_derived_type), intent(in) :: some
    real, intent(inout) :: result
    result = reduce(result, some)
end subroutine driver
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    kernel_a = ProcedureItem(name='transform_derived_type_arguments_mod#kernel_a', source=source)
    kernel = ProcedureItem(name='transform_derived_type_arguments_mod#kernel', source=source)
    reduce = ProcedureItem(name='transform_derived_type_arguments_mod#reduce', source=source)
    source_driver = Sourcefile.from_source(fcode_driver, frontend=frontend, definitions=source.definitions)
    driver = ProcedureItem(name='#driver', source=source_driver)

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
    source['kernel_a'].apply(transformation, role='kernel', item=kernel_a)
    source['kernel'].apply(transformation, role='kernel', item=kernel)
    source['reduce'].apply(transformation, role='kernel', item=reduce)
    source_driver['driver'].apply(transformation, role='driver', item=driver, successors=[reduce])

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
    assert kernel_a.ir.arguments == ('this_a(:)', 'out(:)', 'n')
    assert kernel.ir.arguments == ('this_b(:, :)', 'this_c(:, :)', 'other_b(:, :)', 'm', 'n')
    assert reduce.ir.arguments == ('start', 'this_a(:)', 'this_b(:, :)', 'this_c(:, :)')

    inline_calls = list(FindInlineCalls().visit(driver.ir.body))
    assert len(inline_calls) == 1
    assert inline_calls[0].parameters == ('result', 'some%a', 'some%b', 'some%c')

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

    callee = ProcedureItem(name='some_mod#some_routine', source=source1)
    caller = ProcedureItem(name='#some_routine', source=source2)

    transformation = DerivedTypeArgumentsTransformation()
    source1['some_routine'].apply(transformation, item=callee, role='kernel', successors=())
    source2['some_routine'].apply(transformation, item=caller, role='kernel', successors=(callee,))

    assert caller.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%a',),
    }
    assert callee.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%a',),
    }

    assert caller.ir.arguments == ('t_a(:)',)
    assert callee.ir.arguments == ('t_a(:)',)

    calls = FindNodes(CallStatement).visit(caller.ir.body)
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
        call callee(opt1=1, val=1, t=t, opt2=2)
    end subroutine caller
end module some_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=frontend)

    callee = ProcedureItem(name='some_mod#callee', source=source)
    caller = ProcedureItem(name='some_mod#caller', source=source)

    transformation = DerivedTypeArgumentsTransformation()
    source['callee'].apply(transformation, item=callee, role='kernel', successors=())
    source['caller'].apply(transformation, item=caller, role='driver', successors=(callee,))

    assert not caller.trafo_data[transformation._key]
    assert callee.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%arr',)
    }

    calls = FindNodes(CallStatement).visit(caller.ir.body)
    assert len(calls) == 3
    assert calls[0].arguments == ('t%arr', '1')
    assert calls[0].kwarguments == (('opt2', '2'),)
    assert calls[1].arguments == ('t%arr', '1', '1')
    assert not calls[1].kwarguments
    assert not calls[2].arguments
    assert calls[2].kwarguments == (('opt1', '1'), ('val', '1'), ('t_arr', 't%arr'), ('opt2', '2'))


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP, 'No support for recursive prefix')]))
def test_transform_derived_type_arguments_recursive(frontend):
    fcode = """
module some_mod
    implicit none
    type some_type
        integer, allocatable :: arr(:)
    end type some_type
contains
    recursive subroutine callee(t, val, opt1, opt2, recurse)
        type(some_type), intent(inout) :: t
        integer, intent(in) :: val
        integer, intent(in), optional :: opt1
        integer, intent(in), optional :: opt2
        logical, intent(in), optional :: recurse

        if (present(recurse)) then
            if (recurse) then
                call callee(t, val, opt1, opt2, recurse=.false.)
            endif
        endif

        t%arr(:) = val

        if (present(opt1)) then
            t%arr(:) = t%arr(:) + opt1
        endif
        if (present(opt2)) then
            t%arr(:) = t%arr(:) + opt2
        endif
    end subroutine callee

    recursive function plus(t, val, idx, stop_recurse) result(retval)
        type(some_type), intent(in) :: t
        integer, intent(in) :: val, idx
        logical, intent(in), optional :: stop_recurse
        integer :: retval

        if (present(stop_recurse)) then
            if (stop_recurse) then
                retval = t%arr(idx)
                return
            end if
        endif

        if (val == 2) then
            retval = plus(t, 1, idx)
        elseif (val < 2) then
            retval = plus(t, 0, idx, stop_recurse=.true.)
        else
            retval = plus(t, val-1, idx)
        endif

        retval = retval + 1

    end function plus

    subroutine caller(t)
        type(some_type), intent(inout) :: t
        call callee(t, 1, opt2=2)
        call callee(t, 1, 1)
        t%arr(1) = plus(t, 32, 1)
    end subroutine caller
end module some_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=frontend)

    callee = ProcedureItem(name='some_mod#callee', source=source)
    caller = ProcedureItem(name='some_mod#caller', source=source)
    plus = ProcedureItem(name='some_mod#plus', source=source)

    transformation = DerivedTypeArgumentsTransformation()
    source['callee'].apply(transformation, item=callee, role='kernel', successors=())
    source['plus'].apply(transformation, item=plus, role='kernel', successors=())
    source['caller'].apply(transformation, item=caller, role='driver', successors=(callee, plus))

    assert not caller.trafo_data[transformation._key]
    assert callee.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%arr',)
    }
    assert plus.trafo_data[transformation._key]['expansion_map'] == {
        't': ('t%arr',)
    }

    calls = FindNodes(CallStatement).visit(caller.ir.body)
    assert len(calls) == 2
    assert calls[0].arguments == ('t%arr', '1')
    assert calls[0].kwarguments == (('opt2', '2'),)
    assert calls[1].arguments == ('t%arr', '1', '1')
    assert not calls[1].kwarguments

    inline_calls = list(FindInlineCalls().visit(caller.ir.body))
    assert len(inline_calls) == 1
    assert inline_calls[0].parameters == ('t%arr', '32', '1')
    assert not inline_calls[0].kw_parameters

    assert callee.ir.arguments == ('t_arr(:)', 'val', 'opt1', 'opt2', 'recurse')
    assert callee.ir.arguments[0].type.intent == 'inout'

    calls = FindNodes(CallStatement).visit(callee.ir.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('t_arr', 'val', 'opt1', 'opt2')
    assert calls[0].kwarguments == (('recurse', 'False'),)

    inline_calls = list(FindInlineCalls().visit(plus.ir.body))
    inline_calls = [call for call in inline_calls if call.name == 'plus']
    assert len(inline_calls) == 3
    for call in inline_calls:
        if call.kwarguments:
            assert call.parameters == ('t_arr', '0', 'idx')
            assert call.kwarguments == (('stop_recurse', 'True'),)
        else:
            assert call.parameters in [
                ('t_arr', '1', 'idx'), ('t_arr', 'val - 1', 'idx')
            ]

    assert plus.ir.arguments == ('t_arr(:)', 'val', 'idx', 'stop_recurse')
    assert plus.ir.arguments[0].type.intent == 'in'


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_renamed_calls(frontend):
    fcode_header = """
module header_mod
    implicit none
    type some_type
        integer, allocatable :: some(:)
        integer, allocatable :: other(:)
    end type some_type
end module header_mod
    """.strip()
    fcode_some = """
module some_mod
    implicit none
contains
    subroutine sub(t)
        use header_mod, only: some_type
        type(some_type), intent(inout) :: t
        t%some(:) = 1
    end subroutine sub
end module some_mod
    """.strip()
    fcode_other = """
module other_mod
    implicit none
contains
    subroutine sub(t)
        use header_mod, only: some_type
        type(some_type), intent(inout) :: t
        t%other(:) = 2
    end subroutine sub
end module other_mod
    """.strip()
    fcode_caller = """
subroutine caller(t)
    use header_mod, only: some_type
    use some_mod, only: some_sub => sub
    use other_mod, only: sub
    implicit none
    type(some_type), intent(inout) :: t
    call some_sub(t)
    call sub(t)
end subroutine caller
    """.strip()

    source_header = Sourcefile.from_source(fcode_header, frontend=frontend)
    source_some = Sourcefile.from_source(fcode_some, frontend=frontend, definitions=source_header.definitions)
    source_other = Sourcefile.from_source(fcode_other, frontend=frontend, definitions=source_header.definitions)
    source_caller = Sourcefile.from_source(
        fcode_caller, frontend=frontend,
        definitions=source_header.definitions + source_some.definitions + source_other.definitions
    )

    some_sub = ProcedureItem(name='some_mod#sub', source=source_some)
    other_sub = ProcedureItem(name='other_mod#sub', source=source_other)
    caller = ProcedureItem(name='#caller', source=source_caller)

    transformation = DerivedTypeArgumentsTransformation()
    source_some['sub'].apply(transformation, item=some_sub, role='kernel', successors=())
    source_other['sub'].apply(transformation, item=other_sub, role='kernel', successors=())
    source_caller['caller'].apply(transformation, item=caller, role='driver', successors=(some_sub, other_sub))

    assert some_sub.ir.arguments == ('t_some(:)',)
    assert other_sub.ir.arguments == ('t_other(:)',)
    calls = FindNodes(CallStatement).visit(caller.ir.body)
    assert len(calls) == 2
    assert calls[0].arguments == ('t%some',)
    assert calls[1].arguments == ('t%other',)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_associate_intent(frontend):
    fcode = """
module some_mod
    implicit none
    type some_type
        real, allocatable :: arr(:)
    end type some_type
contains
    subroutine some_routine(t)
        type(some_type), intent(inout) :: t
        associate(arr=>t%arr)
            arr(:) = arr(:) + 1
        end associate
    end subroutine some_routine
end module some_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=frontend)

    variables = FindVariables().visit(source['some_routine'].body)
    assert variables == {'arr(:)', 'arr', 't%arr', 't'}
    variable_map = CaseInsensitiveDict((v.name, v) for v in variables)
    assert variable_map['t'].type.intent == 'inout'
    assert variable_map['arr'].type.intent is None

    resolve_associates(source['some_routine'])
    variables = FindVariables().visit(source['some_routine'].body)
    assert variables == {'t', 't%arr(:)'}
    variable_map = CaseInsensitiveDict((v.name, v) for v in variables)
    assert variable_map['t'].type.intent == 'inout'
    assert variable_map['t%arr'].type.intent is None

    transformation = DerivedTypeArgumentsTransformation()
    source['some_routine'].apply(transformation, role='kernel')
    variables = FindVariables().visit(source['some_routine'].body)
    assert variables == {'t_arr(:)'}
    variable_map = CaseInsensitiveDict((v.name, v) for v in variables)
    assert variable_map['t_arr'].type.intent == 'inout'


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_derived_type_arguments_non_array(frontend):
    fcode = """
module some_mod
    implicit none
    type scalar_type
        integer :: i
    end type scalar_type
    type array_type
        integer, allocatable :: a(:)
    end type array_type
    type nested_type
        type(scalar_type) :: s
    end type nested_type
contains
    subroutine kernel(s, a, n)
        type(scalar_type), intent(inout) :: s
        type(array_type), intent(inout) :: a
        type(nested_type), intent(inout) :: n
        s%i = 1
        a%a(:) = 2
        n%s%i = 3
    end subroutine kernel
end module some_mod
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=frontend)

    transformation = DerivedTypeArgumentsTransformation()
    source['kernel'].apply(transformation, role='kernel')
    # Only type with derived type member
    assert source['kernel'].arguments == ('s', 'a_a(:)', 'n_s_i')


@pytest.mark.parametrize('duplicate', [False,True])
@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_typebound_procedure_calls(frontend, config, duplicate):
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

    scheduler = Scheduler(
        paths=[workdir], config=config, seed_routines=['driver'], frontend=frontend
    )

    transformation = TypeboundProcedureCallTransformation(duplicate_typebound_kernels=duplicate)
    scheduler.process(transformation=transformation)

    driver = scheduler['#driver'].ir
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

    add_other_type = scheduler['typebound_procedure_calls_mod#add_other_type'].ir
    calls = FindNodes(CallStatement).visit(add_other_type.body)
    assert len(calls) == 1
    assert calls[0].name == 'add_my_type'
    assert calls[0].arguments == ('this%arr(i)', 'other%arr(i)%val')

    init = scheduler['other_typebound_procedure_calls_mod#init'].ir
    calls = FindNodes(CallStatement).visit(init.body)
    assert len(calls) == 2
    assert calls[0].name == 'reset'
    assert calls[0].arguments == ('this%stuff(i)%arr(j)',)
    assert calls[1].name == 'add_my_type'
    assert calls[1].arguments == ('this%stuff(i)%arr(j)', 'i + j')

    print_content = scheduler['other_typebound_procedure_calls_mod#print_content'].ir
    calls = FindNodes(CallStatement).visit(print_content.body)
    assert len(calls) == 1
    assert calls[0].name == 'add_other_type'
    assert calls[0].arguments == ('this%stuff(1)', 'this%stuff(2)')

    calls = list(FindInlineCalls().visit(print_content.body))
    assert len(calls) == 1
    assert str(calls[0]).lower() == 'total_sum(this%stuff(1))'

    if duplicate:
        mod = scheduler['typebound_procedure_calls_mod#add_other_type'].ir.parent

        assert [r.name.lower() for r in mod.subroutines] == [
            'reset', 'add_my_type', 'add_other_type', 'total_sum',
            'add_other_type_', 'total_sum_', 'reset_', 'add_my_type_',
        ]

        my_type = mod['my_type']
        assert my_type.variable_map['reset'].type.bind_names == ('reset_',)
        assert my_type.variable_map['add'].type.bind_names == ('add_my_type_',)
        other_type = mod['other_type']
        assert other_type.variable_map['add'].type.bind_names == ('add_other_type_',)
        assert other_type.variable_map['total_sum'].type.bind_names == ('total_sum_',)

        other_mod = scheduler['other_typebound_procedure_calls_mod#init'].ir.parent

        assert [r.name.lower() for r in other_mod.subroutines] == [
            'init', 'print_content', 'init_', 'print_content_'
        ]

        third_type = other_mod['third_type']
        assert third_type.variable_map['init'].type.bind_names == ('init_',)
        assert third_type.variable_map['print'].type.bind_names == ('print_content_',)

        assert [
            str(call.name) for call in FindNodes(CallStatement).visit(other_mod['init_'].ir)
        ] == ['this%stuff(i)%arr(j)%reset', 'this%stuff(i)%arr(j)%add']

    rmtree(workdir)
