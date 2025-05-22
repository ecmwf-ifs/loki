# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Sourcefile, Module
import loki.expression.symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes, Pragma, CallStatement
from loki.logging import log_levels

from loki.transformations import FieldOffloadTransformation, FieldOffloadBlockedTransformation


@pytest.fixture(name="parkind_mod")
def fixture_parkind_mod(tmp_path, frontend):
    fcode = """
    module parkind1
      integer, parameter :: jprb=4
    end module
    """
    return Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])


@pytest.fixture(name="field_module")
def fixture_field_module(tmp_path, frontend):
    fcode = """
    module field_module
      implicit none

      type field_2rb
        real, pointer :: f_ptr(:,:,:)
      end type field_2rb

      type field_3rb
        real, pointer :: f_ptr(:,:,:)
     contains
        procedure :: update_view
      end type field_3rb
      
      type field_4rb
        real, pointer :: f_ptr(:,:,:)
     contains
        procedure :: update_view
      end type field_4rb

    contains
    subroutine update_view(self, idx)
      class(field_3rb), intent(in)  :: self
      integer, intent(in)           :: idx
    end subroutine
    end module
    """
    return Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])


@pytest.fixture(name="state_module")
def fixture_state_module(tmp_path, parkind_mod, field_module, frontend):  # pylint: disable=unused-argument
    fcode = """
    module state_mod
      use parkind1, only: jprb
      use field_module, only: field_2rb, field_3rb
      implicit none

      type state_type
        real(kind=jprb), dimension(10,10), pointer :: a, b, c
        real(kind=jprb), pointer :: d(10,10,10)
        class(field_3rb), pointer :: f_a, f_b, f_c
        class(field_4rb), pointer :: f_d
        contains
        procedure :: update_view => state_update_view
      end type state_type

    contains

      subroutine state_update_view(self, idx)
        class(state_type), intent(in) :: self
        integer, intent(in)           :: idx
      end subroutine
    end module state_mod
"""
    return Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload(frontend, state_module, tmp_path):
    fcode = """
    module driver_mod
      use state_mod, only: state_type
      use parkind1, only: jprb
      use field_module, only: field_2rb, field_3rb
      implicit none

    contains

      subroutine kernel_routine(nlon, nlev, a, b, c)
        integer, intent(in)             :: nlon, nlev
        real(kind=jprb), intent(in)     :: a(nlon,nlev)
        real(kind=jprb), intent(inout)  :: b(nlon,nlev)
        real(kind=jprb), intent(out)    :: c(nlon,nlev)
        integer :: i, j

        do j=1, nlon
          do i=1, nlev
            b(i,j) = a(i,j) + 0.1
            c(i,j) = 0.1
          end do
        end do
      end subroutine kernel_routine

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i

        !$loki data
        do i=1,nlev
            call state%update_view(i)
            call kernel_routine(nlon, nlev, state%a, state%b, state%c)
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """
    driver_mod = Module.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']
    deviceptr_prefix = 'loki_devptr_prefix_'
    driver.apply(FieldOffloadTransformation(devptr_prefix=deviceptr_prefix,
                                            offload_index='i',
                                            field_group_types=['state_type']),
                 role='driver',
                 targets=['kernel_routine'])

    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_call = next(c for c in calls if c.name=='kernel_routine')

    # verify that field offloads are generated properly
    in_calls = [c for c in calls if 'get_device_data_rdonly' in c.name.name.lower()]
    assert len(in_calls) == 1
    inout_calls = [c for c in calls if 'get_device_data_rdwr' in c.name.name.lower()]
    assert len(inout_calls) == 2
    # verify that field sync host calls are generated properly
    sync_calls = [c for c in calls if 'sync_host_rdwr' in c.name.name.lower()]
    assert len(sync_calls) == 2

    # verify that data offload pragmas remain
    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 2
    assert all(p.keyword=='loki' and p.content==c for p, c in zip(pragmas, ['data', 'end data']))

    # verify that new pointer variables are created and used in driver calls
    for var in ['state_a', 'state_b', 'state_c']:
        name = deviceptr_prefix + var
        assert name in driver.variable_map
        devptr = driver.variable_map[name]
        assert isinstance(devptr, sym.Array)
        assert len(devptr.shape) == 3
        assert devptr.name in (arg.name for arg in kernel_call.arguments)
    from loki import fgen


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_slices(frontend, parkind_mod, field_module, tmp_path):  # pylint: disable=unused-argument
    fcode = """
    module driver_mod
      use parkind1, only: jprb
      use field_module, only: field_4rb
      implicit none

      type state_type
        real(kind=jprb), dimension(10,10,10), pointer :: a, b, c, d
        class(field_4rb), pointer :: f_a, f_b, f_c, f_d
        contains
        procedure :: update_view => state_update_view
      end type state_type

    contains

      subroutine state_update_view(self, idx)
        class(state_type), intent(in) :: self
        integer, intent(in)           :: idx
      end subroutine

      subroutine kernel_routine(nlon, nlev, a, b, c, d)
        integer, intent(in)             :: nlon, nlev
        real(kind=jprb), intent(in)     :: a(nlon,nlev,nlon)
        real(kind=jprb), intent(inout)  :: b(nlon,nlev)
        real(kind=jprb), intent(out)    :: c(nlon)
        real(kind=jprb), intent(in)     :: d(nlon,nlev,nlon)
        integer :: i, j
      end subroutine kernel_routine

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i
        !$loki data
        do i=1,nlev
            call kernel_routine(nlon, nlev, state%a(:,:,1), state%b(:,1,1), state%c(1,1,1), state%d)
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """
    driver_mod = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])['driver_mod']
    driver = driver_mod['driver_routine']
    deviceptr_prefix = 'loki_devptr_prefix_'
    driver.apply(FieldOffloadTransformation(devptr_prefix=deviceptr_prefix,
                                            offload_index='i',
                                            field_group_types=['state_type']),
                 role='driver',
                 targets=['kernel_routine'])

    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_call = next(c for c in calls if c.name=='kernel_routine')
    # verify that new pointer variables are created and used in driver calls
    for var, rank in zip(['state_d', 'state_a', 'state_b', 'state_c',], [4, 3, 2, 1]):
        name = deviceptr_prefix + var
        assert name in driver.variable_map
        devptr = driver.variable_map[name]
        assert isinstance(devptr, sym.Array)
        assert len(devptr.shape) == 4
        assert devptr.name in (arg.name for arg in kernel_call.arguments)
        arg = next(arg for arg in kernel_call.arguments if devptr.name in arg.name)
        assert arg.dimensions == ((sym.RangeIndex((None,None)),)*(rank-1) +
                                 (sym.IntLiteral(1),)*(4-rank) +
                                 (sym.Scalar(name='i'),))


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_multiple_calls(frontend, state_module, tmp_path):
    fcode = """
    module driver_mod
      use parkind1, only: jprb
      use state_mod, only: state_type
      implicit none

    contains

      subroutine kernel_routine(nlon, nlev, a, b, c)
        integer, intent(in)             :: nlon, nlev
        real(kind=jprb), intent(in)     :: a(nlon,nlev)
        real(kind=jprb), intent(inout)  :: b(nlon,nlev)
        real(kind=jprb), intent(out)    :: c(nlon,nlev)
        integer :: i, j

        do j=1, nlon
          do i=1, nlev
            b(i,j) = a(i,j) + 0.1
            c(i,j) = 0.1
          end do
        end do
      end subroutine kernel_routine

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i

        !$loki data
        do i=1,nlev
            call state%update_view(i)

            call kernel_routine(nlon, nlev, state%a, state%b, state%c)

            call kernel_routine(nlon, nlev, state%a, state%b, state%c)
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """

    driver_mod = Module.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']
    deviceptr_prefix = 'loki_devptr_prefix_'
    driver.apply(FieldOffloadTransformation(devptr_prefix=deviceptr_prefix,
                                            offload_index='i',
                                            field_group_types=['state_type']),
                 role='driver',
                 targets=['kernel_routine'])
    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_calls = [c for c in calls if c.name=='kernel_routine']

    # verify that field offloads are generated properly
    in_calls = [c for c in calls if 'get_device_data_rdonly' in c.name.name.lower()]
    assert len(in_calls) == 1
    inout_calls = [c for c in calls if 'get_device_data_rdwr' in c.name.name.lower()]
    assert len(inout_calls) == 2
    # verify that field sync host calls are generated properly
    sync_calls = [c for c in calls if 'sync_host_rdwr' in c.name.name.lower()]
    assert len(sync_calls) == 2

    # verify that data offload pragmas remain
    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 2
    assert all(p.keyword=='loki' and p.content==c for p, c in zip(pragmas, ['data', 'end data']))

    # verify that new pointer variables are created and used in driver calls
    for var in ['state_a', 'state_b', 'state_c']:
        name = deviceptr_prefix + var
        assert name in driver.variable_map
        devptr = driver.variable_map[name]
        assert isinstance(devptr, sym.Array)
        assert len(devptr.shape) == 3
        assert devptr.name in (arg.name for kernel_call in kernel_calls for arg in kernel_call.arguments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_unknown_kernel(caplog, frontend, state_module, tmp_path):
    fother = """
    module another_module
      implicit none
    contains
      subroutine another_kernel(nlon, nlev, a, b, c)
        integer, intent(in)             :: nlon, nlev
        real, intent(in)     :: a(nlon,nlev)
        real, intent(inout)  :: b(nlon,nlev)
        real, intent(out)    :: c(nlon,nlev)
        integer :: i, j
      end subroutine
    end module
    """

    fcode = """
    module driver_mod
      use parkind1, only: jprb
      use state_mod, only: state_type
      use another_module, only: another_kernel
      implicit none

    contains

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i

        !$loki data
        do i=1,nlev
            call state%update_view(i)
            call another_kernel(nlon, nlev, state%a, state%b, state%c)
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """

    Sourcefile.from_source(fother, frontend=frontend, xmods=[tmp_path])
    driver_mod = Module.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']
    deviceptr_prefix = 'loki_devptr_prefix_'

    field_offload_trafo = FieldOffloadTransformation(devptr_prefix=deviceptr_prefix,
                                                         offload_index='i',
                                                         field_group_types=['state_type'])
    caplog.clear()
    with caplog.at_level(log_levels['WARNING']):
        driver.apply(field_offload_trafo, role='driver', targets=['another_kernel'])
        assert len(caplog.records) == 1
        assert ('[Loki] Data offload: Routine driver_routine has not been enriched '+
                'in another_kernel') in caplog.records[0].message


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_warnings(caplog, frontend, state_module, tmp_path):
    fother_state = """
    module state_type_mod
      implicit none
      type state_type2
        real, dimension(10,10), pointer :: a, b, c
      contains
        procedure :: update_view => state_update_view
      end type state_type2

    contains

      subroutine state_update_view(self, idx)
        class(state_type2), intent(in) :: self
        integer, intent(in)           :: idx
      end subroutine
    end module
    """

    fother_mod= """
    module another_module
      implicit none
    contains
      subroutine another_kernel(nlon, nlev, a, b, c)
        integer, intent(in)             :: nlon, nlev
        real, intent(in)     :: a(nlon,nlev)
        real, intent(inout)  :: b(nlon,nlev)
        real, intent(out)    :: c(nlon,nlev)
        integer :: i, j
      end subroutine
    end module
    """

    fcode = """
    module driver_mod
      use state_type_mod, only: state_type2
      use parkind1, only: jprb
      use state_mod, only: state_type
      use another_module, only: another_kernel

      implicit none

    contains

      subroutine kernel_routine(nlon, nlev, a, b, c)
        integer, intent(in)             :: nlon, nlev
        real(kind=jprb), intent(in)     :: a(nlon,nlev)
        real(kind=jprb), intent(inout)  :: b(nlon,nlev)
        real(kind=jprb), intent(out)    :: c(nlon,nlev)
        integer :: i, j

        do j=1, nlon
          do i=1, nlev
            b(i,j) = a(i,j) + 0.1
            c(i,j) = 0.1
          end do
        end do
      end subroutine kernel_routine

      subroutine driver_routine(nlon, nlev, state, state2)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        type(state_type2), intent(inout) :: state2

        integer                         :: i
        real(kind=jprb)                 :: a(nlon,nlev)
        real, pointer                   :: loki_devptr_prefix_state_b

        !$loki data
        do i=1,nlev
            call state%update_view(i)
            call kernel_routine(nlon, nlev, a, state%b, state2%c)
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """
    Sourcefile.from_source(fother_state, frontend=frontend, xmods=[tmp_path])
    Sourcefile.from_source(fother_mod, frontend=frontend, xmods=[tmp_path])
    driver_mod = Sourcefile.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )['driver_mod']
    driver = driver_mod['driver_routine']
    deviceptr_prefix = 'loki_devptr_prefix_'

    field_offload_trafo = FieldOffloadTransformation(devptr_prefix=deviceptr_prefix,
                                                         offload_index='i',
                                                         field_group_types=['state_type'])
    caplog.clear()
    with caplog.at_level(log_levels['WARNING']):
        driver.apply(field_offload_trafo, role='driver', targets=['kernel_routine'])
        assert len(caplog.records) == 3
        assert (('[Loki] Data offload: Raw array object a encountered in'
                 +' driver_routine that is not wrapped by a Field API object')
                in caplog.records[0].message)
        assert ('[Loki] Data offload: The parent object state2 of type state_type2 is not in the' +
                ' list of field wrapper types') in caplog.records[1].message
        assert ('[Loki] Data offload: The routine driver_routine already has a' +
                ' variable named loki_devptr_prefix_state_b') in caplog.records[2].message


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_aliasing(frontend, state_module, tmp_path):
    fcode = """
    module driver_mod
      use state_mod, only: state_type
      use parkind1, only: jprb
      implicit none

    contains

      subroutine kernel_routine(nlon, nlev, a1, a2, a3)
        integer, intent(in)             :: nlon, nlev
        real(kind=jprb), intent(in)     :: a1(nlon)
        real(kind=jprb), intent(inout)  :: a2(nlon)
        real(kind=jprb), intent(out)    :: a3(nlon)
        integer :: i

        do i=1, nlon
          a1(i) = a2(i) + 0.1
          a3(i) = 0.1
        end do
      end subroutine kernel_routine

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i

        !$loki data
        do i=1,nlev
            call state%update_view(i)
            call kernel_routine(nlon, nlev, state%a(:,1), state%a(:,2), state%a(:,3))
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """
    driver_mod = Module.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']

    field_offload = FieldOffloadTransformation(
        devptr_prefix='', offload_index='i', field_group_types=['state_type']
    )
    driver.apply(field_offload, role='driver', targets=['kernel_routine'])

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    kernel_call = next(c for c in calls if c.name=='kernel_routine')

    assert 'state_a' in driver.variable_map
    assert driver.variable_map['state_a'].type.shape == (':', ':', ':')

    assert kernel_call.arguments[:2] == ('nlon', 'nlev')
    assert kernel_call.arguments[2] == 'state_a(:,1,i)'
    assert kernel_call.arguments[3] == 'state_a(:,2,i)'
    assert kernel_call.arguments[4] == 'state_a(:,3,i)'

    assert len(calls) == 3
    assert calls[0].name == 'state%f_a%get_device_data_rdwr'
    assert calls[0].arguments == ('state_a',)
    assert calls[1] == kernel_call
    assert calls[2].name == 'state%f_a%sync_host_rdwr'
    assert calls[2].arguments == ()

    decls = FindNodes(ir.VariableDeclaration).visit(driver.spec)
    assert len(decls) == 5 if frontend == OMNI else 4
    assert decls[-1].symbols == ('state_a(:,:,:)',)


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_driver_compute(frontend, state_module, tmp_path):
    fcode = """
    module driver_mod
      use state_mod, only: state_type
      use parkind1, only: jprb
      implicit none

    contains

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i, ibl

        !$loki data
        !$loki loop
        do ibl=1,nlev
          call state%update_view(ibl)
          do i=1, nlon
            state%a(i, 1) = state%b(i, 1) + 0.1
            state%a(i, 2) = state%a(i, 1)
          end do

        end do
        !$loki end loop
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """
    driver_mod = Module.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].name == 'state%update_view'

    field_offload = FieldOffloadTransformation(
        devptr_prefix='', offload_index='ibl', field_group_types=['state_type']
    )
    driver.apply(field_offload, role='driver', targets=['kernel_routine'])

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 3
    assert calls[0].name == 'state%f_b%get_device_data_rdonly'
    assert calls[0].arguments == ('state_b',)
    assert calls[1].name == 'state%f_a%get_device_data_rdwr'
    assert calls[1].arguments == ('state_a',)
    assert calls[2].name == 'state%f_a%sync_host_rdwr'
    assert calls[2].arguments == ()

    assigns = FindNodes(ir.Assignment).visit(driver.body)
    assert len(assigns) == 2
    assert assigns[0].lhs == 'state_a(i,1,ibl)'
    assert assigns[0].rhs == 'state_b(i,1,ibl) + 0.1'
    assert assigns[1].lhs == 'state_a(i,2,ibl)'
    assert assigns[1].rhs == 'state_a(i,1,ibl)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_blocked(frontend, state_module, tmp_path):
    fcode = """
    module driver_mod
      use state_mod, only: state_type
      use parkind1, only: jprb
      use field_module, only: field_2rb, field_3rb
      implicit none

    contains

      subroutine kernel_routine(nlon, nlev, a, b, c)
        integer, intent(in)             :: nlon, nlev
        real(kind=jprb), intent(in)     :: a(nlon,nlev)
        real(kind=jprb), intent(inout)  :: b(nlon,nlev)
        real(kind=jprb), intent(out)    :: c(nlon,nlev)
        integer :: i, j

        do j=1, nlon
          do i=1, nlev
            b(i,j) = a(i,j) + 0.1
            c(i,j) = 0.1
          end do
        end do
      end subroutine kernel_routine

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i

        !$loki data
        !$loki driver-loop
        do i=1,nlev
            call state%update_view(i)
            call kernel_routine(nlon, nlev, state%a, state%b, state%c)
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """
    driver_mod = Module.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']
    deviceptr_prefix = 'loki_devptr_prefix_'
    driver.apply(FieldOffloadBlockedTransformation(devptr_prefix=deviceptr_prefix,
                                                   offload_index='i',
                                                   field_group_types=['state_type'],
                                                   block_size=100),
                 role='driver',
                 targets=['kernel_routine'])

    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_call = next(c for c in calls if c.name=='kernel_routine')

    # verify that field offloads are generated properly
    in_calls = [c for c in calls if 'get_device_data_force' in c.name.name.lower()]
    assert len(in_calls) == 3
    # verify that field sync host calls are generated properly
    sync_calls = [c for c in calls if 'sync_host_force' in c.name.name.lower()]
    assert len(sync_calls) == 2

    # verify that data offload pragmas remain
    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 3
    assert all(p.keyword=='loki' and p.content==c for p, c in zip(pragmas, ['data', 'driver-loop', 'end data']))

    # verify that new pointer variables are created and used in driver calls
    for var in ['state_a', 'state_b', 'state_c']:
        name = deviceptr_prefix + var
        assert name in driver.variable_map
        devptr = driver.variable_map[name]
        assert isinstance(devptr, sym.Array)
        assert len(devptr.shape) == 3
        assert devptr.name in (arg.name for arg in kernel_call.arguments)

@pytest.mark.parametrize('frontend', available_frontends())
def test_field_offload_blocked_async(frontend, state_module, tmp_path):
    fcode = """
    module driver_mod
      use state_mod, only: state_type
      use parkind1, only: jprb
      use field_module, only: field_2rb, field_3rb
      implicit none

    contains

      subroutine kernel_routine(nlon, nlev, a, b, c)
        integer, intent(in)             :: nlon, nlev
        real(kind=jprb), intent(in)     :: a(nlon,nlev)
        real(kind=jprb), intent(inout)  :: b(nlon,nlev)
        real(kind=jprb), intent(out)    :: c(nlon,nlev)
        integer :: i, j

        do j=1, nlon
          do i=1, nlev
            b(i,j) = a(i,j) + 0.1
            c(i,j) = 0.1
          end do
        end do
      end subroutine kernel_routine

      subroutine driver_routine(nlon, nlev, state)
        integer, intent(in)             :: nlon, nlev
        type(state_type), intent(inout) :: state
        integer                         :: i

        !$loki data
        !$loki driver-loop
        do i=1,nlev
            call state%update_view(i)
            call kernel_routine(nlon, nlev, state%a, state%b, state%c)
        end do
        !$loki end data

      end subroutine driver_routine
    end module driver_mod
    """
    driver_mod = Module.from_source(
        fcode, frontend=frontend, definitions=state_module, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']
    deviceptr_prefix = 'loki_devptr_prefix_'
    driver.apply(FieldOffloadBlockedTransformation(devptr_prefix=deviceptr_prefix,
                                                   offload_index='i',
                                                   field_group_types=['state_type'],
                                                   block_size=100,
                                                   asynchronous=True,
                                                   num_queues=3),
                 role='driver',
                 targets=['kernel_routine'])

    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_call = next(c for c in calls if c.name=='kernel_routine')

    # verify that field offloads are generated properly
    in_calls = [c for c in calls if 'get_device_data_force' in c.name.name.lower()]
    assert len(in_calls) == 3
    # verify that field sync host calls are generated properly
    sync_calls = [c for c in calls if 'sync_host_force' in c.name.name.lower()]
    assert len(sync_calls) == 2

    # verify that data offload pragmas remain
    pragmas = FindNodes(Pragma).visit(driver.body)
    assert len(pragmas) == 3
    assert all(p.keyword=='loki' and p.content==c for p, c in zip(pragmas,
                                                                  ['data async(loki_block_queue)',
                                                                   'driver-loop async(loki_block_queue)',
                                                                   'end data']))

    # verify that new pointer variables are created and used in driver calls
    for var in ['state_a', 'state_b', 'state_c']:
        name = deviceptr_prefix + var
        assert name in driver.variable_map
        devptr = driver.variable_map[name]
        assert isinstance(devptr, sym.Array)
        assert len(devptr.shape) == 3
        assert devptr.name in (arg.name for arg in kernel_call.arguments)
