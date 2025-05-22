# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from loki.frontend import available_frontends
from loki import Module, pprint, fgen
from loki.ir import FindNodes, nodes as ir
from loki.transformations.loop_blocking import split_loop


@pytest.mark.parametrize('frontend', available_frontends())
def test_field_blocking_simple_loop(tmp_path, frontend):
    fsource = """
    module driver_mod
      ! use state_mod, only: state_type
      ! use parkind1, only: jprb
      ! use field_module, only: field_2rb, field_3rb

      implicit none
      contains

      subroutine driver_routine (nlon, nlev, a, b, c)
        integer, intent(in) :: nlon
        integer, intent(in) :: nlev
        type(field_2rb), intent(inout) :: a, b, c

        real, pointer :: ptr_a(:,:), ptr_b(:,:), ptr_c(:,:)
        integer :: i
        integer :: j

        call a%get_device_data_rdwr(ptr_a)
        call b%get_device_data_rdwr(ptr_b)
        call c%get_device_data_rdwr(ptr_c)

        !$loki driver-loop
        do j=1,nlon
          do i=1,nlev
            ptr_b(i, j) = ptr_a(i, j) + 0.1
            ptr_c(i, j) = 0.1
          end do
        end do

        call a%sync_host_rdwr()
        call b%sync_host_rdwr()
        call c%sync_host_rdwr()

      end subroutine driver_routine

    end module driver_mod
    """
    driver_mod = Module.from_source(
        fsource, frontend=frontend, xmods=[tmp_path]
    )
    driver = driver_mod['driver_routine']
    loops = FindNodes(ir.Loop).visit(driver.ir)
    assert len(loops) == 2

    block_size = 100
    splitting_vars, inner_loop, outer_loop = split_loop(driver, loops[0], block_size)

    print('\n')
    print(pprint(driver.ir))
    print('\n')
    print(fgen(driver.ir))

