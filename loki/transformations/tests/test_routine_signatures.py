# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module, Subroutine
from loki.frontend import available_frontends
from loki.ir import FindNodes, CallStatement
from loki.transformations.routine_signatures import RemoveDuplicateArgs

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('pass_as_kwarg', (True, False))
@pytest.mark.parametrize('recurse_to_kernels', (True, False))
@pytest.mark.parametrize('rename_common', (True, False))
def test_utilities_remove_duplicate_args(tmp_path, frontend, pass_as_kwarg, recurse_to_kernels, rename_common):
    """
    Test lowering constant array indices
    """
    fcode_driver = f"""
subroutine driver(nlon,nlev,nb,var)
  use kernel_mod, only: kernel
  implicit none
  integer, intent(in) :: nlon,nlev,nb
  real, intent(inout) :: var(nlon,nlev,5,nb)
  integer :: ibl
  integer :: offset
  integer :: some_val
  integer :: loop_start, loop_end
  loop_start = 2
  loop_end = nb
  some_val = 0
  offset = 1
  !$omp test
  do ibl=loop_start, loop_end
    call kernel(nlon,nlev, &
      & {'var1=' if pass_as_kwarg else ''}var(:,:,1,ibl),&
      & {'var2=' if pass_as_kwarg else ''}var(:,:,1,ibl),&
      & {'another_var=' if pass_as_kwarg else ''}var(:,:,2:5,ibl),&
      & {'icend=' if pass_as_kwarg else ''}offset,&
      & {'lstart=' if pass_as_kwarg else ''}loop_start,&
      & {'lend=' if pass_as_kwarg else ''}loop_end,&
      & {'kend=' if pass_as_kwarg else ''}nlev)
    call kernel(nlon,nlev, &
      & {'var1=' if pass_as_kwarg else ''}var(:,:,1,ibl),&
      & {'var2=' if pass_as_kwarg else ''}var(:,:,1,ibl),&
      & {'another_var=' if pass_as_kwarg else ''}var(:,:,2:5,ibl),&
      & {'icend=' if pass_as_kwarg else ''}offset,&
      & {'lstart=' if pass_as_kwarg else ''}loop_start,&
      & {'lend=' if pass_as_kwarg else ''}loop_end,&
      & {'kend=' if pass_as_kwarg else ''}nlev)
  enddo
end subroutine driver
"""

    fcode_kernel = """
module kernel_mod
implicit none
contains
subroutine kernel(nlon,nlev,var1,var2,another_var,icend,lstart,lend,kend)
  use compute_mod, only: compute
  implicit none
  integer, intent(in) :: nlon,nlev,icend,lstart,lend,kend
  real, intent(inout) :: var1(nlon,nlev)
  real, intent(inout) :: var2(nlon,nlev)
  real, intent(inout) :: another_var(nlon,nlev,4)
  integer :: jk, jl, jt
  var1(:,:) = 0.
  do jk = 1,kend
    do jl = 1, nlon
      var1(jl, jk) = 0.
      var2(jl, jk) = 1.0
      do jt= 1,4
        another_var(jl, jk, jt) = 0.0
      end do
    end do
  end do
  call compute(nlon,nlev,var1, var2)
  call compute(nlon,nlev,var1, var2)
end subroutine kernel
end module kernel_mod
"""

    fcode_nested_kernel = """
module compute_mod
implicit none
contains
subroutine compute(nlon,nlev,b_var,a_var)
  implicit none
  integer, intent(in) :: nlon,nlev
  real, intent(inout) :: b_var(nlon,nlev)
  real, intent(inout) :: a_var(nlon,nlev)
  real :: VAR ! create name clash on purpose (if rename_common)
  b_var(:,:) = 0.
  a_var(:,:) = 1.0
end subroutine compute
end module compute_mod
"""

    nested_kernel_mod = Module.from_source(fcode_nested_kernel, frontend=frontend, xmods=[tmp_path])
    kernel_mod = Module.from_source(fcode_kernel, frontend=frontend, definitions=nested_kernel_mod, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=kernel_mod, xmods=[tmp_path])

    transformation = RemoveDuplicateArgs(recurse_to_kernels=recurse_to_kernels, rename_common=rename_common)
    transformation.apply(driver, role='driver', targets=('kernel',))
    transformation.apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    transformation.apply(nested_kernel_mod['compute'], role='kernel')

    # driver
    kernel_var_name = 'var' if rename_common else 'var1'
    kernel_calls = FindNodes(CallStatement).visit(driver.body)
    for kernel_call in kernel_calls:
        if pass_as_kwarg:
            assert (kernel_var_name, 'var(:, :, 1, ibl)') in kernel_call.kwarguments
            assert ('var2', 'var(:, :, 1, ibl)') not in kernel_call.kwarguments
            arg1 = kernel_call.kwarguments[0][1]
            arg2 = kernel_call.kwarguments[1][1]
        else:
            assert 'var(:, :, 1, ibl)' in kernel_call.arguments
            assert 'var2(:, :, 1, ibl)' not in kernel_call.arguments
            arg1 = kernel_call.arguments[2]
            arg2 = kernel_call.arguments[3]
        assert arg1.dimensions == (':', ':', '1', 'ibl')
        assert arg2.dimensions == (':', ':', '2:5', 'ibl')
    # kernel
    kernel_vars = kernel_mod['kernel'].variable_map
    kernel_args = kernel_mod['kernel']._dummies
    assert kernel_var_name in kernel_args
    assert 'var2' not in kernel_args
    assert 'var2' not in kernel_vars
    assert kernel_vars[kernel_var_name].shape == ('nlon', 'nlev')
    assert kernel_vars['another_var'].dimensions == ('nlon', 'nlev', 4)
    compute_calls = FindNodes(CallStatement).visit(kernel_mod['kernel'].body)
    for compute_call in compute_calls:
        assert kernel_var_name in compute_call.arguments
        assert 'var2' not in compute_call.arguments
    # nested_kernel
    nested_kernel = nested_kernel_mod['compute']
    nested_kernel_vars = nested_kernel.variable_map
    nested_kernel_args = [arg.name.lower() for arg in nested_kernel.arguments]
    # it's always 'b_var' as a rename would clash with the already "used" variable "var"
    nested_kernel_var_name = 'b_var'
    if recurse_to_kernels:
        assert nested_kernel_var_name in nested_kernel_args
        assert 'a_var' not in nested_kernel_args
        assert nested_kernel_var_name in nested_kernel_vars
        assert 'a_var' not in nested_kernel_vars
    else:
        assert 'b_var' in nested_kernel_args
        assert 'a_var' in nested_kernel_args
        assert 'b_var' in nested_kernel_vars
        assert 'a_var' in nested_kernel_vars
