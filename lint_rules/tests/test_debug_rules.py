# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
import pytest

from conftest import run_linter, available_frontends
from loki import Sourcefile
from loki.lint import DefaultHandler


pytestmark = pytest.mark.skipif(not available_frontends(),
                                reason='Suitable frontend not available')


@pytest.fixture(scope='module', name='rules')
def fixture_rules():
    rules = importlib.import_module('lint_rules.debug_rules')
    return rules


@pytest.mark.parametrize('frontend', available_frontends())
def test_arg_size_array_slices(rules, frontend):
    """
    Test for argument size mismatch when arguments are passed as array slices.
    """

    fcode_driver = """
subroutine driver(klon, klev, nblk, var0, var1, var2, var3, var4, var5, &
                  var6, var7)
use yomhook, only : lhook,   dr_hook, jphook
implicit none

integer, intent(in) :: klon, klev, nblk
real, intent(in) :: var2(:,:), var4(:,:), var5(:,:), var3(klon, 137), var5(klon, 138)
real, intent(in) :: var6(:,:), var7(:,:)
real, intent(inout) :: var0(klon, nblk), var1(klon, 138, nblk)
real(kind=jphook) :: zhook_handle
integer :: klev, ibl

if(lhook) call dr_hook('driver', 0, zhook_handle)

associate(nlev => klev)
nlev = 137
do ibl = 1, nblk
   call kernel(klon, nlev, var0(:,ibl), var1(:,:,ibl), var2(1:klon, 1:nlev), &
               var3, var4(1:klon, 1:nlev+1), var5(:, 1:nlev+1), &
               var6_d=var6, var7_d=var7(:,1:nlev))
enddo
end associate

if(lhook) call dr_hook('driver', 1, zhook_handle)
end subroutine driver
    """.strip()

    fcode_kernel = """
subroutine kernel(klon, klev, var0_d, var1_d, var2_d, var3_d, var4_d, var5_d, var6_d, var7_d)
use yomhook, only : lhook,   dr_hook, jphook
implicit none
integer, intent(in) :: klon, klev
real, dimension(klon, klev), intent(inout) :: var0_d, var1_d
real, dimension(klon, klev), intent(in) :: var2_d, var3_d, var4_d
real, dimension(klon, klev+1), intent(in) :: var5_d
real, intent(in) :: var6_d(klon, klev), var7_d(klon, klev)
real(kind=jphook) :: zhook_handle

if(lhook) call dr_hook('kernel', 0, zhook_handle)
if(lhook) call dr_hook('kernel', 1, zhook_handle)
end subroutine kernel
    """.strip()

    driver_source = Sourcefile.from_source(fcode_driver, frontend=frontend)
    kernel_source = Sourcefile.from_source(fcode_kernel, frontend=frontend)

    driver = driver_source['driver']
    kernel = kernel_source['kernel']
    driver.enrich_calls([kernel,])

    messages = []
    handler = DefaultHandler(target=messages.append)
    _ = run_linter(driver_source, [rules.ArgSizeMismatchRule], handlers=[handler], targets=['kernel',])

    for msg in messages:
        print(msg)
    assert len(messages) == 3
    keyword = 'ArgSizeMismatchRule'
    assert all(keyword in msg for msg in messages)

    args = ('var0', 'var1', 'var4')
    for msg, ref_arg in zip(messages, args):
        assert f'arg: {ref_arg}' in msg
        assert f'dummy_arg: {ref_arg}_d' in msg


@pytest.mark.parametrize('frontend', available_frontends())
def test_arg_size_array_sequence(rules, frontend):
    """
    Test for argument size mismatch when arguments are passed as array sequences.
    """

    fcode_driver = """
subroutine driver(klon, klev, nblk, var0, var1, var2, var3)
use yomhook, only : lhook,   dr_hook, jphook
implicit none

integer, intent(in) :: klon, klev, nblk
real, intent(inout) :: var0(klon, nblk), var1(klon, 138, nblk)
real, intent(in) ::  var2(klon, 137), var3(klon*137)
real(kind=jphook) :: zhook_handle
real, dimension(klon, 137) :: var4, var5
real :: var6
integer :: klev, ibl

if(lhook) call dr_hook('driver', 0, zhook_handle)

klev = 137
do ibl = 1, nblk
   call kernel(klon, klev, var0(1,ibl), var1(1,1,ibl), var2(1, 1), var3(1), &
               var4(1, 1), var5, var6, 1, .true.)
enddo

if(lhook) call dr_hook('driver', 1, zhook_handle)
end subroutine driver
    """.strip()

    fcode_kernel = """
subroutine kernel(klon, klev, var0_d, var1_d, var2_d, var3_d, var4_d, var5_d, var6_d, &
                  int_arg, log_arg)
use yomhook, only : lhook,   dr_hook, jphook
implicit none
integer, intent(in) :: klon, klev
real, dimension(klon, klev), intent(inout) :: var0_d, var1_d
real, dimension(klon, klev), intent(in) :: var2_d, var3_d
real, intent(out) :: var4_d, var5_d, var6_d(klon, klev)
integer, intent(out) :: int_arg
logical, intent(out) :: log_arg
real(kind=jphook) :: zhook_handle

if(lhook) call dr_hook('kernel', 0, zhook_handle)
if(lhook) call dr_hook('kernel', 1, zhook_handle)
end subroutine kernel
    """.strip()

    driver_source = Sourcefile.from_source(fcode_driver, frontend=frontend)
    kernel_source = Sourcefile.from_source(fcode_kernel, frontend=frontend)

    driver = driver_source['driver']
    kernel = kernel_source['kernel']
    driver.enrich_calls([kernel,])

    messages = []
    handler = DefaultHandler(target=messages.append)
    _ = run_linter(driver_source, [rules.ArgSizeMismatchRule], handlers=[handler], targets=['kernel',])

    assert len(messages) == 4
    keyword = 'ArgSizeMismatchRule'
    assert all(keyword in msg for msg in messages)

    args = ('var0', 'var1', 'var5', 'var6')
    for msg, ref_arg in zip(messages, args):
        assert f'arg: {ref_arg}' in msg
        assert f'dummy_arg: {ref_arg}_d' in msg
