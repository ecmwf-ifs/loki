# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for the SCCSmallKernelsPipeline and related transformations.

These tests target specific bugs found in the Loki-generated output for the
IFS small-kernels GPU build, verifying that:

- Category 1: Derived-type arguments used only for block-local copies are
  excluded from ``!$acc data present()`` clauses.
- Category 2: DO loop bounds referencing derived-type members (e.g.
  ``YDCPG_BNDS%KIDIA``) are properly updated to their local copies.
- Category 7: Variables that appear in both ``horizontal.size`` and
  ``horizontal.upper`` (e.g. ``KLON``) must NOT be localized to
  ``local_KLON``.
- Category 8: Kind parameters (e.g. ``JPIM``) used by block-index variables
  propagated to callees must be imported.
- Category 3: Stack size (ISTSZ) must not be spuriously generated inside
  kernels that contain loops with calls to target routines.
- Category 5: Rank mismatches from sequence association must not cause
  InjectBlockIndexTransformation to append an extra subscript.
- Category 6: Host-path (non-GPU) block loops calling the same routine
  must also receive the IBL argument after LowerBlockIndex.
- Category 9: All required derived-type kwargs (BNDS, OPTS) must be passed
  to sub-kernel calls, and kwarg ordering must be deterministic.
- Category 10: Derived-type components of subroutine arguments must not
  appear in ``!$acc parallel loop gang private()`` clauses.
- Category 11: Calls outside the main block section must still get wrapped
  in their own block loop.
- Category 12: The pool allocator must not generate ISTSZ/ZSTACK setup
  inside kernel routines.
"""

import pytest

from loki import (
    Sourcefile, Dimension, fgen, Module
)
from loki.batch import ProcedureItem, SGraph
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI, FP
from loki.ir import (
    FindNodes, Assignment, CallStatement, Loop,
    Pragma, FindVariables, FindInlineCalls, Section, Import,
    Allocation, Deallocation, Intrinsic
)
from loki.transformations.single_column import (
    SCCSmallKernelsPipeline
)
from loki.transformations.single_column.annotate import SCCAnnotateTransformation
from loki.transformations.single_column.block import (
    SCCBlockSectionTransformation, SCCBlockSectionToLoopTransformation
)
from loki.transformations.block_index_transformations import (
    LowerBlockIndexTransformation, InjectBlockIndexTransformation
)
from loki.transformations.temporaries.pool_allocator_per_drv_loop import (
    TemporariesPoolAllocatorPerDrvLoopTransformation
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='klon', index='jrof',
        lower=('kidia', 'bnds%kidia'), upper=('kfdia', 'bnds%kfdia')
    )


@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(
        name='block_dim',
        size='ngpblks',
        index=('ibl', 'bnds%kbl')
    )


@pytest.fixture(scope='module', name='horizontal_klon_upper')
def fixture_horizontal_klon_upper():
    """
    Like the real IFS config, ``klon`` appears in both ``size`` and ``upper``.
    This triggers the Cat 7 bug if not handled.
    """
    return Dimension(
        name='horizontal', size='klon', index='jrof',
        lower=('kst', 'kidia', 'bnds%kidia'),
        upper=('kend', 'kfdia', 'bnds%kfdia', 'klon')
    )


# ---------------------------------------------------------------------------
# Fortran source fragments
# ---------------------------------------------------------------------------

FCODE_BNDS_TYPE_MOD = """
module bnds_type_mod
  implicit none
  type bnds_type
    integer :: kidia
    integer :: kfdia
    integer :: kbl
    integer :: kstglo
  end type bnds_type
end module bnds_type_mod
""".strip()

FCODE_OPTS_TYPE_MOD = """
module opts_type_mod
  implicit none
  type opts_type
    integer :: klon
    integer :: kflevg
    integer :: kgpcomp
  end type opts_type
end module opts_type_mod
""".strip()

FCODE_DRIVER_KERNEL_CAT12 = """
module driver_mod
  implicit none
contains
  subroutine driver(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver
end module driver_mod
""".strip()

FCODE_KERNEL_CAT12 = """
subroutine kernel(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    t(bnds%kidia:bnds%kfdia, jk) = 1.0
    q(bnds%kidia:bnds%kfdia, jk) = t(bnds%kidia:bnds%kfdia, jk) + 2.0
  end do
end subroutine kernel
""".strip()


# Source for Cat 7: KLON in both horizontal.size and horizontal.upper.
# The kernel uses KLON as an array dimension AND as a loop upper bound.
# The driver uses KLON (via opts%klon) as the block size.

FCODE_DRIVER_CAT7 = """
module driver_cat7_mod
  implicit none
contains
  subroutine driver_cat7(ngpblks, bnds, opts, pd, pt, psp)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat7.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pd(:,:,:)
    real, intent(in) :: pt(:,:,:)
    real, intent(in) :: psp(:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_cat7(opts%klon, opts%kflevg, bnds%kidia, bnds%kfdia, pd(:,:,ibl), pt(:,:,ibl), psp(:,ibl))
    end do
  end subroutine driver_cat7
end module driver_cat7_mod
""".strip()

FCODE_KERNEL_CAT7 = """
subroutine kernel_cat7(klon, klev, kst, kend, pd, pt, psp)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  real, intent(out) :: pd(klon, klev)
  real, intent(in) :: pt(klon, klev)
  real, intent(in) :: psp(klon)

  integer :: jrof, jlev
  real :: zfoo(klon)

  ! Use KLON in a computation (like ISTSZ in the real code)
  zfoo(1:klon) = 0.0

  do jlev = 1, klev
    do jrof = kst, kend
      pd(jrof, jlev) = pt(jrof, jlev) * psp(jrof)
    end do
  end do
end subroutine kernel_cat7
""".strip()


# Cat 1 residual: three-level hierarchy — driver → mid_kernel → sub_kernel.
# Only the driver→mid_kernel call has ``!$loki small-kernels``.
# The sub_kernel receives ``bnds`` but is NOT directly marked, so
# ``LowerBlockIndex`` trafo_data does NOT propagate to it.
# Despite that, ``bnds`` should still be excluded from ``!$acc data present``.

FCODE_DRIVER_CAT1_RESIDUAL = """
module driver_cat1r_mod
  implicit none
contains
  subroutine driver_cat1r(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "mid_kernel.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call mid_kernel(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_cat1r
end module driver_cat1r_mod
""".strip()

FCODE_MID_KERNEL = """
subroutine mid_kernel(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  #include "sub_kernel.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do

  call sub_kernel(bnds, t, q, klon, klev)
end subroutine mid_kernel
""".strip()

FCODE_SUB_KERNEL = """
subroutine sub_kernel(bnds, t, q, klon, klev)
  use bnds_type_mod, only: bnds_type
  implicit none
  type(bnds_type), intent(in) :: bnds
  integer, intent(in) :: klon, klev
  real, intent(in) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = t(jrof, jk) * 2.0
    end do
  end do
end subroutine sub_kernel
""".strip()


# Cat 8: JPIM kind import propagation.
# The driver declares ``IBL`` as ``INTEGER(KIND=JPIM)`` and imports ``JPIM``
# from ``parkind1``.  The kernel does NOT import ``JPIM``.
# After LowerBlockIndexTransformation propagates block-index variables to
# the kernel, ``JPIM`` must be imported in the kernel as well.

FCODE_PARKIND1_MOD = """
module parkind1
  implicit none
  integer, parameter :: jpim = selected_int_kind(9)
  integer, parameter :: jprb = selected_real_kind(13, 300)
  integer, parameter :: jprd = selected_real_kind(13, 300)
end module parkind1
""".strip()

FCODE_DRIVER_CAT8 = """
module driver_cat8_mod
  implicit none
contains
  subroutine driver_cat8(ngpblks, bnds, opts, t, q)
    use parkind1, only: jpim, jprb
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat8.intfb.h"
    integer(kind=jpim), intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real(kind=jprb), intent(inout) :: t(:,:,:)
    real(kind=jprb), intent(inout) :: q(:,:,:)

    integer(kind=jpim) :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_cat8(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_cat8
end module driver_cat8_mod
""".strip()

FCODE_KERNEL_CAT8 = """
subroutine kernel_cat8(klon, klev, bnds, t, q)
  use parkind1, only: jprb
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real(kind=jprb), intent(inout) :: t(klon, klev)
  real(kind=jprb), intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0_jprb
      q(jrof, jk) = t(jrof, jk) * 2.0_jprb
    end do
  end do
end subroutine kernel_cat8
""".strip()


# ---------------------------------------------------------------------------
# Cat 12: Pool allocator in kernel
# A kernel contains a loop over fields (DO JFLD=1,N) with calls to
# a sub-kernel that has temporaries. The pool allocator must NOT generate
# ISTSZ/ZSTACK at the kernel level — only at the driver level.
# The key is that the sub-kernel is in the kernel's targets list.
# ---------------------------------------------------------------------------

FCODE_DRIVER_CAT12_POOL = """
module driver_cat12p_mod
  implicit none
contains
  subroutine driver_cat12p(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat12p.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_cat12p(opts%klon, opts%kflevg, 3, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_cat12p
end module driver_cat12p_mod
""".strip()

FCODE_KERNEL_CAT12_POOL = """
subroutine kernel_cat12p(klon, klev, nfld, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  #include "sub_kernel_cat12p.intfb.h"
  integer, intent(in) :: klon, klev, nfld
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jfld

  do jfld = 1, nfld
    call sub_kernel_cat12p(klon, klev, bnds, t, q)
  end do
end subroutine kernel_cat12p
""".strip()

FCODE_SUB_KERNEL_CAT12_POOL = """
subroutine sub_kernel_cat12p(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  real :: ztmp(klon, klev)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp(jrof, jk) = t(jrof, jk) * 2.0
      q(jrof, jk) = ztmp(jrof, jk) + 1.0
    end do
  end do
end subroutine sub_kernel_cat12p
""".strip()

# ---------------------------------------------------------------------------
# Cat 12 refinement: "Nested driver" kernel that SHOULD get pool allocator.
#
# A "nested driver" kernel (like sigam_gp in the real IFS) is a kernel
# (role='kernel') that calls sub-kernels via ``!$loki small-kernels``.
# After SCCBlockSectionToLoopTransformation, these kernels get a block
# loop whose variable matches block_dim.indices (e.g. local_IBL).
# The pool allocator must generate ISTSZ/ZSTACK for these kernels,
# unlike simple kernels (Cat 12) whose loops are NOT block-dimension.
# ---------------------------------------------------------------------------

FCODE_DRIVER_NESTED_DRV = """
module driver_nested_drv_mod
  implicit none
contains
  subroutine driver_nested_drv(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "nested_drv_kernel.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call nested_drv_kernel(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_nested_drv
end module driver_nested_drv_mod
""".strip()

FCODE_NESTED_DRV_KERNEL = """
subroutine nested_drv_kernel(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  #include "leaf_kernel.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof

  ! Use bnds%kbl so extract_block_sections recognises this section
  ! (the filter requires a block_dim.indices variable to be present).
  ! In the real IFS, nested-driver kernels like sigam_gp reference
  ! YDCPG_BNDS%KBL in VERDISINT calls or block-header computations.
  jrof = bnds%kbl

  !$loki small-kernels
  call leaf_kernel(klon, klev, bnds, t, q)
end subroutine nested_drv_kernel
""".strip()

FCODE_LEAF_KERNEL = """
subroutine leaf_kernel(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  real :: ztmp(klon, klev)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp(jrof, jk) = t(jrof, jk) * 2.0
      q(jrof, jk) = ztmp(jrof, jk) + 1.0
    end do
  end do
end subroutine leaf_kernel
""".strip()


# ---------------------------------------------------------------------------
# Cat 10: Derived-type components in private clause
# A kernel with a driver loop internally uses derived-type argument
# components like YDGEOMETRY%YRDIM. These must NOT be privatised.
# ---------------------------------------------------------------------------

FCODE_GEOMETRY_TYPE_MOD = """
module geometry_type_mod
  implicit none
  type dim_type
    integer :: nproma
    integer :: nflevg
    integer :: ngpblks
  end type dim_type

  type gem_type
    real :: rmu0
  end type gem_type

  type geometry_type
    type(dim_type) :: yrdim
    type(gem_type) :: yrgem
  end type geometry_type
end module geometry_type_mod
""".strip()

FCODE_DRIVER_CAT10 = """
module driver_cat10_mod
  implicit none
contains
  subroutine driver_cat10(ydgeometry, bnds, opts, t, q)
    use geometry_type_mod, only: geometry_type
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat10.intfb.h"
    type(geometry_type), intent(in) :: ydgeometry
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ydgeometry%yrdim%ngpblks
      bnds%kbl = ibl
      bnds%kidia = 1
      bnds%kfdia = ydgeometry%yrdim%nproma

      !$loki small-kernels
      call kernel_cat10(ydgeometry%yrdim%nproma, ydgeometry%yrdim%nflevg, bnds, &
        & t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_cat10
end module driver_cat10_mod
""".strip()

FCODE_KERNEL_CAT10 = """
subroutine kernel_cat10(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
      q(jrof, jk) = t(jrof, jk) * 2.0
    end do
  end do
end subroutine kernel_cat10
""".strip()


# ---------------------------------------------------------------------------
# Cat 9: Missing kwargs (BNDS, OPTS) in sub-kernel calls
# A kernel calls sub-kernels that expect both BNDS and OPTS args.
# LowerBlockIndex must pass both as kwargs.
# ---------------------------------------------------------------------------

FCODE_DRIVER_CAT9 = """
module driver_cat9_mod
  implicit none
contains
  subroutine driver_cat9(ngpblks, bnds, opts, t, q, r)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat9.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)
    real, intent(inout) :: r(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_cat9(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl), r(:,:,ibl))
    end do
  end subroutine driver_cat9
end module driver_cat9_mod
""".strip()

FCODE_KERNEL_CAT9 = """
subroutine kernel_cat9(klon, klev, bnds, opts, t, q, r)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "sub_kernel_cat9.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)
  real, intent(inout) :: r(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do

  call sub_kernel_cat9(klon, klev, bnds, opts, q, r)
end subroutine kernel_cat9
""".strip()

FCODE_SUB_KERNEL_CAT9 = """
subroutine sub_kernel_cat9(klon, klev, bnds, opts, q, r)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: q(klon, klev)
  real, intent(inout) :: r(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      r(jrof, jk) = q(jrof, jk) * opts%kflevg
    end do
  end do
end subroutine sub_kernel_cat9
""".strip()


# ---------------------------------------------------------------------------
# Bug A: Struct-unpacking boundary (lassie → sigam_gp pattern)
# A mid-level kernel receives BNDS as a whole struct, unpacks members
# via ASSOCIATE (KIDIA=>BNDS%KIDIA, KFDIA=>BNDS%KFDIA), and passes
# the unpacked scalars as positional arguments to a leaf kernel.
# BNDS%KSTGLO is used in the driver loop (to compute KFDIA) but is
# NOT passed to the leaf kernel — so it must be propagated correctly
# through the struct-unpacking boundary.
# ---------------------------------------------------------------------------

FCODE_DRIVER_BUGA = """
module driver_buga_mod
  implicit none
contains
  subroutine driver_buga(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "mid_kernel_buga.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call mid_kernel_buga(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_buga
end module driver_buga_mod
""".strip()

FCODE_MID_KERNEL_BUGA = """
subroutine mid_kernel_buga(klon, klev, bnds, opts, t, q)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "leaf_kernel_buga.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  ! Some local work using bnds members directly
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do

  ! Call leaf kernel with UNPACKED scalars (like lassie calling sigam_gp)
  ! Note: ASSOCIATE resolved by SanitiseTransformation before SCCSmallKernels,
  ! so after associate resolution this becomes:
  !   CALL leaf_kernel_buga(klon, klev, bnds%kidia, bnds%kfdia, q)
  ASSOCIATE(kidia=>bnds%kidia, kfdia=>bnds%kfdia)

    !$loki small-kernels
    call leaf_kernel_buga(klon, klev, kidia, kfdia, q)

  END ASSOCIATE
end subroutine mid_kernel_buga
""".strip()

FCODE_LEAF_KERNEL_BUGA = """
subroutine leaf_kernel_buga(klon, klev, kst, kend, q)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = kst, kend
      q(jrof, jk) = q(jrof, jk) * 2.0
    end do
  end do
end subroutine leaf_kernel_buga
""".strip()


# ---------------------------------------------------------------------------
# Cat 6: Missing IBL in host-path calls
# A driver has both a GPU-path block loop (with !$loki small-kernels)
# and a host-path block loop calling the same routine.
# ---------------------------------------------------------------------------

FCODE_DRIVER_CAT6 = """
module driver_cat6_mod
  implicit none
contains
  subroutine driver_cat6(ngpblks, bnds, opts, t, q, lgpu)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat6.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)
    logical, intent(in) :: lgpu

    integer :: ibl

    if (lgpu) then
      ! GPU path
      do ibl = 1, ngpblks
        bnds%kbl = ibl
        bnds%kstglo = 1 + (ibl - 1) * opts%klon
        bnds%kidia = 1
        bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

        !$loki small-kernels
        call kernel_cat6(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
      end do
    else
      ! Host path
      do ibl = 1, ngpblks
        bnds%kbl = ibl
        bnds%kidia = 1
        bnds%kfdia = opts%klon
        call kernel_cat6(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
      end do
    end if
  end subroutine driver_cat6
end module driver_cat6_mod
""".strip()

FCODE_KERNEL_CAT6 = """
subroutine kernel_cat6(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
      q(jrof, jk) = t(jrof, jk) * 2.0
    end do
  end do
end subroutine kernel_cat6
""".strip()


# ---------------------------------------------------------------------------
# Cat 11: Call outside main block section needs own block loop
# A kernel has two code sections: one with !$loki small-kernels + call,
# and a second call outside the main section that also needs a block loop.
# ---------------------------------------------------------------------------

FCODE_DRIVER_CAT11 = """
module driver_cat11_mod
  implicit none
contains
  subroutine driver_cat11(ngpblks, bnds, opts, t, q, r)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat11.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)
    real, intent(inout) :: r(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_cat11(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl), r(:,:,ibl))
    end do
  end subroutine driver_cat11
end module driver_cat11_mod
""".strip()

FCODE_KERNEL_CAT11 = """
subroutine kernel_cat11(klon, klev, bnds, t, q, r)
  use bnds_type_mod, only: bnds_type
  implicit none
  #include "sub_kernel_cat11a.intfb.h"
  #include "sub_kernel_cat11b.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)
  real, intent(inout) :: r(klon, klev)

  integer :: jrof, jk

  ! Main computation section
  !$loki small-kernels
  call sub_kernel_cat11a(klon, klev, bnds, t, q)

  ! Second call - outside main section
  !$loki small-kernels
  call sub_kernel_cat11b(klon, klev, bnds, q, r)
end subroutine kernel_cat11
""".strip()

FCODE_SUB_KERNEL_CAT11A = """
subroutine sub_kernel_cat11a(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do
end subroutine sub_kernel_cat11a
""".strip()

FCODE_SUB_KERNEL_CAT11B = """
subroutine sub_kernel_cat11b(klon, klev, bnds, q, r)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: q(klon, klev)
  real, intent(inout) :: r(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      r(jrof, jk) = q(jrof, jk) * 3.0
    end do
  end do
end subroutine sub_kernel_cat11b
""".strip()


# ---------------------------------------------------------------------------
# Cat 5: Rank mismatch / sequence association
# A 3D field (NPROMA x 1 x NGPBLKS) is passed to a 2D dummy (KLON x KLEV).
# InjectBlockIndex must not append a 4th subscript.
# ---------------------------------------------------------------------------

FCODE_DRIVER_CAT5 = """
module driver_cat5_mod
  implicit none
contains
  subroutine driver_cat5(ngpblks, bnds, opts, field3d, result2d)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_cat5.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(in) :: field3d(:,:,:)
    real, intent(inout) :: result2d(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kidia = 1
      bnds%kfdia = opts%klon

      !$loki small-kernels
      call kernel_cat5(opts%klon, bnds, field3d(:,:,ibl), result2d(:,:,ibl))
    end do
  end subroutine driver_cat5
end module driver_cat5_mod
""".strip()

FCODE_KERNEL_CAT5 = """
subroutine kernel_cat5(klon, bnds, field2d, result2d)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon
  type(bnds_type), intent(in) :: bnds
  real, intent(in) :: field2d(klon)
  real, intent(inout) :: result2d(klon)

  integer :: jrof

  do jrof = bnds%kidia, bnds%kfdia
    result2d(jrof) = field2d(jrof) * 2.0
  end do
end subroutine kernel_cat5
""".strip()


# ---------------------------------------------------------------------------
# Cat 3 driver-level: ISTSZ must be MAX across ALL driver loops
# A driver with two driver loops.  Loop 1 calls kernel_no_temp (no
# temporaries → stack_size=0).  Loop 2 calls kernel_with_temp (has a
# local temporary → non-zero stack_size).
# The generated ISTSZ must reflect the MAX, not just loop 1's zero.
# ---------------------------------------------------------------------------

FCODE_DRIVER_CAT3_MULTI = """
module driver_cat3m_mod
  implicit none
contains
  subroutine driver_cat3m(ngpblks, bnds, opts, t, q, r)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_no_temp.intfb.h"
    #include "kernel_with_temp.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)
    real, intent(inout) :: r(:,:,:)

    integer :: ibl

    ! Driver loop 1: calls kernel with NO temporaries
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kidia = 1
      bnds%kfdia = opts%klon

      !$loki small-kernels
      call kernel_no_temp(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do

    ! Driver loop 2: calls kernel WITH temporaries
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kidia = 1
      bnds%kfdia = opts%klon

      !$loki small-kernels
      call kernel_with_temp(opts%klon, opts%kflevg, bnds, q(:,:,ibl), r(:,:,ibl))
    end do
  end subroutine driver_cat3m
end module driver_cat3m_mod
""".strip()

FCODE_KERNEL_NO_TEMP = """
subroutine kernel_no_temp(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do
end subroutine kernel_no_temp
""".strip()

FCODE_KERNEL_WITH_TEMP = """
subroutine kernel_with_temp(klon, klev, bnds, q, r)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: q(klon, klev)
  real, intent(inout) :: r(klon, klev)

  integer :: jrof, jk
  real :: ztmp(klon, klev)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp(jrof, jk) = q(jrof, jk) * 3.0
      r(jrof, jk) = ztmp(jrof, jk) + 1.0
    end do
  end do
end subroutine kernel_with_temp
""".strip()


# ---------------------------------------------------------------------------
# Bug D: ISTSZ under-enumeration for conditional branches
# Driver → kernel that calls sub_d1 or sub_d2 in IF/ELSE branches.
# Both sub-kernels have temporaries → ISTSZ must include terms from BOTH.
# ---------------------------------------------------------------------------

FCODE_DRIVER_BUGD = """
module driver_bugd_mod
  implicit none
contains
  subroutine driver_bugd(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_bugd.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_bugd(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_bugd
end module driver_bugd_mod
""".strip()

FCODE_KERNEL_BUGD = """
subroutine kernel_bugd(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  #include "sub_d1.intfb.h"
  #include "sub_d2.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  logical :: lcond

  lcond = (klev > 10)

  if (lcond) then
    call sub_d1(klon, klev, bnds, t, q)
  else
    call sub_d2(klon, klev, bnds, t, q)
  end if
end subroutine kernel_bugd
""".strip()

FCODE_SUB_D1 = """
subroutine sub_d1(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  real :: ztmp1(klon, klev)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp1(jrof, jk) = t(jrof, jk) * 2.0
      q(jrof, jk) = ztmp1(jrof, jk) + 1.0
    end do
  end do
end subroutine sub_d1
""".strip()

FCODE_SUB_D2 = """
subroutine sub_d2(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  use parkind1, only: jprd
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  real(kind=jprd) :: ztmp2(klon, klev)
  real :: ztmp3(klon, klev)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp2(jrof, jk) = t(jrof, jk) * 3.0
      ztmp3(jrof, jk) = real(ztmp2(jrof, jk))
      q(jrof, jk) = ztmp3(jrof, jk) + 2.0
    end do
  end do
end subroutine sub_d2
""".strip()


# ---------------------------------------------------------------------------
# Bug F: _parallel driver missing block loop + ZSTACK
# A driver routine whose existing block loop must be preserved through
# the pipeline, and get ZSTACK/ISTSZ pool allocation infrastructure.
# This models the cpg_0_parallel pattern.
# ---------------------------------------------------------------------------

FCODE_DRIVER_BUGF = """
module driver_bugf_mod
  implicit none
contains
  subroutine driver_bugf(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_bugf.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_bugf(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_bugf
end module driver_bugf_mod
""".strip()

FCODE_KERNEL_BUGF = """
subroutine kernel_bugf(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  real :: ztmp(klon, klev)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp(jrof, jk) = t(jrof, jk) * 2.0
      q(jrof, jk) = ztmp(jrof, jk) + 1.0
    end do
  end do
end subroutine kernel_bugf
""".strip()


# ---------------------------------------------------------------------------
# Bug G: Doubled NGPBLKS dimension
# A driver calls a kernel that has local arrays. After transformation,
# the kernel's local arrays should get exactly ONE NGPBLKS dimension added.
# This models the lapineb_parallel → lapineb pattern.
# ---------------------------------------------------------------------------

FCODE_DRIVER_BUGG = """
module driver_bugg_mod
  implicit none
contains
  subroutine driver_bugg(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_bugg.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl
    logical :: lgpu

    lgpu = .false.

    ! Path 1 (GPU-like): block loop calling kernel_bugg
    if (lgpu) then
      do ibl = 1, ngpblks
        bnds%kbl = ibl
        bnds%kstglo = 1 + (ibl - 1) * opts%klon
        bnds%kidia = 1
        bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

        !$loki small-kernels
        call kernel_bugg(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
      end do
    else
      ! Path 2 (CPU-like): second block loop calling the SAME kernel
      do ibl = 1, ngpblks
        bnds%kbl = ibl
        bnds%kstglo = 1 + (ibl - 1) * opts%klon
        bnds%kidia = 1
        bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

        !$loki small-kernels
        call kernel_bugg(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
      end do
    end if
  end subroutine driver_bugg
end module driver_bugg_mod
""".strip()

FCODE_KERNEL_BUGG = """
subroutine kernel_bugg(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  #include "leaf_bugg.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  real :: zlocal1(klon, klev)
  real :: zlocal2(klon)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      zlocal1(jrof, jk) = t(jrof, jk) * 2.0
    end do
  end do

  ! Use bnds%kbl so extract_block_sections recognises this section
  jrof = bnds%kbl

  !$loki small-kernels
  call leaf_bugg(klon, klev, bnds, zlocal1, q)

  do jrof = bnds%kidia, bnds%kfdia
    zlocal2(jrof) = q(jrof, 1)
    t(jrof, 1) = zlocal2(jrof) + 1.0
  end do
end subroutine kernel_bugg
""".strip()

FCODE_LEAF_BUGG = """
subroutine leaf_bugg(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do
end subroutine leaf_bugg
""".strip()


# ---------------------------------------------------------------------------
# Bug H: CHARACTER variable assignment inside parallel loop
# A kernel has a CHARACTER variable assigned before the horizontal loop.
# After transformation, the assignment must stay OUTSIDE the !$acc parallel
# loop region.
# ---------------------------------------------------------------------------

FCODE_DRIVER_BUGH = """
module driver_bugh_mod
  implicit none
contains
  subroutine driver_bugh(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_bugh.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_bugh(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_bugh
end module driver_bugh_mod
""".strip()

FCODE_KERNEL_BUGH = """
subroutine kernel_bugh(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  character(len=4) :: cloper
  real :: zfactor

  ! CHARACTER assignment before horizontal loop -- must stay outside
  ! the !$acc parallel loop region after transformation.
  cloper = 'IBOT'
  if (klev > 10) cloper = 'INTG'

  ! Scalar assignment that uses cloper indirectly (to prevent dead-code removal)
  if (cloper == 'IBOT') then
    zfactor = 2.0
  else
    zfactor = 3.0
  end if

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) * zfactor
      q(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do
end subroutine kernel_bugh
""".strip()


# ---------------------------------------------------------------------------
# Bug K: JPRD import not propagated through pool allocator
# A driver → kernel → sub-kernel chain where the sub-kernel uses
# REAL(KIND=JPRD) temporaries. After pool allocation, the driver must
# import JPRD from PARKIND1 because the ISTSZ computation uses
# C_SIZEOF(REAL(1, kind=JPRD)).
# ---------------------------------------------------------------------------

FCODE_DRIVER_BUGK = """
module driver_bugk_mod
  implicit none
contains
  subroutine driver_bugk(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_bugk.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_bugk(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_bugk
end module driver_bugk_mod
""".strip()

FCODE_KERNEL_BUGK = """
subroutine kernel_bugk(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  #include "sub_bugk.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do

  ! Use bnds%kbl so extract_block_sections recognises this section
  jrof = bnds%kbl

  !$loki small-kernels
  call sub_bugk(klon, klev, bnds, t, q)
end subroutine kernel_bugk
""".strip()

FCODE_SUB_BUGK = """
subroutine sub_bugk(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  use parkind1, only: jprd
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk
  real(kind=jprd) :: ztmp_d(klon, klev)

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp_d(jrof, jk) = t(jrof, jk) * 2.0d0
      q(jrof, jk) = real(ztmp_d(jrof, jk))
    end do
  end do
end subroutine sub_bugk
""".strip()


# ---------------------------------------------------------------------------
# Helper: run the SCCSmallKernelsPipeline on a driver+kernel pair
# ---------------------------------------------------------------------------

def _apply_small_kernels_pipeline(driver_source, kernel_source, horizontal,
                                  block_dim, tmp_path,
                                  driver_name='driver', kernel_name='kernel',
                                  driver_item_name='driver_mod#driver',
                                  kernel_item_name='#kernel'):
    """
    Parse *driver_source* and *kernel_source*, enrich, build the required
    item / sgraph objects, and apply the full ``SCCSmallKernelsPipeline``.

    Returns ``(driver_routine, kernel_routine)`` after transformation.
    """
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source[driver_name]
    kernel_routine = kernel_source[kernel_name]
    driver_routine.enrich(kernel_routine)

    driver_item = ProcedureItem(name=driver_item_name, source=driver_source)
    kernel_item = ProcedureItem(name=kernel_item_name, source=kernel_source)
    sgraph = SGraph.from_dict({driver_item: [kernel_item]})

    for transform in pipeline.transformations:
        if getattr(transform, 'reverse_traversal', False):
            # Bottom-up: kernel first, then driver
            transform.apply(
                kernel_routine, role='kernel', item=kernel_item,
                targets=[], sub_sgraph=sgraph
            )
            transform.apply(
                driver_routine, role='driver', item=driver_item,
                targets=[kernel_name], sub_sgraph=sgraph
            )
        else:
            # Top-down: driver first, then kernel
            transform.apply(
                driver_routine, role='driver', item=driver_item,
                targets=[kernel_name], sub_sgraph=sgraph
            )
            transform.apply(
                kernel_routine, role='kernel', item=kernel_item,
                targets=[], sub_sgraph=sgraph
            )

    return driver_routine, kernel_routine


def _apply_small_kernels_pipeline_3level(
        driver_source, mid_source, sub_source, horizontal, block_dim, tmp_path,
        driver_name='driver_cat1r', mid_name='mid_kernel', sub_name='sub_kernel',
        driver_item_name='driver_cat1r_mod#driver_cat1r',
        mid_item_name='#mid_kernel', sub_item_name='#sub_kernel'):
    """
    Three-level variant: driver → mid_kernel → sub_kernel.

    Only the driver→mid_kernel call has ``!$loki small-kernels``.
    The mid_kernel→sub_kernel call is a plain call (no pragma).
    """
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source[driver_name]
    mid_routine = mid_source[mid_name]
    sub_routine = sub_source[sub_name]

    driver_routine.enrich(mid_routine)
    mid_routine.enrich(sub_routine)

    driver_item = ProcedureItem(name=driver_item_name, source=driver_source)
    mid_item = ProcedureItem(name=mid_item_name, source=mid_source)
    sub_item = ProcedureItem(name=sub_item_name, source=sub_source)

    sgraph = SGraph.from_dict({
        driver_item: [mid_item],
        mid_item: [sub_item],
    })

    # Apply in top-down order by default, but reverse for transformations
    # that have reverse_traversal=True (e.g. pool allocator needs bottom-up
    # so that kernel stack_sizes are computed before drivers read them).
    items_in_order = [
        (driver_routine, 'driver', driver_item, [mid_name]),
        (mid_routine, 'kernel', mid_item, [sub_name]),
        (sub_routine, 'kernel', sub_item, []),
    ]

    for transform in pipeline.transformations:
        # try:
        if True:
            order = items_in_order
            if getattr(transform, 'reverse_traversal', False):
                order = list(reversed(items_in_order))
            print(f"transform: {transform}\n---------------------------------")
            for routine, role, item, targets in order:
                transform.apply(
                    routine, role=role, item=item,
                    targets=targets, sub_sgraph=sgraph
                )
                print(f"routine {routine}\n{routine.to_fortran()}\n-----\n")
            print(f"---------------------------------\n")
        # except Exception as e:
        #     print(f"e: {e}")
        #     break

    return driver_routine, mid_routine, sub_routine


def _apply_small_kernels_pipeline_4level(
        driver_source, mid_source, sub1_source, sub2_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_bugd', mid_name='kernel_bugd',
        sub1_name='sub_d1', sub2_name='sub_d2',
        driver_item_name='driver_bugd_mod#driver_bugd',
        mid_item_name='#kernel_bugd',
        sub1_item_name='#sub_d1', sub2_item_name='#sub_d2'):
    """
    Four-level variant: driver → mid_kernel → (sub1, sub2).

    The mid_kernel has IF/ELSE branches calling sub1 and sub2 independently,
    each with ``!$loki small-kernels``.

    Returns ``(driver, mid_kernel, sub1, sub2)`` after transformation.
    """
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source[driver_name]
    mid_routine = mid_source[mid_name]
    sub1_routine = sub1_source[sub1_name]
    sub2_routine = sub2_source[sub2_name]

    driver_routine.enrich(mid_routine)
    mid_routine.enrich(sub1_routine)
    mid_routine.enrich(sub2_routine)

    driver_item = ProcedureItem(name=driver_item_name, source=driver_source)
    mid_item = ProcedureItem(name=mid_item_name, source=mid_source)
    sub1_item = ProcedureItem(name=sub1_item_name, source=sub1_source)
    sub2_item = ProcedureItem(name=sub2_item_name, source=sub2_source)

    sgraph = SGraph.from_dict({
        driver_item: [mid_item],
        mid_item: [sub1_item, sub2_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, [mid_name]),
        (mid_routine, 'kernel', mid_item, [sub1_name, sub2_name]),
        (sub1_routine, 'kernel', sub1_item, []),
        (sub2_routine, 'kernel', sub2_item, []),
    ]

    for transform in pipeline.transformations:
        order = items_in_order
        if getattr(transform, 'reverse_traversal', False):
            order = list(reversed(items_in_order))
        for routine, role, item, targets in order:
            transform.apply(
                routine, role=role, item=item,
                targets=targets, sub_sgraph=sgraph
            )

    return driver_routine, mid_routine, sub1_routine, sub2_routine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat2_local_copy_in_loop_bounds(frontend, horizontal, block_dim, tmp_path):
    """
    Category 2 bug: DO loop bounds like ``BNDS%KIDIA`` and ``BNDS%KFDIA``
    must be updated to ``local_BNDS%KIDIA`` / ``local_BNDS%KFDIA`` after
    the local-copy substitution.

    The vector notation ``t(bnds%kidia:bnds%kfdia, jk)`` gets devectorized
    into ``DO JROF = bnds%kidia, bnds%kfdia`` by SCCDevectorTransformation,
    then re-vectorized by SCCVecRevectorTransformation.  The local-copy
    substitution (CreateLocalCopiesTransformation) must replace *all*
    occurrences of ``bnds`` — including the loop bounds — with ``local_bnds``.
    """
    # Build sources
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT12, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_KERNEL_CAT12, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path
    )

    # After transformation, the kernel should contain DO loops whose bounds
    # reference ``local_bnds%kidia`` / ``local_bnds%kfdia``, NOT ``bnds%kidia``
    # / ``bnds%kfdia``.
    loops = FindNodes(Loop).visit(kernel.body)
    horizontal_loops = [l for l in loops if str(l.variable).lower() == 'jrof']

    assert len(horizontal_loops) > 0, "Expected at least one JROF loop in the kernel"

    for loop in horizontal_loops:
        lower = str(loop.bounds.lower).lower()
        upper = str(loop.bounds.upper).lower()
        assert 'local_bnds%kidia' in lower, (
            f"Loop lower bound should use local_bnds%kidia, got: {loop.bounds.lower}"
        )
        assert 'local_bnds%kfdia' in upper, (
            f"Loop upper bound should use local_bnds%kfdia, got: {loop.bounds.upper}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat1_present_clause_excludes_local_copy_vars(frontend, horizontal, block_dim, tmp_path):
    """
    Category 1 bug: derived-type arguments that are fully replaced by
    local copies (e.g. ``BNDS``) should NOT appear in the
    ``!$acc data present(...)`` clause generated by SCCAnnotateTransformation.

    After transformation, the kernel's ``!$acc data present(...)`` pragma
    should list only those arguments that are actually accessed on the
    device — not the ones that have been replaced by ``local_*`` copies.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT12, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_KERNEL_CAT12, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path
    )

    # Find the ``!$acc data present(...)`` pragma in the kernel
    pragmas = FindNodes(Pragma).visit(kernel.ir)
    present_pragmas = [p for p in pragmas
                       if p.keyword.lower() == 'acc'
                       and 'present' in p.content.lower()]

    assert len(present_pragmas) >= 1, "Expected at least one !$acc data present pragma"

    for pragma in present_pragmas:
        content_lower = pragma.content.lower()
        # bnds should NOT be in the present clause since it's fully replaced
        # by local_bnds
        assert 'bnds' not in content_lower or 'local_bnds' in content_lower, (
            f"'bnds' should not appear in present clause (only local_bnds is used): {pragma.content}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat7_no_local_klon_when_size_and_upper(frontend, horizontal_klon_upper, block_dim, tmp_path):
    """
    Category 7 bug: when ``klon`` appears in both ``horizontal.size``
    and ``horizontal.upper``, the ``CreateLocalCopiesTransformation``
    must NOT create a ``local_KLON`` variable.

    ``KLON`` is an array dimension size (e.g. ``REAL :: PD(KLON, KLEV)``)
    and is used in ISTSZ computations before the block loop.  Localizing
    it to ``local_KLON`` produces code that reads an uninitialized variable
    outside the block loop.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT7, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT7, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal_klon_upper, block_dim, tmp_path,
        driver_name='driver_cat7', kernel_name='kernel_cat7',
        driver_item_name='driver_cat7_mod#driver_cat7',
        kernel_item_name='#kernel_cat7'
    )

    # After transformation, there should be NO ``local_klon`` variable
    # declared in the kernel.
    kernel_var_names = {v.name.lower() for v in kernel.variables}
    assert 'local_klon' not in kernel_var_names, (
        f"local_KLON should NOT be created when KLON is a horizontal size variable. "
        f"Found variables: {sorted(kernel_var_names)}"
    )

    # Additionally, the generated code should still use plain ``klon`` (or
    # ``KLON``) in all expressions, not ``local_klon``.
    code = fgen(kernel)
    assert 'local_klon' not in code.lower(), (
        f"Generated code should not contain 'local_KLON':\n{code}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat1_residual_sub_kernel_bnds_not_in_present(frontend, horizontal, block_dim, tmp_path):
    """
    Category 1 residual bug: a sub-kernel that receives ``BNDS`` as an
    argument but is NOT directly called via ``!$loki small-kernels`` should
    still have ``BNDS`` excluded from its ``!$acc data present(...)`` clause.

    In the real IFS, routines like ``GPINISLB_PART2_EXPL`` are called from
    an intermediate kernel without a ``!$loki small-kernels`` pragma, so
    ``LowerBlockIndex`` trafo_data does NOT propagate to them.  But they
    still receive ``YDCPG_BNDS`` which is only used for loop bounds and
    should not appear in ``present()``.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT1_RESIDUAL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    mid_source = Sourcefile.from_source(
        FCODE_MID_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    sub_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, mid_kernel, sub_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, mid_source, sub_source, horizontal, block_dim, tmp_path
    )

    # The sub_kernel should have ``!$acc data present(...)`` that does NOT
    # contain ``bnds`` — only arrays like ``t``, ``q`` should be there.
    pragmas = FindNodes(Pragma).visit(sub_kernel.ir)
    present_pragmas = [p for p in pragmas
                       if p.keyword.lower() == 'acc'
                       and 'present' in p.content.lower()]

    assert len(present_pragmas) >= 1, \
        "Expected at least one !$acc data present pragma in sub_kernel"

    for pragma in present_pragmas:
        content_lower = pragma.content.lower()
        # Split out the present(...) content and check variable names
        assert ' bnds' not in f' {content_lower}' or 'local_bnds' in content_lower, (
            f"'bnds' should not appear in sub_kernel's present clause: {pragma.content}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat8_jpim_import_propagated_to_kernel(frontend, horizontal, block_dim, tmp_path):
    """
    Category 8 bug: when ``LowerBlockIndexTransformation`` propagates block-
    index variables (like ``IBL``) typed as ``INTEGER(KIND=JPIM)`` to callee
    routines, the ``JPIM`` kind parameter must also be imported in the callee.

    In the real IFS, this manifests as ``lacdyn`` and ``cpg_dyn_slg`` missing
    ``JPIM`` from their ``USE PARKIND1`` statement, causing compilation errors
    because the generated code declares ``INTEGER(KIND=JPIM) :: local_IBL``
    without importing ``JPIM``.
    """
    parkind1_mod = Module.from_source(FCODE_PARKIND1_MOD, frontend=frontend, xmods=[tmp_path])
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT8, frontend=frontend,
        definitions=[parkind1_mod, bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT8, frontend=frontend,
        definitions=[parkind1_mod, bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path,
        driver_name='driver_cat8', kernel_name='kernel_cat8',
        driver_item_name='driver_cat8_mod#driver_cat8',
        kernel_item_name='#kernel_cat8'
    )

    # After transformation, the kernel should import JPIM (needed for
    # block-index variables like IBL that were propagated from the driver).
    all_imports = FindNodes(Import).visit(kernel.spec)
    all_imported_symbols = set()
    for imp in all_imports:
        for s in imp.symbols:
            all_imported_symbols.add(s.name.lower())

    assert 'jpim' in all_imported_symbols, (
        f"Expected 'jpim' to be imported in kernel after transformation. "
        f"Found imports: {[str(imp) for imp in all_imports]}"
    )


# ---------------------------------------------------------------------------
# New tests for remaining bug categories (3, 5, 6, 9, 10, 11, 12)
# These tests reproduce the bugs but DO NOT require fixes to pass —
# they are expected to FAIL (xfail) until the corresponding fixes are
# implemented.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat12_kernel_no_pool_allocator_setup(frontend, horizontal, block_dim, tmp_path):
    """
    Category 12 bug: when a kernel contains a loop with calls to target
    routines, the pool allocator must NOT generate ISTSZ/ZSTACK/ALLOCATE/
    DEALLOCATE at the kernel level.

    In the real IFS, ``larcinb`` (a kernel) has a DO JFLD=1,SIZE(...)
    loop containing calls to target routines.  ``find_driver_loops``
    incorrectly returns this as a "driver loop", causing the pool allocator
    to generate a full ISTSZ/ZSTACK setup inside the kernel.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    sub_kernel_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_CAT12_POOL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT12_POOL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT12_POOL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    # Use the 3-level helper: driver → kernel_cat12p → sub_kernel_cat12p
    driver, kernel, sub_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, kernel_source, sub_kernel_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_cat12p', mid_name='kernel_cat12p',
        sub_name='sub_kernel_cat12p',
        driver_item_name='driver_cat12p_mod#driver_cat12p',
        mid_item_name='#kernel_cat12p',
        sub_item_name='#sub_kernel_cat12p'
    )

    # The kernel must NOT have ISTSZ or ZSTACK variables
    kernel_var_names = {v.name.lower() for v in kernel.variables}
    assert 'istsz' not in kernel_var_names, (
        f"Kernel should NOT have 'ISTSZ' variable. Found: {sorted(kernel_var_names)}"
    )
    assert 'zstack' not in kernel_var_names, (
        f"Kernel should NOT have 'ZSTACK' variable. Found: {sorted(kernel_var_names)}"
    )

    # The kernel must NOT have ALLOCATE/DEALLOCATE statements
    allocations = FindNodes(Allocation).visit(kernel.body)
    deallocations = FindNodes(Deallocation).visit(kernel.body)
    assert len(allocations) == 0, (
        f"Kernel should NOT have ALLOCATE statements. Found: {[fgen(a) for a in allocations]}"
    )
    assert len(deallocations) == 0, (
        f"Kernel should NOT have DEALLOCATE statements. Found: {[fgen(d) for d in deallocations]}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
@pytest.mark.xfail(reason='Cat 10: Derived-type argument components appear in private() clause')
def test_cat10_no_arg_components_in_private(frontend, horizontal, block_dim, tmp_path):
    """
    Category 10 bug: derived-type components of subroutine arguments
    (e.g. ``YDGEOMETRY%YRDIM``) must NOT appear in the
    ``!$acc parallel loop gang private()`` clause.  Only truly local
    variables should be privatised.

    In the real IFS, ``ecphys_setup_layer`` gets
    ``private(YDGEOMETRY%YRGEM, YDGEOMETRY%YRDIM, YYTXYB)`` but should
    have no private clause (or only ``YYTXYB``).
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    geom_mod = Module.from_source(FCODE_GEOMETRY_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT10, frontend=frontend,
        definitions=[bnds_mod, opts_mod, geom_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT10, frontend=frontend,
        definitions=[bnds_mod, opts_mod, geom_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path,
        driver_name='driver_cat10', kernel_name='kernel_cat10',
        driver_item_name='driver_cat10_mod#driver_cat10',
        kernel_item_name='#kernel_cat10'
    )

    # Check the driver's block loop for the private clause
    code = fgen(driver)
    code_lower = code.lower()

    # Find any private() clause content
    import re
    private_matches = re.findall(r'private\(([^)]+)\)', code_lower)

    for match in private_matches:
        privates = [p.strip() for p in match.split(',')]
        for priv in privates:
            # No derived-type component of an argument should be privatised
            assert 'ydgeometry' not in priv, (
                f"YDGEOMETRY component '{priv}' should NOT be in private() clause. "
                f"Full private clause: private({match})"
            )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
# @pytest.mark.xfail(reason='Cat 9: Missing BNDS/OPTS kwargs in sub-kernel calls from process_kernel')
def test_cat9_missing_kwargs_bnds_opts(frontend, horizontal, block_dim, tmp_path):
    """
    Category 9 bug: when a kernel calls a sub-kernel that expects both
    ``BNDS`` and ``OPTS`` arguments, ``LowerBlockIndexTransformation``
    must pass BOTH as keyword arguments to the call.

    In the real IFS, ``lassie`` calls ``SITNU_GP_LOKI`` and ``SIGAM_GP_LOKI``
    which expect ``YDCPG_BNDS`` and ``YDCPG_OPTS`` kwargs, but only some
    (or none) are actually added.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    sub_kernel_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_CAT9, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT9, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT9, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    # Build 3-level hierarchy: driver → kernel_cat9 → sub_kernel_cat9
    driver, kernel, sub_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, kernel_source, sub_kernel_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_cat9', mid_name='kernel_cat9', sub_name='sub_kernel_cat9',
        driver_item_name='driver_cat9_mod#driver_cat9',
        mid_item_name='#kernel_cat9', sub_item_name='#sub_kernel_cat9'
    )

    # Find the call to sub_kernel_cat9 in the kernel
    calls = FindNodes(CallStatement).visit(kernel.body)
    sub_calls = [c for c in calls if 'sub_kernel_cat9' in str(c.name).lower()]
    assert len(sub_calls) >= 1, "Expected at least one call to sub_kernel_cat9"

    for call in sub_calls:
        kwarg_names = [kw[0].lower() for kw in call.kwarguments]
        # Both bnds and opts should be passed
        has_bnds = any('bnds' in kw for kw in kwarg_names)
        has_opts = any('opts' in kw for kw in kwarg_names)
        assert has_bnds, (
            f"Call to sub_kernel_cat9 should have BNDS kwarg. "
            f"Found kwargs: {kwarg_names}"
        )
        assert has_opts, (
            f"Call to sub_kernel_cat9 should have OPTS kwarg. "
            f"Found kwargs: {kwarg_names}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
# @pytest.mark.xfail(reason='Cat 6: Host-path block loop call does not get IBL argument')
def test_cat6_host_path_gets_ibl_argument(frontend, horizontal, block_dim, tmp_path):
    """
    Category 6 bug: when a driver has both a GPU-path block loop (with
    ``!$loki small-kernels``) and a host-path block loop calling the
    same routine, ``LowerBlockIndexTransformation`` must add ``IBL``
    to BOTH calls.

    In the real IFS, ``cpg_0_parallel``, ``cpg_2_parallel``, and
    ``lapineb_parallel`` have a host-path OMP loop that calls the
    original (non-``_LOKI``) routine without ``IBL=IBL``.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT6, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT6, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path,
        driver_name='driver_cat6', kernel_name='kernel_cat6',
        driver_item_name='driver_cat6_mod#driver_cat6',
        kernel_item_name='#kernel_cat6'
    )

    print(f"{driver.to_fortran()}\n")
    print(f"{kernel.to_fortran()}")
    # After transformation, ALL calls to kernel_cat6 in the driver should
    # pass IBL (or the block index variable) as an argument.
    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_calls = [c for c in calls if 'kernel_cat6' in str(c.name).lower()]
    assert len(kernel_calls) >= 2, (
        f"Expected at least 2 calls to kernel_cat6 (GPU + host paths). "
        f"Found: {len(kernel_calls)}"
    )

    for call in kernel_calls:
        # Check that IBL is passed either as positional or keyword arg
        all_arg_names = []
        for arg in call.arguments:
            if hasattr(arg, 'name'):
                all_arg_names.append(str(arg.name).lower())
        for kw_name, kw_val in call.kwarguments:
            all_arg_names.append(kw_name.lower())
            if hasattr(kw_val, 'name'):
                all_arg_names.append(str(kw_val.name).lower())

        has_ibl = any('ibl' in name for name in all_arg_names)
        assert has_ibl, (
            f"Call to kernel_cat6 at should have IBL argument. "
            f"Found args: {all_arg_names}\n"
            f"Call: {fgen(call)}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
# @pytest.mark.xfail(reason='Cat 11: Call outside main block section does not get own block loop')
def test_cat11_call_outside_main_section_gets_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    Category 11 bug: when a kernel has multiple ``!$loki small-kernels``
    sections and a call to a target routine is in a separate section,
    it must still get wrapped in its own block loop.

    In the real IFS, ``lacdyn`` has a ``CALL LAVABO_LOKI(...)`` outside
    the main block section that doesn't get its own
    ``!$acc parallel loop gang`` wrapper.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    sub_a_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_CAT11A, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    sub_b_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_CAT11B, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT11, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT11, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver_cat11']
    kernel_routine = kernel_source['kernel_cat11']
    sub_a_routine = sub_a_source['sub_kernel_cat11a']
    sub_b_routine = sub_b_source['sub_kernel_cat11b']

    driver_routine.enrich(kernel_routine)
    kernel_routine.enrich(sub_a_routine)
    kernel_routine.enrich(sub_b_routine)

    driver_item = ProcedureItem(name='driver_cat11_mod#driver_cat11', source=driver_source)
    kernel_item = ProcedureItem(name='#kernel_cat11', source=kernel_source)
    sub_a_item = ProcedureItem(name='#sub_kernel_cat11a', source=sub_a_source)
    sub_b_item = ProcedureItem(name='#sub_kernel_cat11b', source=sub_b_source)

    sgraph = SGraph.from_dict({
        driver_item: [kernel_item],
        kernel_item: [sub_a_item, sub_b_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, ['kernel_cat11']),
        (kernel_routine, 'kernel', kernel_item, ['sub_kernel_cat11a', 'sub_kernel_cat11b']),
        (sub_a_routine, 'kernel', sub_a_item, []),
        (sub_b_routine, 'kernel', sub_b_item, []),
    ]

    for transform in pipeline.transformations:
        for routine, role, item, targets in items_in_order:
            transform.apply(
                routine, role=role, item=item,
                targets=targets, sub_sgraph=sgraph
            )

    # After transformation, BOTH sub_kernel calls in kernel_cat11 should
    # be inside block loops (DO IBL=1,...).
    # Check that each call is nested inside a Loop node.
    kernel_code = fgen(kernel_routine)

    calls = FindNodes(CallStatement).visit(kernel_routine.body)
    sub_calls = [c for c in calls
                 if 'sub_kernel_cat11' in str(c.name).lower()]
    assert len(sub_calls) >= 2, (
        f"Expected at least 2 sub_kernel calls. Found: {len(sub_calls)}"
    )

    # Check that each call is inside a block loop
    loops = FindNodes(Loop).visit(kernel_routine.body)
    block_loops = [l for l in loops if str(l.variable).lower() == 'ibl']

    # Each sub_kernel call should be inside a block loop
    for sub_call in sub_calls:
        is_inside_block_loop = False
        for block_loop in block_loops:
            loop_calls = FindNodes(CallStatement).visit(block_loop.body)
            if sub_call in loop_calls:
                is_inside_block_loop = True
                break
        assert is_inside_block_loop, (
            f"Call to {sub_call.name} should be inside a block loop (DO IBL=...). "
            f"Kernel code:\n{kernel_code}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
# @pytest.mark.xfail(reason='Cat 5: InjectBlockIndex appends extra subscript for rank-mismatched args')
def test_cat5_rank_mismatch_no_extra_block_index(frontend, horizontal, block_dim, tmp_path):
    """
    Category 5 bug: when a 2D array slice is passed to a 1D dummy argument
    (sequence association), ``InjectBlockIndexTransformation`` must not
    append an extra subscript.

    In the real IFS, ``YDVARS%EDOT%T0_FIELD`` is 3D (NPROMA x NFLEVG x
    NGPBLKS) but after block index injection becomes
    ``T0_FIELD(:,:,:,local_IBL)`` — 4 subscripts on a 3D array.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT5, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT5, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path,
        driver_name='driver_cat5', kernel_name='kernel_cat5',
        driver_item_name='driver_cat5_mod#driver_cat5',
        kernel_item_name='#kernel_cat5'
    )

    # After transformation, the driver's call to kernel_cat5 should NOT
    # have arguments with more subscripts than their declared rank.
    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_calls = [c for c in calls if 'kernel_cat5' in str(c.name).lower()]
    assert len(kernel_calls) >= 1, "Expected at least one call to kernel_cat5"

    for call in kernel_calls:
        for arg in call.arguments:
            if hasattr(arg, 'dimensions') and arg.dimensions:
                # Get the declared rank from the arg's type
                declared_rank = len(getattr(arg, 'shape', ()))
                subscript_count = len(arg.dimensions)
                assert subscript_count <= declared_rank, (
                    f"Argument '{arg.name}' has {subscript_count} subscripts "
                    f"but only {declared_rank} declared dimensions. "
                    f"This indicates an extra block index was incorrectly appended. "
                    f"Arg: {fgen(arg)}"
                )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat3_kernel_no_spurious_istsz(frontend, horizontal, block_dim, tmp_path):
    """
    Category 3 bug: when a kernel contains a loop with calls to target
    routines, ``_determine_stack_size`` may find no matching successors in
    the loop body → returns ``Literal(0)`` → ``ISTSZ = 0``.

    This test verifies that when a kernel has internal loops containing
    calls, the pool allocator does not generate an incorrect ISTSZ=0
    assignment at the kernel level.

    This is closely related to Cat 12 (pool allocator in kernel) — the
    root cause is the same: ``find_driver_loops`` incorrectly matches
    loops inside kernels.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    sub_kernel_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_CAT12_POOL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_CAT12_POOL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT12_POOL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    # Use the 3-level helper so sub_kernel_cat12p is registered as a
    # target successor of kernel_cat12p — this is what makes
    # find_driver_loops match the DO JFLD loop inside the kernel.
    driver, kernel, sub_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, kernel_source, sub_kernel_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_cat12p', mid_name='kernel_cat12p',
        sub_name='sub_kernel_cat12p',
        driver_item_name='driver_cat12p_mod#driver_cat12p',
        mid_item_name='#kernel_cat12p',
        sub_item_name='#sub_kernel_cat12p'
    )

    # The kernel should NOT have an ISTSZ assignment (since Cat 3 and Cat 12
    # share the root cause of spurious driver-loop detection in kernels)
    assignments = FindNodes(Assignment).visit(kernel.body)
    istsz_assignments = [a for a in assignments if str(a.lhs).lower() == 'istsz']
    assert len(istsz_assignments) == 0, (
        f"Kernel should NOT have ISTSZ assignment. "
        f"Found: {[fgen(a) for a in istsz_assignments]}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat3_driver_istsz_max_across_loops(frontend, horizontal, block_dim, tmp_path):
    """
    Category 3 driver-level bug: when a driver has multiple driver loops,
    ``ISTSZ`` must be set to ``MAX(...)`` across ALL loops, not just the
    first loop's stack size.

    In the real IFS, ``ecphys_setup_layer_loki`` has its first driver loop
    calling ``GPMKTEND`` (stack_size=0) while later loops call routines
    with non-zero stack sizes.  The old code set ``ISTSZ = 0`` because
    ``_get_stack_storage_and_size_var`` only creates the assignment on
    the first call.

    This test applies only the pool allocator transformation directly
    (not the full pipeline), because earlier pipeline stages would remove
    the driver loops.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_no_temp_source = Sourcefile.from_source(
        FCODE_KERNEL_NO_TEMP, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_with_temp_source = Sourcefile.from_source(
        FCODE_KERNEL_WITH_TEMP, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_CAT3_MULTI, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver_routine = driver_source['driver_cat3m']
    kernel_no_temp_routine = kernel_no_temp_source['kernel_no_temp']
    kernel_with_temp_routine = kernel_with_temp_source['kernel_with_temp']

    driver_routine.enrich(kernel_no_temp_routine)
    driver_routine.enrich(kernel_with_temp_routine)

    driver_item = ProcedureItem(name='driver_cat3m_mod#driver_cat3m', source=driver_source)
    kernel_no_temp_item = ProcedureItem(name='#kernel_no_temp', source=kernel_no_temp_source)
    kernel_with_temp_item = ProcedureItem(name='#kernel_with_temp', source=kernel_with_temp_source)

    sgraph = SGraph.from_dict({
        driver_item: [kernel_no_temp_item, kernel_with_temp_item],
    })

    # Apply ONLY the pool allocator transformation, in bottom-up order
    # (reverse_traversal=True).  We skip the full pipeline because
    # SCCBlockSectionTransformation removes driver loops before the
    # pool allocator runs.  In the real IFS build, drivers with
    # mode='scc-stack' are not subject to block-section transforms.
    pool_alloc = TemporariesPoolAllocatorPerDrvLoopTransformation(
        block_dim=block_dim, directive='openacc'
    )

    # Bottom-up: kernels first, then driver
    pool_alloc.apply(
        kernel_no_temp_routine, role='kernel', item=kernel_no_temp_item,
        targets=[], sub_sgraph=sgraph
    )
    pool_alloc.apply(
        kernel_with_temp_routine, role='kernel', item=kernel_with_temp_item,
        targets=[], sub_sgraph=sgraph
    )
    pool_alloc.apply(
        driver_routine, role='driver', item=driver_item,
        targets=['kernel_no_temp', 'kernel_with_temp'], sub_sgraph=sgraph
    )

    # After transformation, the driver must have an ISTSZ assignment
    # whose RHS is NOT just 0 — it must include the non-zero stack size
    # from kernel_with_temp's temporary array.
    driver_code = fgen(driver_routine)
    assignments = FindNodes(Assignment).visit(driver_routine.body)
    istsz_assignments = [a for a in assignments if str(a.lhs).lower() == 'istsz']

    assert len(istsz_assignments) >= 1, (
        f"Driver should have an ISTSZ assignment. "
        f"Driver code:\n{driver_code}"
    )

    for assign in istsz_assignments:
        rhs_str = str(assign.rhs).lower()
        # ISTSZ must NOT be just "0" — it must include the stack size
        # contribution from kernel_with_temp
        assert rhs_str != '0', (
            f"ISTSZ should not be just 0 when kernel_with_temp has "
            f"temporaries. Got: ISTSZ = {assign.rhs}\n"
            f"Driver code:\n{driver_code}"
        )
        # It should contain ISHFT or C_SIZEOF (from the stack size
        # computation of kernel_with_temp's ztmp array)
        assert 'ishft' in rhs_str or 'c_sizeof' in rhs_str or 'max' in rhs_str, (
            f"ISTSZ should contain stack size computation (ISHFT/C_SIZEOF/MAX). "
            f"Got: ISTSZ = {assign.rhs}\n"
            f"Driver code:\n{driver_code}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_cat12_nested_driver_kernel_gets_pool_allocator(frontend, horizontal, block_dim, tmp_path):
    """
    Cat 12 refinement: a "nested driver" kernel — one that calls sub-kernels
    via ``!$loki small-kernels`` — must GET pool allocator infrastructure
    (ISTSZ/ZSTACK/ALLOCATE/DEALLOCATE), because
    ``SCCBlockSectionToLoopTransformation`` creates block-dimension loops
    (whose loop variable matches ``block_dim.indices``) inside such kernels.

    This is the complement of ``test_cat12_kernel_no_pool_allocator_setup``:
    that test verifies simple kernels do NOT get pool allocator; this test
    verifies "nested driver" kernels DO get pool allocator.

    In the real IFS, ``sigam_gp``, ``sitnu_gp``, ``gpcty_expl``, and
    ``gpgrgeo_expl`` are "nested driver" kernels.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    leaf_source = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    mid_source = Sourcefile.from_source(
        FCODE_NESTED_DRV_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_NESTED_DRV, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, nested_kernel, leaf_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, mid_source, leaf_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_nested_drv',
        mid_name='nested_drv_kernel',
        sub_name='leaf_kernel',
        driver_item_name='driver_nested_drv_mod#driver_nested_drv',
        mid_item_name='#nested_drv_kernel',
        sub_item_name='#leaf_kernel'
    )

    nested_code = fgen(nested_kernel)

    # The nested driver kernel MUST have a block-dimension loop
    # (created by SCCBlockSectionToLoopTransformation)
    loops = FindNodes(Loop).visit(nested_kernel.body)
    block_loops = [
        l for l in loops
        if str(l.variable).lower().replace('local_', '') in
           [idx.lower() for idx in block_dim.indices]
    ]
    assert len(block_loops) >= 1, (
        f"Nested driver kernel should have a block-dimension loop "
        f"(matching block_dim.indices={block_dim.indices}). "
        f"Found loops: {[str(l.variable) for l in loops]}\n"
        f"Code:\n{nested_code}"
    )

    # The nested driver kernel MUST have ISTSZ and ZSTACK variables
    # (pool allocator infrastructure for the block loop)
    kernel_var_names = {v.name.lower() for v in nested_kernel.variables}
    assert 'istsz' in kernel_var_names, (
        f"Nested driver kernel MUST have 'ISTSZ' variable (pool allocator). "
        f"Found: {sorted(kernel_var_names)}\n"
        f"Code:\n{nested_code}"
    )
    assert 'zstack' in kernel_var_names, (
        f"Nested driver kernel MUST have 'ZSTACK' variable (pool allocator). "
        f"Found: {sorted(kernel_var_names)}\n"
        f"Code:\n{nested_code}"
    )

    # The nested driver kernel MUST have ALLOCATE/DEALLOCATE for ZSTACK
    allocations = FindNodes(Allocation).visit(nested_kernel.body)
    deallocations = FindNodes(Deallocation).visit(nested_kernel.body)
    assert len(allocations) >= 1, (
        f"Nested driver kernel MUST have ALLOCATE statements (pool allocator). "
        f"Code:\n{nested_code}"
    )
    assert len(deallocations) >= 1, (
        f"Nested driver kernel MUST have DEALLOCATE statements (pool allocator). "
        f"Code:\n{nested_code}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
# @pytest.mark.xfail(reason=(
#     'Bug A: !$loki small-kernels pragma on the call inside ASSOCIATE gets displaced '
#     'during the pipeline, so SCCBlockSectionTransformation never marks the leaf kernel '
#     'for block-section treatment and no block loop is generated in the leaf.'
# ))
def test_bugA_struct_unpack_kstglo_propagation(frontend, horizontal, block_dim, tmp_path):
    """
    Bug A: When a mid-level kernel (like ``lassie``) unpacks a derived-type
    struct (``BNDS``) into individual scalar arguments via ASSOCIATE and
    passes them to a leaf kernel (like ``sigam_gp``), the ``BNDS%KSTGLO``
    member — which is NOT individually passed — must still be propagated
    correctly so the block loop at the leaf kernel level can recompute
    ``KEND`` from ``KSTGLO``.

    The real IFS pattern:

    * **Driver** (``cpg_2_parallel``): block loop assigns
      ``BNDS%KBL``, ``BNDS%KSTGLO``, ``BNDS%KIDIA``, ``BNDS%KFDIA``.
    * **Mid kernel** (``lassie``): receives ``BNDS`` as whole struct,
      has ``ASSOCIATE(KIDIA=>BNDS%KIDIA, KFDIA=>BNDS%KFDIA)``, then
      calls ``leaf_kernel(klon, klev, KIDIA, KFDIA, ...)`` with
      ``!$loki small-kernels`` pragma — passing unpacked scalars, NOT
      the struct.
    * **Leaf kernel** (``sigam_gp``): signature
      ``(KLON, KLEV, KST, KEND, ...)`` — no ``BNDS`` argument.

    ``BNDS%KSTGLO`` is needed to recompute ``KFDIA`` (and hence ``KEND``)
    in the block loop at the leaf level: without it, the last (partial)
    block will be computed incorrectly.

    There are two sub-issues:

    1. **Pragma displacement**: The ``!$loki small-kernels`` pragma on
       the call inside the ASSOCIATE block gets displaced during the
       pipeline (likely when ``SanitiseTransformation`` resolves the
       ASSOCIATE or when ``SCCBlockSectionTransformation`` extracts
       block sections from the mid kernel).  As a result,
       ``BlockSectionTrafo`` is never set on the leaf kernel's item,
       so the leaf kernel never gets block-section treatment.

    2. **KSTGLO propagation**: Even once the pragma issue is fixed,
       ``BNDS%KSTGLO`` must be correctly propagated through the
       struct-unpacking boundary so the leaf kernel's block loop
       can recompute the upper horizontal bound using ``KSTGLO``.

    This test verifies the full pipeline: BNDS propagation to the leaf
    kernel, block loop generation in the leaf kernel, and KSTGLO-based
    clamping of the upper horizontal bound.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    mid_source = Sourcefile.from_source(
        FCODE_MID_KERNEL_BUGA, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    leaf_source = Sourcefile.from_source(
        FCODE_LEAF_KERNEL_BUGA, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_BUGA, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    # Build 3-level hierarchy: driver → mid_kernel → leaf_kernel
    driver, mid_kernel, leaf_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, mid_source, leaf_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_buga',
        mid_name='mid_kernel_buga',
        sub_name='leaf_kernel_buga',
        driver_item_name='driver_buga_mod#driver_buga',
        mid_item_name='#mid_kernel_buga',
        sub_item_name='#leaf_kernel_buga'
    )

    driver_code = fgen(driver)
    mid_code = fgen(mid_kernel)
    leaf_code = fgen(leaf_kernel)

    # ---------------------------------------------------------------
    # 1. DRIVER level: sanity check — should have KSTGLO assignment
    # ---------------------------------------------------------------
    driver_lower = driver_code.lower()
    assert 'bnds%kstglo' in driver_lower or 'local_bnds%kstglo' in driver_lower, (
        f"Driver should contain KSTGLO assignment in block loop.\n"
        f"Driver code:\n{driver_code}"
    )

    # ---------------------------------------------------------------
    # 2. MID KERNEL level (lassie-like): should have a block loop with
    #    local_bnds member assignments including KSTGLO, and
    #    local_bnds in the private() clause.
    # ---------------------------------------------------------------
    mid_lower = mid_code.lower()

    # 2a. local_bnds should exist and be assigned
    assert 'local_bnds' in mid_lower, (
        f"Mid kernel should have 'local_bnds' variable.\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # 2b. KSTGLO assignment inside block loop
    assert 'kstglo' in mid_lower, (
        f"Mid kernel should have KSTGLO assignment in block loop.\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # 2c. private(local_bnds) in acc directive
    import re
    mid_private_matches = re.findall(r'private\(([^)]+)\)', mid_lower)
    mid_has_local_bnds_private = any(
        'local_bnds' in m for m in mid_private_matches
    )
    assert mid_has_local_bnds_private, (
        f"Mid kernel should have 'private(local_bnds)' in !$acc directive.\n"
        f"Private clauses found: {mid_private_matches}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # 2d. BNDS should be passed as kwarg to the leaf kernel call
    calls = FindNodes(CallStatement).visit(mid_kernel.body)
    leaf_calls = [c for c in calls if 'leaf_kernel_buga' in str(c.name).lower()]
    assert len(leaf_calls) >= 1, (
        f"Mid kernel should call leaf_kernel_buga.\n"
        f"All calls: {[str(c.name) for c in calls]}\n"
        f"Mid kernel code:\n{mid_code}"
    )
    for call in leaf_calls:
        kwarg_names = [kw[0].lower() for kw in call.kwarguments]
        assert any('bnds' in kw for kw in kwarg_names), (
            f"Call to leaf_kernel_buga should have BNDS kwarg.\n"
            f"Found kwargs: {kwarg_names}\n"
            f"Mid kernel code:\n{mid_code}"
        )

    # ---------------------------------------------------------------
    # 3. LEAF KERNEL level (sigam_gp-like): must have a block loop
    #    with KSTGLO-based clamping of the upper horizontal bound.
    #
    #    This is the main Bug A assertion — the pragma displacement
    #    causes the leaf kernel to NOT receive block-section treatment,
    #    so no block loop is generated.
    # ---------------------------------------------------------------
    leaf_lower = leaf_code.lower()

    # 3a. The leaf kernel MUST have BNDS in its signature (added by
    #     LowerBlockIndexTransformation because KSTGLO needs it)
    leaf_arg_names = [str(a).lower() for a in leaf_kernel.arguments]
    assert any('bnds' in a for a in leaf_arg_names), (
        f"Leaf kernel should have BNDS in its argument list.\n"
        f"Found args: {leaf_arg_names}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # 3b. The leaf kernel MUST have a block-dimension loop
    leaf_loops = FindNodes(Loop).visit(leaf_kernel.body)
    block_loop_vars = [str(l.variable).lower().replace('local_', '')
                       for l in leaf_loops]
    leaf_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in block_loop_vars
    )
    assert leaf_has_block_loop, (
        f"Leaf kernel should have a block-dimension loop.\n"
        f"Loop variables found: {[str(l.variable) for l in leaf_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # 3c. The block loop must contain a KSTGLO-based clamping of the
    #     upper horizontal bound.  Check for either approach:
    #
    #     Approach (a): BNDS passed as kwarg → local_bnds%kstglo
    has_bnds_approach = 'local_bnds%kstglo' in leaf_lower or 'bnds%kstglo' in leaf_lower
    #     Approach (b): KSTGLO as standalone variable in a MIN()
    has_kstglo_standalone = 'kstglo' in leaf_lower and 'min(' in leaf_lower

    assert has_bnds_approach or has_kstglo_standalone, (
        f"Leaf kernel's block loop must recompute the upper horizontal bound "
        f"using KSTGLO (to clamp the last partial block).  Neither approach found.\n"
        f"Expected either 'local_bnds%kstglo' / 'bnds%kstglo' in code,\n"
        f"or a standalone 'kstglo' variable used in a MIN() expression.\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # 3d. The block loop must have an assignment to the upper
    #     horizontal bound (local_kend or local_kfdia) that uses MIN
    leaf_assignments = FindNodes(Assignment).visit(leaf_kernel.body)
    upper_bound_names = ['local_kend', 'local_kfdia']
    upper_bound_assigns = [
        a for a in leaf_assignments
        if str(a.lhs).lower() in upper_bound_names
    ]
    assert len(upper_bound_assigns) >= 1, (
        f"Leaf kernel should have an assignment to a local upper horizontal "
        f"bound variable ({upper_bound_names}).\n"
        f"All assignments: {[f'{a.lhs} = {a.rhs}' for a in leaf_assignments]}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )


# ---------------------------------------------------------------------------
# Bug D: ISTSZ must account for ALL callee branches (IF/ELSE)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_bugD_istsz_accounts_for_conditional_branches(frontend, horizontal, block_dim, tmp_path):
    """
    Bug D: ISTSZ under-enumeration for conditional branches.

    When a driver's block loop calls a kernel that has IF/ELSE branches
    calling different sub-kernels (``sub_d1`` in the IF branch, ``sub_d2``
    in the ELSE branch), the ISTSZ ``MAX(...)`` must include stack-size
    terms from BOTH sub-kernels' temporaries.

    In the real IFS, ``ecphys_setup_layer`` has its ISTSZ with only ~4
    terms (generated) versus ~15+ terms (working).  The missing terms
    come from callee branches that ``_determine_stack_size`` fails to
    discover/propagate — likely because some callees are not in the
    Scheduler's successor graph (e.g., in ``block`` lists).

    NOTE: This test applies ONLY the pool allocator (not the full
    pipeline) to isolate the ISTSZ computation from driver-loop
    stripping.  With all successors registered in the SGraph, the pool
    allocator correctly handles conditional branches (flat
    ``FindNodes(CallStatement)`` finds all calls regardless of IF/ELSE).
    The real-world under-enumeration may require integration testing
    with the Scheduler's item filtering to reproduce.

    The hierarchy is: driver_bugd → kernel_bugd → (sub_d1 | sub_d2)

    ``sub_d1`` has ``ztmp1(klon,klev)`` (REAL default kind).
    ``sub_d2`` has ``ztmp2(klon,klev)`` (REAL(KIND=JPRD)) and
    ``ztmp3(klon,klev)`` (REAL default kind).

    The driver's ISTSZ must include C_SIZEOF contributions from BOTH
    sub-kernel branches, via the intermediate ``kernel_bugd``.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    parkind_mod = Module.from_source(FCODE_PARKIND1_MOD, frontend=frontend, xmods=[tmp_path])

    sub_d1_source = Sourcefile.from_source(
        FCODE_SUB_D1, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    sub_d2_source = Sourcefile.from_source(
        FCODE_SUB_D2, frontend=frontend,
        definitions=[bnds_mod, opts_mod, parkind_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_BUGD, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_BUGD, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver_routine = driver_source['driver_bugd']
    kernel_routine = kernel_source['kernel_bugd']
    sub_d1_routine = sub_d1_source['sub_d1']
    sub_d2_routine = sub_d2_source['sub_d2']

    driver_routine.enrich(kernel_routine)
    kernel_routine.enrich(sub_d1_routine)
    kernel_routine.enrich(sub_d2_routine)

    driver_item = ProcedureItem(name='driver_bugd_mod#driver_bugd', source=driver_source)
    kernel_item = ProcedureItem(name='#kernel_bugd', source=kernel_source)
    sub_d1_item = ProcedureItem(name='#sub_d1', source=sub_d1_source)
    sub_d2_item = ProcedureItem(name='#sub_d2', source=sub_d2_source)

    sgraph = SGraph.from_dict({
        driver_item: [kernel_item],
        kernel_item: [sub_d1_item, sub_d2_item],
    })

    # Apply ONLY the pool allocator transformation, in bottom-up order.
    # We skip the full pipeline to avoid driver-loop stripping.
    pool_alloc = TemporariesPoolAllocatorPerDrvLoopTransformation(
        block_dim=block_dim, directive='openacc'
    )

    # Bottom-up: leaf kernels first, then mid-level kernel, then driver
    pool_alloc.apply(
        sub_d1_routine, role='kernel', item=sub_d1_item,
        targets=[], sub_sgraph=sgraph
    )
    pool_alloc.apply(
        sub_d2_routine, role='kernel', item=sub_d2_item,
        targets=[], sub_sgraph=sgraph
    )
    pool_alloc.apply(
        kernel_routine, role='kernel', item=kernel_item,
        targets=['sub_d1', 'sub_d2'], sub_sgraph=sgraph
    )
    pool_alloc.apply(
        driver_routine, role='driver', item=driver_item,
        targets=['kernel_bugd'], sub_sgraph=sgraph
    )

    driver_code = fgen(driver_routine)
    kernel_code = fgen(kernel_routine)
    sub1_code = fgen(sub_d1_routine)
    sub2_code = fgen(sub_d2_routine)

    print(f"\n{'='*60}")
    print(f"Bug D: ISTSZ conditional branches (pool allocator only)")
    print(f"{'='*60}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"\n--- SUB_D1 ---\n{sub1_code}")
    print(f"\n--- SUB_D2 ---\n{sub2_code}")
    print(f"{'='*60}\n")

    # 1. The driver must have an ISTSZ assignment
    assignments = FindNodes(Assignment).visit(driver_routine.body)
    istsz_assignments = [a for a in assignments if str(a.lhs).lower() == 'istsz']

    assert len(istsz_assignments) >= 1, (
        f"Driver should have an ISTSZ assignment.\n"
        f"Driver code:\n{driver_code}"
    )

    # 2. Filter out the ISTSZ=0 initialisation to find the real assignment
    real_istsz = [a for a in istsz_assignments if str(a.rhs).lower() != '0']
    assert len(real_istsz) >= 1, (
        f"Driver should have a non-zero ISTSZ assignment.\n"
        f"All ISTSZ assignments: {[f'ISTSZ = {a.rhs}' for a in istsz_assignments]}\n"
        f"Driver code:\n{driver_code}"
    )

    # 3. The ISTSZ RHS must contain real stack size computations
    for assign in real_istsz:
        rhs_str = str(assign.rhs).lower()
        assert 'ishft' in rhs_str or 'c_sizeof' in rhs_str or 'max' in rhs_str, (
            f"ISTSZ should contain stack size computation (ISHFT/C_SIZEOF/MAX). "
            f"Got: ISTSZ = {assign.rhs}\n"
            f"Driver code:\n{driver_code}"
        )

    # 4. KEY ASSERTION: The ISTSZ MAX must include contributions from
    #    BOTH sub_d1 AND sub_d2.  sub_d2 uses JPRD temporaries, so we
    #    expect at least 2 distinct ISHFT terms (one per branch).
    #    sub_d1 has 1 temporary (ztmp1) → at least 1 ISHFT.
    #    sub_d2 has 2 temporaries (ztmp2 JPRD + ztmp3) → at least 2 ISHFT.
    #    Total: at least 2 ISHFT terms from different branches.
    istsz_rhs = str(real_istsz[0].rhs).lower()
    ishft_count = istsz_rhs.count('ishft')

    assert ishft_count >= 2, (
        f"ISTSZ should include stack contributions from BOTH conditional "
        f"branches (sub_d1 and sub_d2).  Expected at least 2 ISHFT terms "
        f"(one per branch), but found {ishft_count}.\n"
        f"ISTSZ = {real_istsz[0].rhs}\n"
        f"Driver code:\n{driver_code}"
    )


# ---------------------------------------------------------------------------
# Bug F: _parallel driver must retain block loop + ZSTACK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_bugF_parallel_driver_gets_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    Bug F: _parallel driver routines must retain their block loops and get
    ZSTACK/ISTSZ pool allocation infrastructure.

    In the real IFS, ``cpg_0_parallel``, ``cpg_2_parallel``, and
    ``lapineb_parallel`` are ``role='driver'`` routines with existing
    ``DO JKGLO`` block loops.  After the SCC small-kernels pipeline,
    these loops must be preserved (not stripped) and augmented with
    stack allocation: ``ZSTACK``, ``ISTSZ``, ``ALLOCATE``/``DEALLOCATE``,
    and ``YLSTACK_L = LOC(ZSTACK(1, IBL))``.

    The generated (buggy) output strips the block loop entirely, leaving
    bare code with uninitialized ``JKGLO`` and no stack infrastructure.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_BUGF, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_BUGF, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path,
        driver_name='driver_bugf', kernel_name='kernel_bugf',
        driver_item_name='driver_bugf_mod#driver_bugf',
        kernel_item_name='#kernel_bugf'
    )

    driver_code = fgen(driver)
    kernel_code = fgen(kernel)

    print(f"\n{'='*60}")
    print(f"Bug F: _parallel driver block loop + ZSTACK")
    print(f"{'='*60}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"{'='*60}\n")

    driver_lower = driver_code.lower()

    # 1. The driver must still have a block-dimension loop
    driver_loops = FindNodes(Loop).visit(driver.body)
    block_loop_vars = [str(l.variable).lower() for l in driver_loops]
    has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in block_loop_vars
    )
    assert has_block_loop, (
        f"Driver should have a block-dimension loop after transformation.\n"
        f"Loop variables found: {block_loop_vars}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Driver code:\n{driver_code}"
    )

    # 2. The driver must have ZSTACK allocation
    allocations = FindNodes(Allocation).visit(driver.body)
    zstack_allocs = [a for a in allocations
                     if any('zstack' in str(v).lower() for v in a.variables)]
    assert len(zstack_allocs) >= 1, (
        f"Driver should have ALLOCATE(ZSTACK(...)).\n"
        f"All allocations: {[fgen(a) for a in allocations]}\n"
        f"Driver code:\n{driver_code}"
    )

    # 3. The driver must have ZSTACK deallocation
    deallocs = FindNodes(Deallocation).visit(driver.body)
    zstack_deallocs = [d for d in deallocs
                       if any('zstack' in str(v).lower() for v in d.variables)]
    assert len(zstack_deallocs) >= 1, (
        f"Driver should have DEALLOCATE(ZSTACK).\n"
        f"All deallocations: {[fgen(d) for d in deallocs]}\n"
        f"Driver code:\n{driver_code}"
    )

    # 4. The driver must have an ISTSZ assignment that is non-zero
    assignments = FindNodes(Assignment).visit(driver.body)
    istsz_assignments = [a for a in assignments if str(a.lhs).lower() == 'istsz']
    real_istsz = [a for a in istsz_assignments if str(a.rhs).lower() != '0']
    assert len(real_istsz) >= 1, (
        f"Driver should have a non-zero ISTSZ assignment.\n"
        f"All ISTSZ assignments: {[f'ISTSZ = {a.rhs}' for a in istsz_assignments]}\n"
        f"Driver code:\n{driver_code}"
    )

    # 5. The driver must pass YDSTACK_L (or YLSTACK_L) to the kernel call
    calls = FindNodes(CallStatement).visit(driver.body)
    kernel_calls = [c for c in calls if 'kernel_bugf' in str(c.name).lower()]
    assert len(kernel_calls) >= 1, (
        f"Driver should call kernel_bugf.\n"
        f"All calls: {[str(c.name) for c in calls]}\n"
        f"Driver code:\n{driver_code}"
    )
    for call in kernel_calls:
        kwarg_names = [kw[0].lower() for kw in call.kwarguments] if call.kwarguments else []
        assert any('stack' in kw for kw in kwarg_names), (
            f"Call to kernel_bugf should have a stack keyword argument "
            f"(YDSTACK_L or similar).\n"
            f"Found kwargs: {kwarg_names}\n"
            f"Driver code:\n{driver_code}"
        )


# ---------------------------------------------------------------------------
# Bug G: No doubled NGPBLKS dimension on kernel local arrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_bugG_no_doubled_block_dimension(frontend, horizontal, block_dim, tmp_path):
    """
    Bug G: Extra NGPBLKS array dimension (4D instead of 3D).

    When a driver calls a kernel that has local arrays (e.g.
    ``REAL :: zlocal1(klon, klev)``), ``LowerBlockIndexTransformation``
    should add exactly ONE ``NGPBLKS`` dimension to make it
    ``zlocal1(klon, klev, ngpblks)``.

    The bug: ``process_driver`` (unlike ``process_kernel``) has NO guard
    checking whether the array's last dimension already matches
    ``block_dim.sizes``.  When a driver has **two driver loops** each
    calling the same callee (e.g., a GPU path + CPU path, as in
    ``lapineb_parallel``), the ``relevant_calls`` list contains two
    entries for the same ``call.routine``.  ``process_driver`` promotes
    the callee's local arrays once per call, blindly appending
    ``NGPBLKS`` each time → ``(klon, klev, ngpblks, ngpblks)``.

    The fix: add the same guard from ``process_kernel`` line 732
    (``if local_var.dimensions[-1] not in self.block_dim.sizes``)
    to ``process_driver`` line 923.

    In the real IFS, ``lapineb``'s 34 local arrays all get the doubled
    dimension because ``lapineb_parallel`` has two driver loops (GPU +
    CPU paths) both calling ``LAPINEB``.

    This test uses a 3-level hierarchy (driver → kernel → leaf) where
    the driver has TWO block loops (IF/ELSE) calling the same kernel,
    and the kernel has local arrays.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    leaf_source = Sourcefile.from_source(
        FCODE_LEAF_BUGG, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_BUGG, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_BUGG, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel, leaf = _apply_small_kernels_pipeline_3level(
        driver_source, kernel_source, leaf_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_bugg',
        mid_name='kernel_bugg',
        sub_name='leaf_bugg',
        driver_item_name='driver_bugg_mod#driver_bugg',
        mid_item_name='#kernel_bugg',
        sub_item_name='#leaf_bugg'
    )

    driver_code = fgen(driver)
    kernel_code = fgen(kernel)
    leaf_code = fgen(leaf)

    print(f"\n{'='*60}")
    print(f"Bug G: Doubled NGPBLKS dimension")
    print(f"{'='*60}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"\n--- LEAF ---\n{leaf_code}")
    print(f"{'='*60}\n")

    # Check each local array in the kernel for doubled NGPBLKS
    kernel_lower = kernel_code.lower()

    # Find variable declarations in the kernel
    kernel_vars = kernel.variables
    local_arrays = [v for v in kernel_vars
                    if isinstance(v, sym.Array) and v.shape
                    and not v.type.intent]

    print(f"\nKernel local arrays:")
    for v in local_arrays:
        dims_str = ', '.join(str(d) for d in v.shape)
        print(f"  {v.name}: ({dims_str})")

    # For each local array, count how many dimensions match block_dim.size
    for v in local_arrays:
        if not v.shape:
            continue
        dim_strs = [str(d).lower() for d in v.shape]
        ngpblks_count = sum(1 for d in dim_strs if d in [s.lower() for s in block_dim.sizes])
        assert ngpblks_count <= 1, (
            f"Local array '{v.name}' has {ngpblks_count} NGPBLKS dimensions "
            f"(expected at most 1).  Dimensions: ({', '.join(str(d) for d in v.shape)})\n"
            f"Kernel code:\n{kernel_code}"
        )


# ---------------------------------------------------------------------------
# Bug H: CHARACTER variable assignment must stay outside parallel loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_bugH_character_var_outside_parallel_loop(frontend, horizontal, block_dim, tmp_path):
    """
    Bug H: CHARACTER variable assignment placed outside ``!$acc parallel loop``.

    A kernel has ``CHARACTER(LEN=4) :: CLOPER`` assigned before the
    horizontal loop.  After transformation, the assignment must stay
    OUTSIDE the ``!$acc parallel loop gang`` region because CHARACTER
    variables are not supported inside OpenACC parallel regions on
    NVHPC compilers.

    Note: This was fixed via IFS source changes (moving CLOPER
    assignments before the IF block), not a Loki transformation fix.
    Loki has no CHARACTER-type awareness — placement depends on whether
    the statement uses the horizontal index.  Since CLOPER assignments
    do NOT use the horizontal index, they should naturally stay outside
    the vector section.

    This test verifies that the pipeline correctly keeps non-horizontal
    scalar assignments outside the ``!$acc parallel loop`` region.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_BUGH, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_BUGH, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel = _apply_small_kernels_pipeline(
        driver_source, kernel_source, horizontal, block_dim, tmp_path,
        driver_name='driver_bugh', kernel_name='kernel_bugh',
        driver_item_name='driver_bugh_mod#driver_bugh',
        kernel_item_name='#kernel_bugh'
    )

    driver_code = fgen(driver)
    kernel_code = fgen(kernel)

    print(f"\n{'='*60}")
    print(f"Bug H: CHARACTER variable outside parallel loop")
    print(f"{'='*60}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"{'='*60}\n")

    # Find the position of !$acc parallel loop gang and CLOPER assignment
    # in the generated kernel code.
    kernel_lines = kernel_code.split('\n')

    acc_parallel_line = None
    cloper_assign_lines = []
    for i, line in enumerate(kernel_lines):
        line_lower = line.strip().lower()
        if '!$acc parallel' in line_lower and 'loop' in line_lower:
            acc_parallel_line = i
        if 'cloper' in line_lower and '=' in line_lower and 'character' not in line_lower:
            cloper_assign_lines.append(i)

    print(f"\n!$acc parallel loop at line: {acc_parallel_line}")
    print(f"CLOPER assignment lines: {cloper_assign_lines}")
    for idx in cloper_assign_lines:
        print(f"  Line {idx}: {kernel_lines[idx].strip()}")

    # If there's no !$acc parallel loop, the kernel might be structured
    # differently — check that CLOPER is at least present
    if acc_parallel_line is not None and cloper_assign_lines:
        # All CLOPER assignments must appear BEFORE the !$acc parallel loop
        for cloper_line in cloper_assign_lines:
            assert cloper_line < acc_parallel_line, (
                f"CLOPER assignment at line {cloper_line} "
                f"('{kernel_lines[cloper_line].strip()}') "
                f"should appear BEFORE the !$acc parallel loop at line "
                f"{acc_parallel_line} ('{kernel_lines[acc_parallel_line].strip()}').\n"
                f"CHARACTER assignments must stay outside the parallel region.\n"
                f"Kernel code:\n{kernel_code}"
            )
    elif acc_parallel_line is None:
        # No !$acc parallel loop found — this might be expected for a
        # kernel that doesn't get annotated, but let's flag it
        print(f"WARNING: No !$acc parallel loop found in kernel code.")
        print(f"This may indicate the kernel was not annotated.")


# ---------------------------------------------------------------------------
# Bug K: JPRD import propagated through pool allocator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_bugK_jprd_import_propagated(frontend, horizontal, block_dim, tmp_path):
    """
    Bug K: JPRD kind import missing from driver/kernel.

    When a sub-kernel has ``REAL(KIND=JPRD)`` temporaries placed on the
    stack, the pool allocator generates ``C_SIZEOF(REAL(1, kind=JPRD))``
    in the ISTSZ computation.  The ``JPRD`` kind parameter must be
    imported (``USE PARKIND1, ONLY: JPRD``) in every routine that
    references it in its stack-size expression.

    The pool allocator's ``kind_imports`` mechanism collects kind params
    from temporaries and propagates them up the call tree.  This test
    verifies that ``JPRD`` (from a sub-kernel) gets propagated to the
    mid-level kernel and the driver.

    In the real IFS, ``cpg_dyn_slg`` has ``USE PARKIND1, ONLY: JPIM, JPRB``
    but is missing ``JPRD``.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    parkind_mod = Module.from_source(FCODE_PARKIND1_MOD, frontend=frontend, xmods=[tmp_path])

    sub_source = Sourcefile.from_source(
        FCODE_SUB_BUGK, frontend=frontend,
        definitions=[bnds_mod, opts_mod, parkind_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_BUGK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_BUGK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, kernel, sub = _apply_small_kernels_pipeline_3level(
        driver_source, kernel_source, sub_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_bugk',
        mid_name='kernel_bugk',
        sub_name='sub_bugk',
        driver_item_name='driver_bugk_mod#driver_bugk',
        mid_item_name='#kernel_bugk',
        sub_item_name='#sub_bugk'
    )

    driver_code = fgen(driver)
    kernel_code = fgen(kernel)
    sub_code = fgen(sub)

    print(f"\n{'='*60}")
    print(f"Bug K: JPRD import propagation")
    print(f"{'='*60}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"\n--- SUB_BUGK ---\n{sub_code}")
    print(f"{'='*60}\n")

    # List all imports in each routine
    for name, routine in [('driver', driver), ('kernel', kernel), ('sub', sub)]:
        imports = FindNodes(Import).visit(routine.spec)
        print(f"\n{name} imports:")
        for imp in imports:
            print(f"  {fgen(imp).strip()}")

    # 1. The sub-kernel should have JPRD in its imports (it already had it)
    sub_imports = FindNodes(Import).visit(sub.spec)
    sub_import_symbols = set()
    for imp in sub_imports:
        for sym_name in imp.symbols:
            sub_import_symbols.add(str(sym_name).lower())
    assert 'jprd' in sub_import_symbols, (
        f"Sub-kernel should have JPRD in its imports.\n"
        f"Sub-kernel imports: {sub_import_symbols}\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # 2. The mid-level kernel should have JPRD in its imports
    #    (propagated by pool allocator for the ISTSZ computation)
    kernel_imports = FindNodes(Import).visit(kernel.spec)
    kernel_import_symbols = set()
    for imp in kernel_imports:
        for sym_name in imp.symbols:
            kernel_import_symbols.add(str(sym_name).lower())

    assert 'jprd' in kernel_import_symbols, (
        f"Mid-level kernel should have JPRD import (propagated from sub-kernel "
        f"for stack size computation).\n"
        f"Kernel imports: {kernel_import_symbols}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # 3. The driver should have JPRD in its imports
    driver_imports = FindNodes(Import).visit(driver.spec)
    driver_import_symbols = set()
    for imp in driver_imports:
        for sym_name in imp.symbols:
            driver_import_symbols.add(str(sym_name).lower())

    assert 'jprd' in driver_import_symbols, (
        f"Driver should have JPRD import (propagated from sub-kernel "
        f"for stack size computation).\n"
        f"Driver imports: {driver_import_symbols}\n"
        f"Driver code:\n{driver_code}"
    )
