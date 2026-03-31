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
        order = items_in_order
        if getattr(transform, 'reverse_traversal', False):
            order = list(reversed(items_in_order))
        for routine, role, item, targets in order:
            transform.apply(
                routine, role=role, item=item,
                targets=targets, sub_sgraph=sgraph
            )

    return driver_routine, mid_routine, sub_routine


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
@pytest.mark.xfail(reason='Cat 9: Missing BNDS/OPTS kwargs in sub-kernel calls from process_kernel')
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
@pytest.mark.xfail(reason='Cat 6: Host-path block loop call does not get IBL argument')
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
@pytest.mark.xfail(reason='Cat 11: Call outside main block section does not get own block loop')
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
@pytest.mark.xfail(reason='Cat 5: InjectBlockIndex appends extra subscript for rank-mismatched args')
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
