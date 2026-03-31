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
    Pragma, FindVariables, FindInlineCalls, Section
)
from loki.transformations.single_column import (
    SCCSmallKernelsPipeline
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

    # Apply in top-down order: driver, mid, sub
    items_in_order = [
        (driver_routine, 'driver', driver_item, [mid_name]),
        (mid_routine, 'kernel', mid_item, [sub_name]),
        (sub_routine, 'kernel', sub_item, []),
    ]

    for transform in pipeline.transformations:
        for routine, role, item, targets in items_in_order:
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
