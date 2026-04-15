# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for the SCCSmallKernelsPipeline — v2 (fresh start).

This test file focuses on the remaining real bugs in the Loki-generated
output for the IFS small-kernels GPU build. The primary test validates
the **BNDS recompute pattern**: when a mid-level kernel (like ``LASSIE``)
unpacks ``YDCPG_BNDS`` members via ASSOCIATE and passes them as positional
scalars to a leaf kernel (like ``SIGAM_GP``), the pipeline must:

1. Propagate ``YDCPG_BNDS`` and ``YDCPG_OPTS`` as new kwargs to the
   leaf kernel.
2. Generate a block loop inside the leaf kernel that recomputes
   ``local_YDCPG_BNDS%KBL``, ``local_YDCPG_BNDS%KSTGLO``, and
   ``local_KEND = MIN(KLON, OPTS%KGPCOMP - local_BNDS%KSTGLO + 1)``.
3. Add ``private(local_YDCPG_BNDS)`` to the ``!$acc parallel loop``
   directive.

This pattern affects 8+ IFS source files (sigam_mod, sitnu_mod,
gpcty_expl, gpgeo_expl, gpgrgeo_expl, lattes, cpg_2, cpg_gp).
"""

import re
import pytest

from loki import (
    Sourcefile, Dimension, fgen, Module
)
from loki.batch import ProcedureItem, SGraph
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    FindNodes, Assignment, CallStatement, Loop,
    Pragma, Import, Allocation, Deallocation,
    Section, Intrinsic
)
from loki.transformations.single_column import (
    SCCSmallKernelsPipeline
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    """
    Horizontal dimension matching the real IFS config: ``kst``/``kend``
    as lower/upper (used by sigam_gp), plus ``kidia``/``kfdia`` and
    ``bnds%kidia``/``bnds%kfdia`` as additional recognised aliases.
    ``klon`` appears in both ``size`` and ``upper``.
    """
    return Dimension(
        name='horizontal', size='klon', index='jrof',
        lower=('kst', 'kidia', 'bnds%kidia'),
        upper=('kend', 'kfdia', 'bnds%kfdia', 'klon')
    )


@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(
        name='block_dim',
        size='ngpblks',
        index=('ibl', 'bnds%kbl')
    )


# ---------------------------------------------------------------------------
# Shared Fortran module fragments
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

FCODE_PARKIND1_MOD = """
module parkind1
  implicit none
  integer, parameter :: jpim = selected_int_kind(9)
  integer, parameter :: jprb = selected_real_kind(13, 300)
  integer, parameter :: jprd = selected_real_kind(13, 300)
end module parkind1
""".strip()


# ---------------------------------------------------------------------------
# BNDS recompute test: 3-level hierarchy matching lassie -> sigam_gp
#
# Driver (cpg_2-like):
#   DO IBL = 1, NGPBLKS
#     BNDS%KBL = IBL; BNDS%KSTGLO = ...; BNDS%KIDIA = 1; BNDS%KFDIA = MIN(...)
#     !$loki small-kernels
#     CALL mid_kernel(OPTS%KLON, OPTS%KFLEVG, BNDS, OPTS, T(:,:,IBL), Q(:,:,IBL))
#   END DO
#
# Mid kernel (lassie-like):
#   args: KLON, KLEV, BNDS, OPTS, T(KLON,KLEV), Q(KLON,KLEV)
#   ASSOCIATE(KIDIA => BNDS%KIDIA, KFDIA => BNDS%KFDIA)
#     ! local horizontal work
#     !$loki small-kernels
#     CALL leaf_kernel(KLON, KLEV, KIDIA, KFDIA, T, Q)
#   END ASSOCIATE
#
# Leaf kernel (sigam_gp-like):
#   args: KLON, KLEV, KST, KEND, T(KLON,KLEV), Q(KLON,KLEV)
#   DO JK = 1, KLEV
#     DO JROF = KST, KEND
#       ...
#     END DO
#   END DO
# ---------------------------------------------------------------------------

FCODE_DRIVER_BNDS_RECOMPUTE = """
module driver_bnds_recompute_mod
  implicit none
contains
  subroutine driver_bnds_recompute(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "mid_kernel_bnds_recompute.intfb.h"
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
      call mid_kernel_bnds_recompute(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine driver_bnds_recompute
end module driver_bnds_recompute_mod
""".strip()

FCODE_MID_KERNEL_BNDS_RECOMPUTE = """
subroutine mid_kernel_bnds_recompute(klon, klev, bnds, opts, t, q)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "leaf_kernel_bnds_recompute.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  ! Unpack BNDS members via ASSOCIATE (like LASSIE does).
  ! After SCCBaseTransformation resolves this ASSOCIATE, the code
  ! becomes: CALL leaf_kernel(..., BNDS%KIDIA, BNDS%KFDIA, ...)
  ASSOCIATE(KIDIA => BNDS%KIDIA, KFDIA => BNDS%KFDIA)

    ! Local horizontal work using unpacked bounds
    do jk = 1, klev
      do jrof = kidia, kfdia
        t(jrof, jk) = t(jrof, jk) + 1.0
      end do
    end do

    ! Call leaf kernel with UNPACKED scalars (like lassie calling sigam_gp).
    ! The leaf kernel signature uses KST/KEND — positional args, no BNDS.

    !$loki small-kernels
    call leaf_kernel_bnds_recompute(klon, klev, kidia, kfdia, t, q)

  END ASSOCIATE
end subroutine mid_kernel_bnds_recompute
""".strip()

FCODE_LEAF_KERNEL_BNDS_RECOMPUTE = """
subroutine leaf_kernel_bnds_recompute(klon, klev, kst, kend, t, q)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = kst, kend
      q(jrof, jk) = t(jrof, jk) * 2.0
    end do
  end do
end subroutine leaf_kernel_bnds_recompute
""".strip()


# ---------------------------------------------------------------------------
# Dual-path driver test: 4-level hierarchy matching the real IFS pattern
#
# cpg_2_parallel (driver) -> cpg_2 (mid1) -> lassie (mid2) -> sigam_gp (leaf)
#
# The driver has TWO code paths inside an IF/ELSE:
#   GPU path:  DO IBL ... inline BNDS assignments ... CALL mid_kernel
#   CPU path:  DO IBL ... CALL update_bnds(bnds, ibl, opts) ... CALL mid_kernel
#
# process_driver finds BOTH loops. If it processes the CPU path second,
# the driver_loop stored for the successor has an EMPTY body (the
# update_bnds call is stripped as a CallStatement), overwriting the
# GPU path's 4 inline BNDS assignments. This causes all downstream
# kernels to lack BNDS recompute blocks.
# ---------------------------------------------------------------------------

FCODE_DUAL_PATH_DRIVER = """
module dual_path_driver_mod
  implicit none
contains
  subroutine dual_path_driver(on_gpu, ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "cpg2_like_kernel.intfb.h"
    logical, intent(in) :: on_gpu
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)

    integer :: ibl

    if (on_gpu) then
      ! GPU path: inline BNDS assignments (like cpg_2_parallel GPU path)
      do ibl = 1, ngpblks
        bnds%kbl = ibl
        bnds%kstglo = 1 + (ibl - 1) * opts%klon
        bnds%kidia = 1
        bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

        !$loki small-kernels
        call cpg2_like_kernel(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl))
      end do
    else
      ! CPU path: BNDS set via subroutine call (models YLCPG_BNDS%UPDATE(IBL))
      do ibl = 1, ngpblks
        call update_bnds(bnds, ibl, opts)

        !$loki small-kernels
        call cpg2_like_kernel(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl))
      end do
    end if
  end subroutine dual_path_driver
end module dual_path_driver_mod
""".strip()

FCODE_CPG2_LIKE_KERNEL = """
subroutine cpg2_like_kernel(klon, klev, bnds, opts, t, q)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "mid_kernel_bnds_recompute.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof

  ! Some local horizontal work (like cpg_2's ISETTLOFF assignment)
  do jrof = bnds%kidia, bnds%kfdia
    t(jrof, 1) = t(jrof, 1) + 0.5
  end do

  !$loki small-kernels
  call mid_kernel_bnds_recompute(klon, klev, bnds, opts, t, q)
end subroutine cpg2_like_kernel
""".strip()


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _apply_small_kernels_pipeline_3level(
        driver_source, mid_source, sub_source, horizontal, block_dim, tmp_path,
        driver_name, mid_name, sub_name,
        driver_item_name, mid_item_name, sub_item_name):
    """
    Three-level variant: driver -> mid_kernel -> sub_kernel.

    The driver->mid_kernel call has ``!$loki small-kernels``.
    The mid_kernel->sub_kernel call also has ``!$loki small-kernels``.

    Returns ``(driver_routine, mid_routine, sub_routine)`` after transformation.
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


def _apply_small_kernels_pipeline_4level(
        driver_source, mid1_source, mid2_source, leaf_source,
        horizontal, block_dim, tmp_path,
        driver_name, mid1_name, mid2_name, leaf_name,
        driver_item_name, mid1_item_name, mid2_item_name, leaf_item_name):
    """
    Four-level variant: driver -> mid1_kernel -> mid2_kernel -> leaf_kernel.

    Models the real IFS call chain::

        cpg_2_parallel (driver) -> cpg_2 (mid1) -> lassie (mid2) -> sigam_gp (leaf)

    All inter-kernel calls have ``!$loki small-kernels``.

    Returns ``(driver_routine, mid1_routine, mid2_routine, leaf_routine)``
    after transformation.
    """
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source[driver_name]
    mid1_routine = mid1_source[mid1_name]
    mid2_routine = mid2_source[mid2_name]
    leaf_routine = leaf_source[leaf_name]

    driver_routine.enrich(mid1_routine)
    mid1_routine.enrich(mid2_routine)
    mid2_routine.enrich(leaf_routine)

    driver_item = ProcedureItem(name=driver_item_name, source=driver_source)
    mid1_item = ProcedureItem(name=mid1_item_name, source=mid1_source)
    mid2_item = ProcedureItem(name=mid2_item_name, source=mid2_source)
    leaf_item = ProcedureItem(name=leaf_item_name, source=leaf_source)

    sgraph = SGraph.from_dict({
        driver_item: [mid1_item],
        mid1_item: [mid2_item],
        mid2_item: [leaf_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, [mid1_name]),
        (mid1_routine, 'kernel', mid1_item, [mid2_name]),
        (mid2_routine, 'kernel', mid2_item, [leaf_name]),
        (leaf_routine, 'kernel', leaf_item, []),
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

    return driver_routine, mid1_routine, mid2_routine, leaf_routine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_bnds_recompute_propagated_to_leaf_kernel(frontend, horizontal, block_dim, tmp_path):
    """
    The main remaining bug: BNDS recompute pattern not propagated through
    ASSOCIATE boundary to leaf kernels.

    This test models the real IFS call chain::

        CPG_2 (driver) -> LASSIE (mid kernel) -> SIGAM_GP (leaf kernel)

    The mid kernel receives ``BNDS`` as a whole struct, unpacks
    ``KIDIA => BNDS%KIDIA`` and ``KFDIA => BNDS%KFDIA`` via ASSOCIATE,
    then calls the leaf kernel with positional args ``(KLON, KLEV,
    KIDIA, KFDIA, T, Q)`` — the leaf's formals are ``(KLON, KLEV,
    KST, KEND, T, Q)``.

    After the full pipeline, the leaf kernel must:

    1. Have ``BNDS`` and ``OPTS`` added to its argument list (so it
       can recompute per-block bounds).
    2. Have a block loop containing::

           local_bnds%kbl = local_ibl
           local_bnds%kstglo = 1 + (local_ibl - 1) * klon
           local_kend = MIN(klon, opts%kgpcomp - local_bnds%kstglo + 1)

    3. Have ``private(local_bnds)`` in ``!$acc parallel loop gang``.

    The mid kernel must similarly have a block loop with the recompute
    pattern and pass ``BNDS``/``OPTS`` as kwargs to the leaf call.

    **This test is expected to FAIL** until the BNDS recompute
    propagation through ASSOCIATE boundaries is fixed.

    Affected IFS files: sigam_mod, sitnu_mod, gpcty_expl, gpgeo_expl,
    gpgrgeo_expl, lattes, cpg_2, cpg_gp (8+ files, Categories 2+3+4
    from the bug inventory).
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    mid_source = Sourcefile.from_source(
        FCODE_MID_KERNEL_BNDS_RECOMPUTE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    leaf_source = Sourcefile.from_source(
        FCODE_LEAF_KERNEL_BNDS_RECOMPUTE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_BNDS_RECOMPUTE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, mid_kernel, leaf_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, mid_source, leaf_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_bnds_recompute',
        mid_name='mid_kernel_bnds_recompute',
        sub_name='leaf_kernel_bnds_recompute',
        driver_item_name='driver_bnds_recompute_mod#driver_bnds_recompute',
        mid_item_name='#mid_kernel_bnds_recompute',
        sub_item_name='#leaf_kernel_bnds_recompute'
    )

    driver_code = fgen(driver)
    mid_code = fgen(mid_kernel)
    leaf_code = fgen(leaf_kernel)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"BNDS recompute test: 3-level hierarchy")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- MID KERNEL ---\n{mid_code}")
    print(f"\n--- LEAF KERNEL ---\n{leaf_code}")
    print(f"{'='*70}\n")

    # ===================================================================
    # DRIVER assertions (sanity checks)
    # ===================================================================

    driver_lower = driver_code.lower()
    assert 'bnds%kstglo' in driver_lower or 'local_bnds%kstglo' in driver_lower, (
        f"Driver should contain KSTGLO assignment in block loop.\n"
        f"Driver code:\n{driver_code}"
    )

    # ===================================================================
    # MID KERNEL assertions (lassie-like)
    # ===================================================================

    mid_lower = mid_code.lower()

    # --- 1. Mid kernel must have a block loop ---
    mid_loops = FindNodes(Loop).visit(mid_kernel.body)
    mid_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                           for l in mid_loops]
    mid_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in mid_block_loop_vars
    )
    assert mid_has_block_loop, (
        f"Mid kernel should have a block-dimension loop.\n"
        f"Loop variables found: {[str(l.variable) for l in mid_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # --- 2. Mid kernel must have local_bnds with KBL and KSTGLO assignments ---
    assert 'local_bnds' in mid_lower or 'local_ydcpg_bnds' in mid_lower, (
        f"Mid kernel should have a 'local_bnds' variable.\n"
        f"Mid kernel code:\n{mid_code}"
    )
    assert 'kstglo' in mid_lower, (
        f"Mid kernel should have KSTGLO assignment in block loop.\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # --- 3. Mid kernel must have private(local_bnds) in !$acc parallel ---
    mid_private_matches = re.findall(r'private\(([^)]+)\)', mid_lower)
    mid_has_local_bnds_private = any(
        'local_bnds' in m or 'local_ydcpg_bnds' in m
        for m in mid_private_matches
    )
    assert mid_has_local_bnds_private, (
        f"Mid kernel should have 'private(local_bnds)' in !$acc directive.\n"
        f"Private clauses found: {mid_private_matches}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # --- 4. Mid kernel must pass BNDS/OPTS as kwargs to leaf kernel call ---
    mid_calls = FindNodes(CallStatement).visit(mid_kernel.body)
    leaf_calls = [c for c in mid_calls
                  if 'leaf_kernel_bnds_recompute' in str(c.name).lower()]
    assert len(leaf_calls) >= 1, (
        f"Mid kernel should call leaf_kernel_bnds_recompute.\n"
        f"All calls: {[str(c.name) for c in mid_calls]}\n"
        f"Mid kernel code:\n{mid_code}"
    )
    for call in leaf_calls:
        kwarg_names = [kw[0].lower() for kw in call.kwarguments] if call.kwarguments else []
        assert any('bnds' in kw for kw in kwarg_names), (
            f"Call to leaf_kernel should have BNDS kwarg.\n"
            f"Found kwargs: {kwarg_names}\n"
            f"Call: {fgen(call)}\n"
            f"Mid kernel code:\n{mid_code}"
        )
        assert any('opts' in kw for kw in kwarg_names), (
            f"Call to leaf_kernel should have OPTS kwarg.\n"
            f"Found kwargs: {kwarg_names}\n"
            f"Call: {fgen(call)}\n"
            f"Mid kernel code:\n{mid_code}"
        )

    # ===================================================================
    # LEAF KERNEL assertions (sigam_gp-like) -- the core bug
    # ===================================================================

    leaf_lower = leaf_code.lower()

    # --- 1. Leaf kernel must have BNDS in its argument list ---
    leaf_arg_names = [str(a).lower() for a in leaf_kernel.arguments]
    print(f"\nLeaf kernel arguments: {leaf_arg_names}")

    assert any('bnds' in a for a in leaf_arg_names), (
        f"Leaf kernel should have BNDS in its argument list "
        f"(added by LowerBlockIndexTransformation for recompute).\n"
        f"Found args: {leaf_arg_names}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )
    assert any('opts' in a for a in leaf_arg_names), (
        f"Leaf kernel should have OPTS in its argument list "
        f"(needed for KGPCOMP in the KEND recompute).\n"
        f"Found args: {leaf_arg_names}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 2. Leaf kernel must have a block-dimension loop ---
    leaf_loops = FindNodes(Loop).visit(leaf_kernel.body)
    leaf_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                            for l in leaf_loops]
    leaf_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in leaf_block_loop_vars
    )
    assert leaf_has_block_loop, (
        f"Leaf kernel should have a block-dimension loop.\n"
        f"Loop variables found: {[str(l.variable) for l in leaf_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 3. Leaf kernel block loop must recompute KSTGLO ---
    has_kstglo_recompute = (
        'local_bnds%kstglo' in leaf_lower
        or 'bnds%kstglo' in leaf_lower
        or ('kstglo' in leaf_lower and '1 +' in leaf_lower)
    )
    assert has_kstglo_recompute, (
        f"Leaf kernel's block loop must recompute KSTGLO.\n"
        f"Expected 'local_bnds%kstglo' or 'bnds%kstglo' or a standalone\n"
        f"'kstglo' assignment in the code.\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 4. Leaf kernel must recompute KEND using MIN(..., KGPCOMP - KSTGLO + 1) ---
    # The recomputed upper horizontal bound should use MIN() with KGPCOMP
    has_kend_min_recompute = (
        'min(' in leaf_lower
        and 'kgpcomp' in leaf_lower
    )
    assert has_kend_min_recompute, (
        f"Leaf kernel should have KEND = MIN(KLON, OPTS%KGPCOMP - ...) "
        f"recompute inside the block loop.\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 5. Leaf kernel must have private(local_bnds) in !$acc parallel ---
    leaf_private_matches = re.findall(r'private\(([^)]+)\)', leaf_lower)
    leaf_has_local_bnds_private = any(
        'local_bnds' in m or 'local_ydcpg_bnds' in m
        for m in leaf_private_matches
    )
    assert leaf_has_local_bnds_private, (
        f"Leaf kernel should have 'private(local_bnds)' in !$acc directive.\n"
        f"Private clauses found: {leaf_private_matches}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 6. Leaf kernel must have OPTS in !$acc data present ---
    leaf_present_matches = re.findall(r'present\(([^)]+)\)', leaf_lower)
    leaf_has_opts_present = any(
        'opts' in m
        for m in leaf_present_matches
    )
    assert leaf_has_opts_present, (
        f"Leaf kernel should have OPTS in '!$acc data present(...)' directive.\n"
        f"Present clauses found: {leaf_present_matches}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 7. Leaf kernel: local_kst and local_kend must be assigned inside
    #    the block loop from local_bnds%kidia and local_bnds%kfdia ---
    #
    # The leaf kernel has formal args KST/KEND which get localized to
    # local_kst/local_kend. These must be assigned per-block from the
    # recomputed BNDS members. Without these assignments, the horizontal
    # loop ``DO jrof = local_kst, local_kend`` uses uninitialized values.
    #
    # Expected pattern inside the block loop:
    #   local_kst = local_bnds%kidia    (or = 1)
    #   local_kend = local_bnds%kfdia   (or = MIN(...))
    leaf_assignments = FindNodes(Assignment).visit(leaf_kernel.body)
    leaf_assign_lhs = [str(a.lhs).lower() for a in leaf_assignments]

    print(f"\nLeaf kernel assignments (LHS):")
    for a in leaf_assignments:
        print(f"  {a.lhs} = {a.rhs}")

    # Check that local_kst is assigned (either from local_bnds%kidia or as = 1)
    has_local_kst_assign = any(
        lhs in ('local_kst', 'local_kidia')
        for lhs in leaf_assign_lhs
    )
    assert has_local_kst_assign, (
        f"Leaf kernel must assign 'local_kst' (or 'local_kidia') inside the "
        f"block loop from local_bnds%kidia or as '= 1'.\n"
        f"Without this, the horizontal loop uses uninitialized bounds.\n"
        f"All assignments: {[(str(a.lhs), str(a.rhs)) for a in leaf_assignments]}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # Check that local_kend is assigned (from local_bnds%kfdia or MIN(...))
    has_local_kend_assign = any(
        lhs in ('local_kend', 'local_kfdia')
        for lhs in leaf_assign_lhs
    )
    assert has_local_kend_assign, (
        f"Leaf kernel must assign 'local_kend' (or 'local_kfdia') inside the "
        f"block loop from local_bnds%kfdia or MIN(...).\n"
        f"Without this, the horizontal loop uses uninitialized bounds.\n"
        f"All assignments: {[(str(a.lhs), str(a.rhs)) for a in leaf_assignments]}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # ===================================================================
    # MID KERNEL: leaf call must be INSIDE the block loop
    # ===================================================================
    #
    # The ``!$loki small-kernels`` pragma on the call inside the ASSOCIATE
    # block can get displaced during the pipeline (Bug A: pragma
    # displacement). When this happens, the call ends up outside the
    # block loop — meaning the leaf kernel is called once instead of
    # per-block. This is wrong: either the call must be inside the mid
    # kernel's block loop, or the leaf kernel must have its own block
    # loop (which it does). But the call must still pass the correct
    # per-block bounds.
    #
    # Check that the leaf call is inside a block loop in the mid kernel.
    mid_block_loops = [l for l in mid_loops
                       if str(l.variable).lower().replace('local_', '') in
                       [idx.lower() for idx in block_dim.indices]]

    # No IT SHOULD NOT BE WITHIN THE BLOCK LOOP since the call has the `loki small-kernels` pragma 
    # if mid_block_loops:
    #     # Check if the leaf call is inside the block loop
    #     calls_in_block_loop = FindNodes(CallStatement).visit(mid_block_loops[0].body)
    #     leaf_calls_in_block = [c for c in calls_in_block_loop
    #                            if 'leaf_kernel_bnds_recompute' in str(c.name).lower()]
    #     assert len(leaf_calls_in_block) >= 1, (
    #         f"Mid kernel's call to leaf_kernel_bnds_recompute should be "
    #         f"INSIDE the block loop (not displaced outside).\n"
    #         f"Calls inside block loop: {[str(c.name) for c in calls_in_block_loop]}\n"
    #         f"This is the Bug A pragma displacement issue: the !$loki "
    #         f"small-kernels pragma on the call inside ASSOCIATE gets "
    #         f"displaced during the pipeline, causing the call to end up "
    #         f"outside the block section.\n"
    #         f"Mid kernel code:\n{mid_code}"
    #     )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_dual_path_driver_preserves_bnds_recompute(frontend, horizontal, block_dim, tmp_path):
    """
    Dual-path driver (GPU + CPU) must preserve BNDS recompute assignments
    for all downstream kernels.

    This test models the real IFS call chain::

        cpg_2_parallel (driver) -> cpg_2 (mid1) -> lassie (mid2) -> sigam_gp (leaf)

    The driver has TWO code paths inside an ``IF (on_gpu) THEN / ELSE``
    conditional:

    * **GPU path**: ``DO IBL`` loop with 4 inline BNDS assignments
      (``bnds%kbl = ibl``, ``bnds%kstglo = ...``, ``bnds%kidia = 1``,
      ``bnds%kfdia = MIN(...)``), followed by ``!$loki small-kernels``
      and ``CALL cpg2_like_kernel(...)``.

    * **CPU path**: ``DO IBL`` loop with ``CALL update_bnds(bnds, ibl,
      opts)`` (modeling ``YLCPG_BNDS%UPDATE(IBL)``), followed by the
      same ``CALL cpg2_like_kernel(...)``.

    ``process_driver`` in ``LowerBlockIndexTransformation`` finds BOTH
    loops via ``find_driver_loops``, because both contain the same
    target call.  When the CPU path is processed **second**, the
    ``update_bnds`` call is stripped (it is a ``CallStatement``),
    leaving an empty driver-loop body.  If this empty body **overwrites**
    the GPU path's result in ``trafo_data['LowerBlockIndex']['driver_loop']``,
    all downstream kernels lose the BNDS recompute assignments.

    After the fix, the first-set (GPU path) driver_loop must be
    preserved, so that ``cpg2_like_kernel``, ``mid_kernel``, and
    ``leaf_kernel`` all get the correct BNDS recompute pattern inside
    their block loops.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DUAL_PATH_DRIVER, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    mid1_source = Sourcefile.from_source(
        FCODE_CPG2_LIKE_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    mid2_source = Sourcefile.from_source(
        FCODE_MID_KERNEL_BNDS_RECOMPUTE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    leaf_source = Sourcefile.from_source(
        FCODE_LEAF_KERNEL_BNDS_RECOMPUTE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, mid1_kernel, mid2_kernel, leaf_kernel = _apply_small_kernels_pipeline_4level(
        driver_source, mid1_source, mid2_source, leaf_source,
        horizontal, block_dim, tmp_path,
        driver_name='dual_path_driver',
        mid1_name='cpg2_like_kernel',
        mid2_name='mid_kernel_bnds_recompute',
        leaf_name='leaf_kernel_bnds_recompute',
        driver_item_name='dual_path_driver_mod#dual_path_driver',
        mid1_item_name='#cpg2_like_kernel',
        mid2_item_name='#mid_kernel_bnds_recompute',
        leaf_item_name='#leaf_kernel_bnds_recompute'
    )

    driver_code = fgen(driver)
    mid1_code = fgen(mid1_kernel)
    mid2_code = fgen(mid2_kernel)
    leaf_code = fgen(leaf_kernel)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"Dual-path driver test: 4-level hierarchy")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- MID1 KERNEL (cpg2-like) ---\n{mid1_code}")
    print(f"\n--- MID2 KERNEL (lassie-like) ---\n{mid2_code}")
    print(f"\n--- LEAF KERNEL (sigam_gp-like) ---\n{leaf_code}")
    print(f"{'='*70}\n")

    mid1_lower = mid1_code.lower()
    mid2_lower = mid2_code.lower()
    leaf_lower = leaf_code.lower()

    # ===================================================================
    # MID1 KERNEL assertions (cpg_2-like — receives BNDS struct directly)
    # ===================================================================

    # --- 1. Must have a block loop ---
    mid1_loops = FindNodes(Loop).visit(mid1_kernel.body)
    mid1_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                            for l in mid1_loops]
    mid1_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in mid1_block_loop_vars
    )
    assert mid1_has_block_loop, (
        f"Mid1 kernel (cpg2-like) should have a block-dimension loop.\n"
        f"Loop variables found: {[str(l.variable) for l in mid1_loops]}\n"
        f"Mid1 kernel code:\n{mid1_code}"
    )

    # --- 2. Block loop must contain KSTGLO recompute ---
    assert 'kstglo' in mid1_lower, (
        f"Mid1 kernel (cpg2-like) should have KSTGLO assignment in block loop.\n"
        f"This fails when the CPU path's empty driver_loop overwrites the\n"
        f"GPU path's BNDS assignments during process_driver.\n"
        f"Mid1 kernel code:\n{mid1_code}"
    )

    # --- 3. Must have local_bnds variable ---
    assert 'local_bnds' in mid1_lower or 'local_ydcpg_bnds' in mid1_lower, (
        f"Mid1 kernel (cpg2-like) should have a 'local_bnds' variable.\n"
        f"Mid1 kernel code:\n{mid1_code}"
    )

    # --- 4. Must have private(local_bnds) in !$acc parallel ---
    mid1_private_matches = re.findall(r'private\(([^)]+)\)', mid1_lower)
    mid1_has_local_bnds_private = any(
        'local_bnds' in m or 'local_ydcpg_bnds' in m
        for m in mid1_private_matches
    )
    assert mid1_has_local_bnds_private, (
        f"Mid1 kernel should have 'private(local_bnds)' in !$acc directive.\n"
        f"Private clauses found: {mid1_private_matches}\n"
        f"Mid1 kernel code:\n{mid1_code}"
    )

    # ===================================================================
    # MID2 KERNEL assertions (lassie-like — ASSOCIATE unpacking of BNDS)
    # ===================================================================

    # --- 1. Must have a block loop ---
    mid2_loops = FindNodes(Loop).visit(mid2_kernel.body)
    mid2_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                            for l in mid2_loops]
    mid2_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in mid2_block_loop_vars
    )
    assert mid2_has_block_loop, (
        f"Mid2 kernel (lassie-like) should have a block-dimension loop.\n"
        f"Loop variables found: {[str(l.variable) for l in mid2_loops]}\n"
        f"Mid2 kernel code:\n{mid2_code}"
    )

    # --- 2. Must have KSTGLO recompute ---
    assert 'kstglo' in mid2_lower, (
        f"Mid2 kernel (lassie-like) should have KSTGLO assignment in block loop.\n"
        f"Mid2 kernel code:\n{mid2_code}"
    )

    # --- 3. Must pass BNDS/OPTS as kwargs to leaf kernel call ---
    mid2_calls = FindNodes(CallStatement).visit(mid2_kernel.body)
    leaf_calls_from_mid2 = [c for c in mid2_calls
                            if 'leaf_kernel_bnds_recompute' in str(c.name).lower()]
    assert len(leaf_calls_from_mid2) >= 1, (
        f"Mid2 kernel should call leaf_kernel_bnds_recompute.\n"
        f"All calls: {[str(c.name) for c in mid2_calls]}\n"
        f"Mid2 kernel code:\n{mid2_code}"
    )
    for call in leaf_calls_from_mid2:
        kwarg_names = [kw[0].lower() for kw in call.kwarguments] if call.kwarguments else []
        assert any('bnds' in kw for kw in kwarg_names), (
            f"Mid2 kernel's call to leaf_kernel should have BNDS kwarg.\n"
            f"Found kwargs: {kwarg_names}\n"
            f"Call: {fgen(call)}\n"
            f"Mid2 kernel code:\n{mid2_code}"
        )
        assert any('opts' in kw for kw in kwarg_names), (
            f"Mid2 kernel's call to leaf_kernel should have OPTS kwarg.\n"
            f"Found kwargs: {kwarg_names}\n"
            f"Call: {fgen(call)}\n"
            f"Mid2 kernel code:\n{mid2_code}"
        )

    # --- 4. Must have private(local_bnds) in !$acc parallel ---
    mid2_private_matches = re.findall(r'private\(([^)]+)\)', mid2_lower)
    mid2_has_local_bnds_private = any(
        'local_bnds' in m or 'local_ydcpg_bnds' in m
        for m in mid2_private_matches
    )
    assert mid2_has_local_bnds_private, (
        f"Mid2 kernel should have 'private(local_bnds)' in !$acc directive.\n"
        f"Private clauses found: {mid2_private_matches}\n"
        f"Mid2 kernel code:\n{mid2_code}"
    )

    # ===================================================================
    # LEAF KERNEL assertions (sigam_gp-like — KST/KEND positional args)
    # ===================================================================

    # --- 1. Must have BNDS and OPTS in argument list ---
    leaf_arg_names = [str(a).lower() for a in leaf_kernel.arguments]
    print(f"\nLeaf kernel arguments: {leaf_arg_names}")

    assert any('bnds' in a for a in leaf_arg_names), (
        f"Leaf kernel should have BNDS in its argument list.\n"
        f"Found args: {leaf_arg_names}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )
    assert any('opts' in a for a in leaf_arg_names), (
        f"Leaf kernel should have OPTS in its argument list.\n"
        f"Found args: {leaf_arg_names}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 2. Must have a block-dimension loop ---
    leaf_loops = FindNodes(Loop).visit(leaf_kernel.body)
    leaf_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                            for l in leaf_loops]
    leaf_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in leaf_block_loop_vars
    )
    assert leaf_has_block_loop, (
        f"Leaf kernel should have a block-dimension loop.\n"
        f"Loop variables found: {[str(l.variable) for l in leaf_loops]}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 3. Block loop must recompute KSTGLO ---
    has_kstglo_recompute = (
        'local_bnds%kstglo' in leaf_lower
        or 'bnds%kstglo' in leaf_lower
        or ('kstglo' in leaf_lower and '1 +' in leaf_lower)
    )
    assert has_kstglo_recompute, (
        f"Leaf kernel's block loop must recompute KSTGLO.\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 4. Must recompute KEND using MIN(..., KGPCOMP - KSTGLO + 1) ---
    has_kend_min_recompute = (
        'min(' in leaf_lower
        and 'kgpcomp' in leaf_lower
    )
    assert has_kend_min_recompute, (
        f"Leaf kernel should have KEND = MIN(KLON, OPTS%KGPCOMP - ...) "
        f"recompute inside the block loop.\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 5. Must have private(local_bnds) in !$acc parallel ---
    leaf_private_matches = re.findall(r'private\(([^)]+)\)', leaf_lower)
    leaf_has_local_bnds_private = any(
        'local_bnds' in m or 'local_ydcpg_bnds' in m
        for m in leaf_private_matches
    )
    assert leaf_has_local_bnds_private, (
        f"Leaf kernel should have 'private(local_bnds)' in !$acc directive.\n"
        f"Private clauses found: {leaf_private_matches}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    # --- 6. local_kst and local_kend must be assigned ---
    leaf_assignments = FindNodes(Assignment).visit(leaf_kernel.body)
    leaf_assign_lhs = [str(a.lhs).lower() for a in leaf_assignments]

    print(f"\nLeaf kernel assignments (LHS):")
    for a in leaf_assignments:
        print(f"  {a.lhs} = {a.rhs}")

    has_local_kst_assign = any(
        lhs in ('local_kst', 'local_kidia')
        for lhs in leaf_assign_lhs
    )
    assert has_local_kst_assign, (
        f"Leaf kernel must assign 'local_kst' (or 'local_kidia') inside the "
        f"block loop.\n"
        f"All assignments: {[(str(a.lhs), str(a.rhs)) for a in leaf_assignments]}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )

    has_local_kend_assign = any(
        lhs in ('local_kend', 'local_kfdia')
        for lhs in leaf_assign_lhs
    )
    assert has_local_kend_assign, (
        f"Leaf kernel must assign 'local_kend' (or 'local_kfdia') inside the "
        f"block loop.\n"
        f"All assignments: {[(str(a.lhs), str(a.rhs)) for a in leaf_assignments]}\n"
        f"Leaf kernel code:\n{leaf_code}"
    )


# ---------------------------------------------------------------------------
# Test: acc enter/exit data must stay outside the block loop
#
# Models the sigam_gp pattern: a leaf kernel with local temporaries
# allocated via !$acc enter data create(...) before compute loops,
# and !$acc exit data delete(...) after.  No !$loki small-kernels
# pragmas on internal calls (there are none -- it's a leaf).
#
# When extract_block_sections returns empty (no separator calls), the
# fallback wraps the ENTIRE body as one block section, pulling the
# acc unstructured data directives inside the !$acc parallel loop gang
# block loop.  This is invalid OpenACC -- unstructured enter/exit data
# must execute on the host, not inside a device-side parallel region.
#
# Expected:
#   - Compute sections ARE inside the block loop
#   - !$acc enter data create(...) is OUTSIDE the block loop
#   - !$acc exit data delete(...) is OUTSIDE the block loop
# ---------------------------------------------------------------------------

FCODE_DRIVER_ACC_DATA = """
module driver_acc_data_mod
  implicit none
contains
  subroutine driver_acc_data(ngpblks, bnds, opts, pd, pt, psp)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_acc_data.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(out)   :: pd(:,:,:)
    real, intent(in)    :: pt(:,:,:)
    real, intent(in)    :: psp(:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_acc_data(opts%klon, opts%kflevg, bnds, opts, pd(:,:,ibl), pt(:,:,ibl), psp(:,ibl))
    end do
  end subroutine driver_acc_data
end module driver_acc_data_mod
""".strip()

FCODE_KERNEL_ACC_DATA = """
subroutine kernel_acc_data(klon, klev, bnds, opts, pd, pt, psp)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in)  :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(out) :: pd(klon, klev)
  real, intent(in)  :: pt(klon, klev)
  real, intent(in)  :: psp(klon)

  ! Local temporaries that get !$acc enter/exit data directives
  real :: ztmp(klon, 0:klev+1)
  real :: zout(klon, klev)

  integer :: jrof, jk

  !$acc enter data create(ztmp, zout)

  ! Compute section using horizontal bounds
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp(jrof, jk) = pt(jrof, jk) * 2.0
    end do
  end do

  do jrof = bnds%kidia, bnds%kfdia
    ztmp(jrof, 0) = 0.0
    ztmp(jrof, klev + 1) = 0.0
  end do

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      zout(jrof, jk) = ztmp(jrof, jk) + psp(jrof)
    end do
  end do

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      pd(jrof, jk) = zout(jrof, jk)
    end do
  end do

  !$acc exit data delete(ztmp, zout)

end subroutine kernel_acc_data
""".strip()


# ---------------------------------------------------------------------------
# Test: YDSTACK_L kwarg propagation to sub-kernel calls
#
# Models the lavabo pattern: a kernel (caller) calls a sub-kernel
# (callee).  The pool allocator adds YDSTACK_L as a formal argument
# to ALL kernels unconditionally.  However, it only propagates
# YDSTACK_L=YLSTACK_L to the caller's call statement when
# stack_size != 0.  If a kernel has no local temporaries (stack_size
# == 0), inject_pool_allocator_into_calls is never reached, so the
# caller's call to the callee is missing YDSTACK_L.
#
# Expected:
#   - The callee has YDSTACK_L in its argument list
#   - The caller's call to the callee has YDSTACK_L=YLSTACK_L as kwarg
# ---------------------------------------------------------------------------

FCODE_DRIVER_STACK_PROP = """
module driver_stack_prop_mod
  implicit none
contains
  subroutine driver_stack_prop(ngpblks, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "caller_kernel_stack.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call caller_kernel_stack(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
  end subroutine driver_stack_prop
end module driver_stack_prop_mod
""".strip()

FCODE_CALLER_KERNEL_STACK = """
subroutine caller_kernel_stack(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "callee_kernel_stack.intfb.h"
  integer, intent(in)  :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: pfield(klon, klev)

  integer :: jrof, jk

  ! Some local compute (no local temporaries -- important!)
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      pfield(jrof, jk) = pfield(jrof, jk) + 1.0
    end do
  end do

  ! Call sub-kernel (callee has local temporaries that trigger pool allocator)
  !$loki small-kernels
  call callee_kernel_stack(klon, klev, bnds, opts, pfield)

end subroutine caller_kernel_stack
""".strip()

FCODE_CALLEE_KERNEL_STACK = """
subroutine callee_kernel_stack(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in)  :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: pfield(klon, klev)

  ! Local temporaries that trigger pool allocation
  real :: ztmp1(klon, klev)
  real :: ztmp2(klon)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp1(jrof, jk) = pfield(jrof, jk) * 2.0
    end do
  end do

  do jrof = bnds%kidia, bnds%kfdia
    ztmp2(jrof) = 0.0
  end do

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp2(jrof) = ztmp2(jrof) + ztmp1(jrof, jk)
    end do
  end do

  do jrof = bnds%kidia, bnds%kfdia
    pfield(jrof, 1) = ztmp2(jrof)
  end do

end subroutine callee_kernel_stack
""".strip()


# ---------------------------------------------------------------------------
# Test: thin wrapper should NOT generate a block loop
#
# Models the gpgeo pattern: a thin wrapper kernel that just has
# !$loki small-kernels + CALL inner_kernel(...).  The block loop
# belongs in the inner kernel, NOT in the wrapper.
#
# gpgeo.F90 original:
#   !$loki small-kernels
#   CALL GPGEO_EXPL(KPROMA, KST, KEND, ...)
#
# Working output (correct):
#   gpgeo has NO block loop, just a direct CALL GPGEO_EXPL_LOKI(...)
#   gpgeo_expl HAS the block loop with BNDS recompute.
#
# BUILD4 output (bug):
#   gpgeo has a block loop wrapping the CALL GPGEO_EXPL_LOKI(...)
#   This is wrong because the block loop belongs INSIDE gpgeo_expl.
#
# Root cause: the fallback in block.py wraps the entire body (including
# the call) as one block section, then ReblockSectionTransformer wraps
# it in a block loop.  The call itself should NOT be inside a block
# section since it has !$loki small-kernels -- it IS a separator.
# ---------------------------------------------------------------------------

FCODE_DRIVER_THIN_WRAPPER = """
module driver_thin_wrapper_mod
  implicit none
contains
  subroutine driver_thin_wrapper(ngpblks, bnds, opts, phi, pt)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "thin_wrapper_kernel.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: phi(:,:,:)
    real, intent(in) :: pt(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call thin_wrapper_kernel(opts%klon, opts%kflevg, bnds%kidia, bnds%kfdia, phi(:,:,ibl), pt(:,:,ibl))
    end do
  end subroutine driver_thin_wrapper
end module driver_thin_wrapper_mod
""".strip()

FCODE_THIN_WRAPPER_KERNEL = """
subroutine thin_wrapper_kernel(klon, klev, kst, kend, phi, pt)
  implicit none
  #include "inner_compute_kernel.intfb.h"
  integer, intent(in)  :: klon, klev, kst, kend
  real, intent(inout) :: phi(klon, klev)
  real, intent(in) :: pt(klon, klev)

  ! Thin wrapper: just delegates to inner kernel.
  ! The block loop belongs INSIDE inner_compute_kernel, not here.

  !$loki small-kernels
  call inner_compute_kernel(klon, klev, kst, kend, phi, pt)

end subroutine thin_wrapper_kernel
""".strip()

FCODE_INNER_COMPUTE_KERNEL = """
subroutine inner_compute_kernel(klon, klev, kst, kend, phi, pt)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  real, intent(inout) :: phi(klon, klev)
  real, intent(in) :: pt(klon, klev)

  real :: ztmp(klon)
  integer :: jrof, jk

  do jrof = kst, kend
    ztmp(jrof) = 0.0
  end do

  do jk = 1, klev
    do jrof = kst, kend
      ztmp(jrof) = ztmp(jrof) + pt(jrof, jk)
    end do
  end do

  do jk = 1, klev
    do jrof = kst, kend
      phi(jrof, jk) = phi(jrof, jk) + ztmp(jrof)
    end do
  end do

end subroutine inner_compute_kernel
""".strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_acc_enter_exit_data_stays_outside_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    Unstructured !$acc enter/exit data directives must remain outside
    the block loop.

    This test models the sigam_gp pattern: a leaf kernel with local
    temporaries guarded by ``!$acc enter data create(...)`` before
    the compute loops and ``!$acc exit data delete(...)`` after.  The
    kernel has NO ``!$loki small-kernels`` pragma on any internal call
    (it is a leaf with no calls), so ``extract_block_sections`` returns
    empty and the fallback wraps the entire body as one block section.

    The fallback must NOT include the ``!$acc enter data`` and
    ``!$acc exit data`` pragmas in the block section, because these
    are host-side unstructured data directives that cannot appear
    inside a device-side ``!$acc parallel loop gang`` region.

    Affected IFS files: sigam_mod, sitnu_mod, lassie, lattes,
    lacdyn, cpg_gp_hyd (6 files).
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_ACC_DATA, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_ACC_DATA, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc', trim_vector_sections=True
    )

    driver_routine = driver_source['driver_acc_data']
    kernel_routine = kernel_source['kernel_acc_data']

    driver_routine.enrich(kernel_routine)

    driver_item = ProcedureItem(name='driver_acc_data_mod#driver_acc_data', source=driver_source)
    kernel_item = ProcedureItem(name='#kernel_acc_data', source=kernel_source)

    sgraph = SGraph.from_dict({
        driver_item: [kernel_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, ['kernel_acc_data']),
        (kernel_routine, 'kernel', kernel_item, []),
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

    kernel_code = fgen(kernel_routine)
    kernel_lower = kernel_code.lower()

    # Print for diagnostics
    print(f"\n{'='*70}")
    print(f"ACC enter/exit data test")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{fgen(driver_routine)}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"{'='*70}\n")

    # ===================================================================
    # 1. Kernel must have a block loop (from the fallback)
    # ===================================================================
    kernel_loops = FindNodes(Loop).visit(kernel_routine.body)
    block_loop_vars = [str(l.variable).lower().replace('local_', '')
                       for l in kernel_loops]
    has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in block_loop_vars
    )
    assert has_block_loop, (
        f"Kernel should have a block-dimension loop.\n"
        f"Loop variables found: {[str(l.variable) for l in kernel_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # ===================================================================
    # 2. Find the block loop
    # ===================================================================
    block_loops = [l for l in kernel_loops
                   if str(l.variable).lower().replace('local_', '') in
                   [idx.lower() for idx in block_dim.indices]]
    assert len(block_loops) >= 1, (
        f"Expected at least one block loop.\n"
        f"Kernel code:\n{kernel_code}"
    )
    block_loop = block_loops[0]
    block_loop_code = fgen(block_loop).lower()

    # ===================================================================
    # 3. !$acc enter data create(...) must NOT be inside the block loop
    # ===================================================================
    # Look for pragmas inside the block loop body
    block_loop_pragmas = FindNodes(Pragma).visit(block_loop.body)
    enter_data_inside = [p for p in block_loop_pragmas
                         if p.keyword.lower() == 'acc'
                         and 'enter' in p.content.lower()
                         and 'data' in p.content.lower()
                         and 'create' in p.content.lower()]

    # Also check for !$loki unstructured-data pragmas (pipeline may
    # convert !$acc enter data to !$loki unstructured-data)
    loki_enter_inside = [p for p in block_loop_pragmas
                         if p.keyword.lower() == 'loki'
                         and 'unstructured-data' in p.content.lower()
                         and 'exit' not in p.content.lower()]

    all_enter_inside = enter_data_inside + loki_enter_inside
    assert len(all_enter_inside) == 0, (
        f"!$acc enter data create(...) / !$loki unstructured-data must NOT\n"
        f"be inside the block loop! These are host-side unstructured data\n"
        f"directives that cannot appear inside a device-side !$acc parallel\n"
        f"loop gang region.\n"
        f"Found {len(all_enter_inside)} enter-data pragma(s) inside block loop:\n"
        f"  {[fgen(p) for p in all_enter_inside]}\n"
        f"Block loop code:\n{block_loop_code}\n"
        f"Full kernel code:\n{kernel_code}"
    )

    # ===================================================================
    # 4. !$acc exit data delete(...) must NOT be inside the block loop
    # ===================================================================
    exit_data_inside = [p for p in block_loop_pragmas
                        if p.keyword.lower() == 'acc'
                        and 'exit' in p.content.lower()
                        and 'data' in p.content.lower()
                        and 'delete' in p.content.lower()]

    loki_exit_inside = [p for p in block_loop_pragmas
                        if p.keyword.lower() == 'loki'
                        and 'exit' in p.content.lower()
                        and 'unstructured-data' in p.content.lower()]

    all_exit_inside = exit_data_inside + loki_exit_inside
    assert len(all_exit_inside) == 0, (
        f"!$acc exit data delete(...) / !$loki exit unstructured-data must NOT\n"
        f"be inside the block loop! These are host-side unstructured data\n"
        f"directives that cannot appear inside a device-side !$acc parallel\n"
        f"loop gang region.\n"
        f"Found {len(all_exit_inside)} exit-data pragma(s) inside block loop:\n"
        f"  {[fgen(p) for p in all_exit_inside]}\n"
        f"Block loop code:\n{block_loop_code}\n"
        f"Full kernel code:\n{kernel_code}"
    )

    # ===================================================================
    # 5. Compute sections (loops with horizontal bounds) ARE inside block loop
    # ===================================================================
    loops_in_block = FindNodes(Loop).visit(block_loop.body)
    compute_loops = [l for l in loops_in_block
                     if str(l.variable).lower() in ('jrof',)]
    assert len(compute_loops) > 0, (
        f"At least one compute loop (DO JROF=...) should be inside the block loop.\n"
        f"Loops found inside block loop: {[str(l.variable) for l in loops_in_block]}\n"
        f"Kernel code:\n{kernel_code}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_stack_arg_propagated_to_sub_kernel_call(frontend, horizontal, block_dim, tmp_path):
    """
    YDSTACK_L must be passed from a caller kernel to a callee kernel
    when the callee has YDSTACK_L in its argument list.

    This test models the lavabo pattern: a caller kernel (with no
    local temporaries) calls a callee kernel (which has local
    temporaries that trigger pool allocation).  The pool allocator
    adds YDSTACK_L to ALL kernel argument lists unconditionally via
    ``_get_stack_arg``, but ``inject_pool_allocator_into_calls`` is
    only reached when ``stack_size != 0``.  If the caller kernel
    itself has no temporaries AND the pool allocator's
    ``_determine_stack_size`` computes 0 for the caller, the call to
    ``inject_pool_allocator_into_calls`` is skipped and the caller's
    call to the callee is missing the YDSTACK_L kwarg.

    Affected IFS files: lavabo (call to lavabo_expl_laitvspcqm_part1).
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_STACK_PROP, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    caller_source = Sourcefile.from_source(
        FCODE_CALLER_KERNEL_STACK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    callee_source = Sourcefile.from_source(
        FCODE_CALLEE_KERNEL_STACK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, caller_kernel, callee_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, caller_source, callee_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_stack_prop',
        mid_name='caller_kernel_stack',
        sub_name='callee_kernel_stack',
        driver_item_name='driver_stack_prop_mod#driver_stack_prop',
        mid_item_name='#caller_kernel_stack',
        sub_item_name='#callee_kernel_stack'
    )

    driver_code = fgen(driver)
    caller_code = fgen(caller_kernel)
    callee_code = fgen(callee_kernel)

    # Print for diagnostics
    print(f"\n{'='*70}")
    print(f"YDSTACK_L propagation test")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- CALLER KERNEL ---\n{caller_code}")
    print(f"\n--- CALLEE KERNEL ---\n{callee_code}")
    print(f"{'='*70}\n")

    callee_lower = callee_code.lower()
    caller_lower = caller_code.lower()

    # ===================================================================
    # 1. Callee kernel must have YDSTACK_L in its argument list
    # ===================================================================
    callee_arg_names = [str(a).lower() for a in callee_kernel.arguments]
    print(f"\nCallee kernel arguments: {callee_arg_names}")

    has_stack_arg = any('stack' in a for a in callee_arg_names)
    assert has_stack_arg, (
        f"Callee kernel should have YDSTACK_L (or similar stack arg) in its\n"
        f"argument list. The pool allocator adds this unconditionally via\n"
        f"_get_stack_arg to all kernel routines.\n"
        f"Found args: {callee_arg_names}\n"
        f"Callee kernel code:\n{callee_code}"
    )

    # ===================================================================
    # 2. Caller kernel must have YDSTACK_L in its argument list too
    # ===================================================================
    caller_arg_names = [str(a).lower() for a in caller_kernel.arguments]
    print(f"\nCaller kernel arguments: {caller_arg_names}")

    caller_has_stack_arg = any('stack' in a for a in caller_arg_names)
    assert caller_has_stack_arg, (
        f"Caller kernel should also have YDSTACK_L in its argument list.\n"
        f"Found args: {caller_arg_names}\n"
        f"Caller kernel code:\n{caller_code}"
    )

    # ===================================================================
    # 3. Caller's call to callee must include YDSTACK_L as kwarg
    # ===================================================================
    caller_calls = FindNodes(CallStatement).visit(caller_kernel.body)
    callee_calls = [c for c in caller_calls
                    if 'callee_kernel_stack' in str(c.name).lower()]
    assert len(callee_calls) >= 1, (
        f"Caller kernel should call callee_kernel_stack.\n"
        f"All calls: {[str(c.name) for c in caller_calls]}\n"
        f"Caller kernel code:\n{caller_code}"
    )

    for call in callee_calls:
        call_code = fgen(call)
        # Check all arguments (positional + keyword) for stack reference
        all_arg_strs = [str(a).lower() for a in call.arguments]
        kwarg_names = [kw[0].lower() for kw in call.kwarguments] if call.kwarguments else []
        kwarg_vals = [str(kw[1]).lower() for kw in call.kwarguments] if call.kwarguments else []

        print(f"\nCall to callee: {call_code}")
        print(f"  Positional args: {all_arg_strs}")
        print(f"  Kwargs (names): {kwarg_names}")
        print(f"  Kwargs (vals):  {kwarg_vals}")

        has_stack_kwarg = any('stack' in kw for kw in kwarg_names)
        has_stack_in_args = any('stack' in a for a in all_arg_strs)
        assert has_stack_kwarg or has_stack_in_args, (
            f"Caller's call to callee_kernel_stack must pass YDSTACK_L.\n"
            f"The callee has YDSTACK_L in its formal argument list, so the\n"
            f"caller must pass it. This is the lavabo bug: the pool allocator\n"
            f"adds YDSTACK_L to all kernel signatures but only propagates it\n"
            f"to the call when stack_size != 0 for the caller.\n"
            f"Found kwarg names: {kwarg_names}\n"
            f"Found positional args: {all_arg_strs}\n"
            f"Call: {call_code}\n"
            f"Caller kernel code:\n{caller_code}"
        )

    # ===================================================================
    # 4. Driver's call to caller must also include YDSTACK_L
    # ===================================================================
    driver_calls = FindNodes(CallStatement).visit(driver.body)
    caller_calls_from_driver = [c for c in driver_calls
                                if 'caller_kernel_stack' in str(c.name).lower()]
    assert len(caller_calls_from_driver) >= 1, (
        f"Driver should call caller_kernel_stack.\n"
        f"All calls: {[str(c.name) for c in driver_calls]}\n"
        f"Driver code:\n{driver_code}"
    )
    for call in caller_calls_from_driver:
        call_code = fgen(call)
        kwarg_names = [kw[0].lower() for kw in call.kwarguments] if call.kwarguments else []
        all_arg_strs = [str(a).lower() for a in call.arguments]
        has_stack_kwarg = any('stack' in kw for kw in kwarg_names)
        has_stack_in_args = any('stack' in a for a in all_arg_strs)
        assert has_stack_kwarg or has_stack_in_args, (
            f"Driver's call to caller_kernel_stack must pass YDSTACK_L.\n"
            f"Found kwarg names: {kwarg_names}\n"
            f"Found positional args: {all_arg_strs}\n"
            f"Call: {call_code}\n"
            f"Driver code:\n{driver_code}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_thin_wrapper_call_not_wrapped_in_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    A thin wrapper kernel (like gpgeo) should NOT have a block loop
    wrapping its call to the inner kernel.

    This test models the gpgeo pattern: a thin wrapper kernel that
    just has ``!$loki small-kernels`` + ``CALL inner_kernel(...)``.
    The block loop belongs inside the inner kernel, not in the wrapper.

    In the working IFS output, gpgeo has NO block loop — just a direct
    call to ``GPGEO_EXPL_LOKI``.  In the BUILD4 buggy output, gpgeo
    has a ``!$acc parallel loop gang`` block loop wrapping the call.

    Root cause: ``extract_block_sections`` finds the call as a
    separator (due to ``!$loki small-kernels``), creating empty
    subsections on either side.  But the fallback wraps the entire
    body as one section when ``extract_block_sections`` returns empty.
    Since the wrapper has no other compute referencing block_dim
    indices, the fallback should return empty and the wrapper should
    have no block loop at all.

    Affected IFS files: gpgeo.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_THIN_WRAPPER, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    wrapper_source = Sourcefile.from_source(
        FCODE_THIN_WRAPPER_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    inner_source = Sourcefile.from_source(
        FCODE_INNER_COMPUTE_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    driver, wrapper_kernel, inner_kernel = _apply_small_kernels_pipeline_3level(
        driver_source, wrapper_source, inner_source,
        horizontal, block_dim, tmp_path,
        driver_name='driver_thin_wrapper',
        mid_name='thin_wrapper_kernel',
        sub_name='inner_compute_kernel',
        driver_item_name='driver_thin_wrapper_mod#driver_thin_wrapper',
        mid_item_name='#thin_wrapper_kernel',
        sub_item_name='#inner_compute_kernel'
    )

    driver_code = fgen(driver)
    wrapper_code = fgen(wrapper_kernel)
    inner_code = fgen(inner_kernel)

    # Print for diagnostics
    print(f"\n{'='*70}")
    print(f"Thin wrapper test (gpgeo pattern)")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- WRAPPER KERNEL ---\n{wrapper_code}")
    print(f"\n--- INNER KERNEL ---\n{inner_code}")
    print(f"{'='*70}\n")

    wrapper_lower = wrapper_code.lower()
    inner_lower = inner_code.lower()

    # ===================================================================
    # 1. Wrapper kernel must NOT have a block loop
    # ===================================================================
    wrapper_loops = FindNodes(Loop).visit(wrapper_kernel.body)
    wrapper_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                               for l in wrapper_loops]
    wrapper_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in wrapper_block_loop_vars
    )
    assert not wrapper_has_block_loop, (
        f"Thin wrapper kernel should NOT have a block-dimension loop.\n"
        f"The block loop belongs inside the inner kernel, not in the\n"
        f"wrapper.  This is the gpgeo bug: the fallback in block.py\n"
        f"wraps the entire wrapper body as one block section, inserting\n"
        f"a block loop around the call to the inner kernel.\n"
        f"Loop variables found: {[str(l.variable) for l in wrapper_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Wrapper kernel code:\n{wrapper_code}"
    )

    # ===================================================================
    # 2. Wrapper kernel must still have the call to inner kernel
    # ===================================================================
    wrapper_calls = FindNodes(CallStatement).visit(wrapper_kernel.body)
    inner_calls = [c for c in wrapper_calls
                   if 'inner_compute_kernel' in str(c.name).lower()]
    assert len(inner_calls) >= 1, (
        f"Wrapper kernel should still call inner_compute_kernel.\n"
        f"All calls: {[str(c.name) for c in wrapper_calls]}\n"
        f"Wrapper kernel code:\n{wrapper_code}"
    )

    # ===================================================================
    # 3. Inner kernel MUST have a block loop (it's a leaf with compute)
    # ===================================================================
    inner_loops = FindNodes(Loop).visit(inner_kernel.body)
    inner_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                             for l in inner_loops]
    inner_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in inner_block_loop_vars
    )
    assert inner_has_block_loop, (
        f"Inner kernel should have a block-dimension loop.\n"
        f"The block loop belongs here, not in the wrapper.\n"
        f"Loop variables found: {[str(l.variable) for l in inner_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Inner kernel code:\n{inner_code}"
    )

    # ===================================================================
    # 4. Inner kernel's block loop must contain KSTGLO recompute
    # ===================================================================
    assert 'kstglo' in inner_lower, (
        f"Inner kernel should have KSTGLO assignment in block loop.\n"
        f"Inner kernel code:\n{inner_code}"
    )

    # ===================================================================
    # 5. Inner kernel's block loop must contain compute loops
    # ===================================================================
    inner_block_loops = [l for l in inner_loops
                         if str(l.variable).lower().replace('local_', '') in
                         [idx.lower() for idx in block_dim.indices]]
    if inner_block_loops:
        loops_in_block = FindNodes(Loop).visit(inner_block_loops[0].body)
        compute_loops_in_block = [l for l in loops_in_block
                                  if str(l.variable).lower() in ('jrof',)]
        assert len(compute_loops_in_block) > 0, (
            f"Inner kernel's block loop should contain compute loops.\n"
            f"Loops found: {[str(l.variable) for l in loops_in_block]}\n"
            f"Inner kernel code:\n{inner_code}"
        )


# ---------------------------------------------------------------------------
# Test: call to transformed kernel WITHOUT !$loki small-kernels pragma
#       must still be wrapped in a block loop
#
# Models the lacdyn pattern: a mid-level kernel (lacdyn) has:
#   - Multiple !$loki small-kernels calls (LASSIE, LATTES) that act
#     as separator nodes in extract_block_sections
#   - A call to LAVABO that has NO !$loki small-kernels pragma but is
#     still a Loki-transformed sub-kernel that needs to be called
#     per-block inside the block loop
#
# In lacdyn.F90, the structure is:
#   IF (...) THEN
#     ...                          ! (removed branches)
#   ELSE
#     !$loki small-kernels
#     CALL LASSIE(...)             ! separator 1
#   END IF
#   CALL LAVENT(...)               ! non-separator call
#   CALL LATTEX(...)               ! non-separator call
#   !$loki small-kernels
#   CALL LATTES(...)               ! separator 2
#   ...
#   IF (iteration_condition) THEN
#     CALL LAVABO(...)             ! NO pragma! But still a transformed kernel
#   END IF
#
# The LAVABO call sits in the subsection AFTER the last separator
# (LATTES).  extract_block_sections filters subsections to keep only
# those referencing block_dim.indices.  The IF-wrapped LAVABO call
# doesn't directly reference IBL or BNDS%KBL, so it gets filtered out.
# Result: LAVABO is called once outside any block loop rather than
# per-block.
#
# Expected: The call to the unpragma'd kernel (LAVABO-like) must end up
# inside a block loop, either by:
#   a) The subsection being included as a block section, OR
#   b) A separate block loop being generated around this call
# ---------------------------------------------------------------------------

FCODE_DRIVER_LACDYN_LIKE = """
module driver_lacdyn_like_mod
  implicit none
contains
  subroutine driver_lacdyn_like(ngpblks, bnds, opts, t, q, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "mid_kernel_lacdyn_like.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)
    real, intent(inout) :: pfield(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call mid_kernel_lacdyn_like(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl), pfield(:,:,ibl))
    end do
  end subroutine driver_lacdyn_like
end module driver_lacdyn_like_mod
""".strip()

FCODE_MID_KERNEL_LACDYN_LIKE = """
subroutine mid_kernel_lacdyn_like(klon, klev, bnds, opts, t, q, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "sub_kernel_lassie_like.intfb.h"
  #include "sub_kernel_lattes_like.intfb.h"
  #include "sub_kernel_lavabo_like.intfb.h"
  integer, intent(in)  :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)
  real, intent(inout) :: pfield(klon, klev)

  integer :: jrof, jk
  logical :: ldo_lavabo

  ldo_lavabo = .true.

  ! ---- Section 1: horizontal compute + pragmaed call (models LASSIE) ----
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do

  !$loki small-kernels
  call sub_kernel_lassie_like(klon, klev, bnds, opts, t, q)

  ! ---- Section 2: horizontal compute + pragmaed call (models LATTES) ----
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = q(jrof, jk) * 0.5
    end do
  end do

  !$loki small-kernels
  call sub_kernel_lattes_like(klon, klev, bnds, opts, q)

  ! ---- Section 3: conditional call WITHOUT !$loki small-kernels ----
  ! This models the LAVABO call in lacdyn: it sits after the last
  ! separator (LATTES) and has NO pragma.  It is wrapped in an IF
  ! conditional that does NOT reference block_dim indices.
  ! Crucially, the call passes only derived types and scalars — no
  ! explicit array arguments — so InjectBlockIndexTransformation
  ! will NOT inject bnds%kbl subscripts into the call arguments.
  ! This matches the real IFS pattern where LAVABO receives all data
  ! through derived types (YDGEOMETRY, YDMODEL, YDCPG_BNDS, etc.).
  if (ldo_lavabo) then
    call sub_kernel_lavabo_like(klon, klev, bnds, opts)
  end if

end subroutine mid_kernel_lacdyn_like
""".strip()

FCODE_SUB_KERNEL_LASSIE_LIKE = """
subroutine sub_kernel_lassie_like(klon, klev, bnds, opts, t, q)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = t(jrof, jk) * 2.0
    end do
  end do
end subroutine sub_kernel_lassie_like
""".strip()

FCODE_SUB_KERNEL_LATTES_LIKE = """
subroutine sub_kernel_lattes_like(klon, klev, bnds, opts, q)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: q(klon, klev)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = q(jrof, jk) + 0.1
    end do
  end do
end subroutine sub_kernel_lattes_like
""".strip()

FCODE_SUB_KERNEL_LAVABO_LIKE = """
subroutine sub_kernel_lavabo_like(klon, klev, bnds, opts)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts

  real :: zlocal(klon, klev)
  integer :: jrof, jk

  ! Kernel that operates on local arrays only — no explicit array
  ! arguments at the call site.  This models the real IFS LAVABO
  ! pattern where all data flows through derived types, so the call
  ! site has no array subscripts referencing block_dim indices.
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      zlocal(jrof, jk) = 0.0
    end do
  end do
end subroutine sub_kernel_lavabo_like
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_unpragmaed_call_to_transformed_kernel_gets_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    A call to a transformed sub-kernel WITHOUT ``!$loki small-kernels``
    must still end up inside a block loop.

    This test models the ``lacdyn`` -> ``LAVABO`` pattern in the IFS:

    * ``lacdyn`` has two calls with ``!$loki small-kernels`` pragmas
      (``LASSIE`` and ``LATTES``), which act as separator nodes in
      ``extract_block_sections``.
    * After the last separator (``LATTES``), there is a call to
      ``LAVABO`` **without** any pragma, wrapped in an ``IF``
      conditional.
    * ``LAVABO`` is a Loki-transformed sub-kernel (it appears in the
      pipeline's target list and gets processed) that must be called
      per-block inside the block loop.

    The bug: ``extract_block_sections`` splits the body at separator
    calls, then filters subsections to keep only those referencing
    ``block_dim.indices``.  The subsection after the last separator
    contains the ``IF (ldo_lavabo) THEN / CALL LAVABO(...) / END IF``
    conditional.  Since this conditional doesn't reference ``IBL`` or
    ``BNDS%KBL`` directly, it gets filtered out.  As a result, the
    ``CALL LAVABO_LOKI(...)`` in the BUILD4 output is emitted outside
    any block loop, meaning it processes only one block instead of all.

    Affected IFS files: lacdyn (LAVABO call).
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_LACDYN_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    mid_source = Sourcefile.from_source(
        FCODE_MID_KERNEL_LACDYN_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    lassie_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_LASSIE_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    lattes_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_LATTES_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    lavabo_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_LAVABO_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver_lacdyn_like']
    mid_routine = mid_source['mid_kernel_lacdyn_like']
    lassie_routine = lassie_source['sub_kernel_lassie_like']
    lattes_routine = lattes_source['sub_kernel_lattes_like']
    lavabo_routine = lavabo_source['sub_kernel_lavabo_like']

    # Enrich call graph
    driver_routine.enrich(mid_routine)
    mid_routine.enrich(lassie_routine)
    mid_routine.enrich(lattes_routine)
    mid_routine.enrich(lavabo_routine)

    # Create items
    driver_item = ProcedureItem(name='driver_lacdyn_like_mod#driver_lacdyn_like', source=driver_source)
    mid_item = ProcedureItem(name='#mid_kernel_lacdyn_like', source=mid_source)
    lassie_item = ProcedureItem(name='#sub_kernel_lassie_like', source=lassie_source)
    lattes_item = ProcedureItem(name='#sub_kernel_lattes_like', source=lattes_source)
    lavabo_item = ProcedureItem(name='#sub_kernel_lavabo_like', source=lavabo_source)

    sgraph = SGraph.from_dict({
        driver_item: [mid_item],
        mid_item: [lassie_item, lattes_item, lavabo_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, ['mid_kernel_lacdyn_like']),
        (mid_routine, 'kernel', mid_item, ['sub_kernel_lassie_like', 'sub_kernel_lattes_like', 'sub_kernel_lavabo_like']),
        (lassie_routine, 'kernel', lassie_item, []),
        (lattes_routine, 'kernel', lattes_item, []),
        (lavabo_routine, 'kernel', lavabo_item, []),
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

    driver_code = fgen(driver_routine)
    mid_code = fgen(mid_routine)
    lassie_code = fgen(lassie_routine)
    lattes_code = fgen(lattes_routine)
    lavabo_code = fgen(lavabo_routine)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"Unpragma'd call to transformed kernel test (lacdyn/LAVABO pattern)")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- MID KERNEL (lacdyn-like) ---\n{mid_code}")
    print(f"\n--- SUB KERNEL A (lassie-like) ---\n{lassie_code}")
    print(f"\n--- SUB KERNEL B (lattes-like) ---\n{lattes_code}")
    print(f"\n--- SUB KERNEL C (lavabo-like) ---\n{lavabo_code}")
    print(f"{'='*70}\n")

    mid_lower = mid_code.lower()

    # ===================================================================
    # 1. Mid kernel must have block loop(s) for sections with compute
    # ===================================================================
    mid_loops = FindNodes(Loop).visit(mid_routine.body)
    mid_block_loop_vars = [str(l.variable).lower().replace('local_', '')
                           for l in mid_loops]
    mid_has_block_loop = any(
        v in [idx.lower() for idx in block_dim.indices]
        for v in mid_block_loop_vars
    )
    assert mid_has_block_loop, (
        f"Mid kernel (lacdyn-like) should have at least one block loop.\n"
        f"Loop variables found: {[str(l.variable) for l in mid_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # ===================================================================
    # 2. Find all block loops in the mid kernel
    # ===================================================================
    mid_block_loops = [l for l in mid_loops
                       if str(l.variable).lower().replace('local_', '') in
                       [idx.lower() for idx in block_dim.indices]]

    # ===================================================================
    # 3. The lavabo-like call must be inside SOME block loop
    # ===================================================================
    # Collect all calls inside ALL block loops
    all_calls_in_block_loops = []
    for bl in mid_block_loops:
        calls_in_loop = FindNodes(CallStatement).visit(bl.body)
        all_calls_in_block_loops.extend(calls_in_loop)

    lavabo_calls_in_block = [c for c in all_calls_in_block_loops
                             if 'sub_kernel_lavabo_like' in str(c.name).lower()]

    # Also check: does the mid kernel have the lavabo call at all?
    all_mid_calls = FindNodes(CallStatement).visit(mid_routine.body)
    all_lavabo_calls = [c for c in all_mid_calls
                        if 'sub_kernel_lavabo_like' in str(c.name).lower()]

    print(f"\nAll calls in mid kernel: {[str(c.name) for c in all_mid_calls]}")
    print(f"Lavabo-like calls in mid kernel: {[str(c.name) for c in all_lavabo_calls]}")
    print(f"Block loops in mid kernel: {len(mid_block_loops)}")
    print(f"Calls inside block loops: {[str(c.name) for c in all_calls_in_block_loops]}")
    print(f"Lavabo-like calls inside block loops: {[str(c.name) for c in lavabo_calls_in_block]}")

    assert len(all_lavabo_calls) >= 1, (
        f"Mid kernel should still contain the call to sub_kernel_lavabo_like.\n"
        f"All calls: {[str(c.name) for c in all_mid_calls]}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    assert len(lavabo_calls_in_block) >= 1, (
        f"The call to sub_kernel_lavabo_like must be INSIDE a block loop!\n"
        f"\n"
        f"This is the LAVABO bug: the call to a transformed sub-kernel\n"
        f"without a !$loki small-kernels pragma sits outside any block\n"
        f"loop in the generated code, meaning it processes only one block\n"
        f"instead of all blocks.\n"
        f"\n"
        f"In the real IFS, lacdyn calls LAVABO after the last !$loki\n"
        f"small-kernels separator (LATTES). Since LAVABO has no pragma,\n"
        f"extract_block_sections doesn't include the subsection containing\n"
        f"it in the block sections (the IF conditional doesn't reference\n"
        f"block_dim indices like IBL). The call ends up outside the block\n"
        f"loop.\n"
        f"\n"
        f"Block loops found: {len(mid_block_loops)}\n"
        f"All calls inside block loops: {[str(c.name) for c in all_calls_in_block_loops]}\n"
        f"Lavabo-like calls outside block loops: {[str(c.name) for c in all_lavabo_calls]}\n"
        f"\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # ===================================================================
    # 4. The pragmaed calls (lassie, lattes) should be OUTSIDE block
    #    loops since they have !$loki small-kernels (they ARE separators)
    # ===================================================================
    lassie_calls_in_block = [c for c in all_calls_in_block_loops
                             if 'sub_kernel_lassie_like' in str(c.name).lower()]
    lattes_calls_in_block = [c for c in all_calls_in_block_loops
                             if 'sub_kernel_lattes_like' in str(c.name).lower()]

    # The !$loki small-kernels calls should NOT be inside the block loop
    # because they ARE the separators — the block loop wraps the compute
    # sections between them, not the calls themselves.
    assert len(lassie_calls_in_block) == 0, (
        f"The lassie-like call (with !$loki small-kernels) should NOT be\n"
        f"inside a block loop — it is a separator, and the block loop wraps\n"
        f"the compute sections between separators.\n"
        f"Found {len(lassie_calls_in_block)} lassie calls inside block loops.\n"
        f"Mid kernel code:\n{mid_code}"
    )
    assert len(lattes_calls_in_block) == 0, (
        f"The lattes-like call (with !$loki small-kernels) should NOT be\n"
        f"inside a block loop — it is a separator, and the block loop wraps\n"
        f"the compute sections between separators.\n"
        f"Found {len(lattes_calls_in_block)} lattes calls inside block loops.\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # ===================================================================
    # 5. Verify lavabo sub-kernel is still present and processed
    #    (It does NOT need its own block loop — it is a leaf kernel
    #    called per-block from the mid kernel's block loop.  In the
    #    real IFS, lavabo has !$acc routine vector, not its own gang loop.)
    # ===================================================================
    print(f"\n--- Lavabo sub-kernel code ---\n{lavabo_code}")
    # Sanity check: lavabo routine still exists and has some content
    assert lavabo_routine.body is not None, (
        f"Lavabo sub-kernel body should not be None after pipeline.\n"
        f"Lavabo kernel code:\n{lavabo_code}"
    )


# ---------------------------------------------------------------------------
# Test: !$acc enter/exit data directives must NOT be swept into block
#       loop when extract_block_sections keeps a subsection via the
#       enriched-call filter (LAVABO fix).
#
# Models the lacdyn pattern with !$acc enter/exit data around the
# body. The mid kernel has:
#   !$acc enter data create(ztmp)
#   <section 1 compute + LASSIE separator>
#   <section 2 compute + LATTES separator>
#   <section 3: IF (ldo_lavabo) CALL LAVABO — no pragma, derived-type args>
#   !$acc exit data delete(ztmp)
#
# After the LAVABO fix, extract_block_sections keeps the post-LATTES
# subsection because it contains an enriched call (LAVABO).  But
# get_trimmed_sections receives an empty block_nodes list (LAVABO has
# no block_dim.indices references), falls through to the else branch,
# and keeps the ENTIRE section including the trailing !$acc exit data.
# ReblockSectionTransformer then wraps it in a block loop — which is
# illegal because !$acc exit data is a host-side unstructured data
# directive.
#
# Expected:
#   - !$acc enter data is OUTSIDE all block loops (before first separator)
#   - !$acc exit data is OUTSIDE all block loops (after last section)
#   - LAVABO call IS inside a block loop
#
# BUILD4 bug: !$acc exit data at line 431 of lacdyn.scc_stack.F90,
#             INSIDE the third block loop (DO local_JKGLO, lines 377-434)
# Working:    !$acc exit data at line 428, OUTSIDE any block loop
# ---------------------------------------------------------------------------

FCODE_DRIVER_ACC_SWEEP = """
module driver_acc_sweep_mod
  implicit none
contains
  subroutine driver_acc_sweep(ngpblks, bnds, opts, t, q, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "mid_kernel_acc_sweep.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: t(:,:,:)
    real, intent(inout) :: q(:,:,:)
    real, intent(inout) :: pfield(:,:,:)

    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call mid_kernel_acc_sweep(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl), pfield(:,:,ibl))
    end do
  end subroutine driver_acc_sweep
end module driver_acc_sweep_mod
""".strip()

FCODE_MID_KERNEL_ACC_SWEEP = """
subroutine mid_kernel_acc_sweep(klon, klev, bnds, opts, t, q, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "sub_kernel_lassie_like.intfb.h"
  #include "sub_kernel_lattes_like.intfb.h"
  #include "sub_kernel_lavabo_like.intfb.h"
  integer, intent(in)  :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)
  real, intent(inout) :: pfield(klon, klev)

  real :: ztmp(klon, klev)
  integer :: jrof, jk
  logical :: ldo_lavabo

  ldo_lavabo = .true.

  ! Host-side unstructured data directive — must stay outside block loops
  !$acc enter data create(ztmp)

  ! ---- Section 1: horizontal compute + pragmaed call (LASSIE) ----
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do

  !$loki small-kernels
  call sub_kernel_lassie_like(klon, klev, bnds, opts, t, q)

  ! ---- Section 2: horizontal compute + pragmaed call (LATTES) ----
  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      q(jrof, jk) = q(jrof, jk) * 0.5
    end do
  end do

  !$loki small-kernels
  call sub_kernel_lattes_like(klon, klev, bnds, opts, q)

  ! ---- Section 3: conditional call WITHOUT pragma (LAVABO pattern) ----
  ! Derived-type-only args — no explicit arrays — so no block_dim
  ! subscripts are injected by InjectBlockIndexTransformation.
  if (ldo_lavabo) then
    call sub_kernel_lavabo_like(klon, klev, bnds, opts)
  end if

  ! Host-side unstructured data directive — must stay outside block loops
  !$acc exit data delete(ztmp)

end subroutine mid_kernel_acc_sweep
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_acc_exit_data_not_swept_into_block_section(frontend, horizontal, block_dim, tmp_path):
    """
    ``!$acc enter/exit data`` directives must stay outside block loops
    even when ``extract_block_sections`` keeps a post-separator subsection
    via the enriched-call filter (the LAVABO fix).

    This test models the ``lacdyn`` pattern with ``!$acc enter data create``
    before the compute sections and ``!$acc exit data delete`` after the
    last section (which contains the unpragma'd LAVABO call).

    The bug: the LAVABO fix in ``extract_block_sections`` keeps the
    post-LATTES subsection because it contains an enriched call.  But
    ``get_trimmed_sections`` receives empty ``block_nodes`` (LAVABO has
    no ``block_dim.indices`` refs — it takes only derived types), so it
    falls through to ``else: trimmed_sections += (sec,)`` and keeps the
    ENTIRE section, including the trailing ``!$acc exit data delete``.
    ``ReblockSectionTransformer`` then wraps it in a block loop, which
    is illegal OpenACC (host-side unstructured data directive inside a
    device-side ``!$acc parallel loop gang`` region).

    BUILD4 bug: ``!$acc exit data delete(...)`` at line 431 of
    ``lacdyn.scc_stack.F90``, inside block loop (lines 377-434).
    Working: ``!$acc exit data delete(...)`` at line 428, outside any loop.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_ACC_SWEEP, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    mid_source = Sourcefile.from_source(
        FCODE_MID_KERNEL_ACC_SWEEP, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    lassie_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_LASSIE_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    lattes_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_LATTES_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    lavabo_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_LAVABO_LIKE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver_acc_sweep']
    mid_routine = mid_source['mid_kernel_acc_sweep']
    lassie_routine = lassie_source['sub_kernel_lassie_like']
    lattes_routine = lattes_source['sub_kernel_lattes_like']
    lavabo_routine = lavabo_source['sub_kernel_lavabo_like']

    # Enrich call graph
    driver_routine.enrich(mid_routine)
    mid_routine.enrich(lassie_routine)
    mid_routine.enrich(lattes_routine)
    mid_routine.enrich(lavabo_routine)

    # Create items
    driver_item = ProcedureItem(name='driver_acc_sweep_mod#driver_acc_sweep', source=driver_source)
    mid_item = ProcedureItem(name='#mid_kernel_acc_sweep', source=mid_source)
    lassie_item = ProcedureItem(name='#sub_kernel_lassie_like', source=lassie_source)
    lattes_item = ProcedureItem(name='#sub_kernel_lattes_like', source=lattes_source)
    lavabo_item = ProcedureItem(name='#sub_kernel_lavabo_like', source=lavabo_source)

    sgraph = SGraph.from_dict({
        driver_item: [mid_item],
        mid_item: [lassie_item, lattes_item, lavabo_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, ['mid_kernel_acc_sweep']),
        (mid_routine, 'kernel', mid_item, ['sub_kernel_lassie_like', 'sub_kernel_lattes_like', 'sub_kernel_lavabo_like']),
        (lassie_routine, 'kernel', lassie_item, []),
        (lattes_routine, 'kernel', lattes_item, []),
        (lavabo_routine, 'kernel', lavabo_item, []),
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

    driver_code = fgen(driver_routine)
    mid_code = fgen(mid_routine)
    lassie_code = fgen(lassie_routine)
    lattes_code = fgen(lattes_routine)
    lavabo_code = fgen(lavabo_routine)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"ACC enter/exit data NOT swept into block section test")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- MID KERNEL (acc sweep) ---\n{mid_code}")
    print(f"\n--- SUB KERNEL A (lassie-like) ---\n{lassie_code}")
    print(f"\n--- SUB KERNEL B (lattes-like) ---\n{lattes_code}")
    print(f"\n--- SUB KERNEL C (lavabo-like) ---\n{lavabo_code}")
    print(f"{'='*70}\n")

    # ===================================================================
    # 1. Mid kernel must have block loop(s)
    # ===================================================================
    mid_loops = FindNodes(Loop).visit(mid_routine.body)
    mid_block_loops = [l for l in mid_loops
                       if str(l.variable).lower().replace('local_', '') in
                       [idx.lower() for idx in block_dim.indices]]
    assert len(mid_block_loops) >= 1, (
        f"Mid kernel should have at least one block loop.\n"
        f"Loop variables found: {[str(l.variable) for l in mid_loops]}\n"
        f"block_dim.indices: {block_dim.indices}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # ===================================================================
    # 2. Collect all pragmas inside ALL block loops
    # ===================================================================
    all_pragmas_in_block_loops = []
    for bl in mid_block_loops:
        pragmas_in_loop = FindNodes(Pragma).visit(bl.body)
        all_pragmas_in_block_loops.extend(pragmas_in_loop)

    # ===================================================================
    # 3. !$acc enter data create(...) must NOT be inside any block loop
    # ===================================================================
    enter_data_inside = [p for p in all_pragmas_in_block_loops
                         if p.keyword.lower() == 'acc'
                         and 'enter' in p.content.lower()
                         and 'data' in p.content.lower()
                         and 'create' in p.content.lower()]

    assert len(enter_data_inside) == 0, (
        f"!$acc enter data create(...) must NOT be inside a block loop!\n"
        f"These are host-side unstructured data directives that cannot\n"
        f"appear inside a device-side !$acc parallel loop gang region.\n"
        f"Found {len(enter_data_inside)} enter-data pragma(s) inside block loop(s):\n"
        f"  {[fgen(p) for p in enter_data_inside]}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # ===================================================================
    # 4. !$acc exit data delete(...) must NOT be inside any block loop
    #    THIS IS THE ASSERTION EXPECTED TO FAIL (the bug)
    # ===================================================================
    exit_data_inside = [p for p in all_pragmas_in_block_loops
                        if p.keyword.lower() == 'acc'
                        and 'exit' in p.content.lower()
                        and 'data' in p.content.lower()
                        and 'delete' in p.content.lower()]

    assert len(exit_data_inside) == 0, (
        f"!$acc exit data delete(...) must NOT be inside a block loop!\n"
        f"\n"
        f"This is the acc-sweep bug: the LAVABO fix keeps the post-separator\n"
        f"subsection (containing the enriched LAVABO call), but\n"
        f"get_trimmed_sections gets empty block_nodes and keeps the ENTIRE\n"
        f"section — including the trailing !$acc exit data delete.\n"
        f"ReblockSectionTransformer wraps it in a block loop, placing the\n"
        f"host-side !$acc exit data directive inside a device-side\n"
        f"!$acc parallel loop gang region.\n"
        f"\n"
        f"In the real IFS, lacdyn.scc_stack.F90 BUILD4 output has\n"
        f"!$acc exit data delete(...) at line 431, INSIDE the third block\n"
        f"loop (DO local_JKGLO, lines 377-434). The working output has it\n"
        f"at line 428, OUTSIDE any block loop.\n"
        f"\n"
        f"Found {len(exit_data_inside)} exit-data pragma(s) inside block loop(s):\n"
        f"  {[fgen(p) for p in exit_data_inside]}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # ===================================================================
    # 5. LAVABO call must still be INSIDE a block loop
    #    (this should pass — the LAVABO fix handles this correctly)
    # ===================================================================
    all_calls_in_block_loops = []
    for bl in mid_block_loops:
        calls_in_loop = FindNodes(CallStatement).visit(bl.body)
        all_calls_in_block_loops.extend(calls_in_loop)

    lavabo_calls_in_block = [c for c in all_calls_in_block_loops
                             if 'sub_kernel_lavabo_like' in str(c.name).lower()]
    all_mid_calls = FindNodes(CallStatement).visit(mid_routine.body)
    all_lavabo_calls = [c for c in all_mid_calls
                        if 'sub_kernel_lavabo_like' in str(c.name).lower()]

    print(f"\nAll calls in mid kernel: {[str(c.name) for c in all_mid_calls]}")
    print(f"Lavabo-like calls inside block loops: {[str(c.name) for c in lavabo_calls_in_block]}")
    print(f"Pragmas inside block loops: {[fgen(p) for p in all_pragmas_in_block_loops]}")

    assert len(all_lavabo_calls) >= 1, (
        f"Mid kernel should still contain the call to sub_kernel_lavabo_like.\n"
        f"All calls: {[str(c.name) for c in all_mid_calls]}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    assert len(lavabo_calls_in_block) >= 1, (
        f"The call to sub_kernel_lavabo_like must be INSIDE a block loop.\n"
        f"(This was fixed by the LAVABO fix — if this fails, the LAVABO\n"
        f"fix may have regressed.)\n"
        f"Block loops found: {len(mid_block_loops)}\n"
        f"Mid kernel code:\n{mid_code}"
    )

    # ===================================================================
    # 6. Pragmaed calls (LASSIE, LATTES) should be OUTSIDE block loops
    # ===================================================================
    lassie_calls_in_block = [c for c in all_calls_in_block_loops
                             if 'sub_kernel_lassie_like' in str(c.name).lower()]
    lattes_calls_in_block = [c for c in all_calls_in_block_loops
                             if 'sub_kernel_lattes_like' in str(c.name).lower()]

    assert len(lassie_calls_in_block) == 0, (
        f"The lassie-like call (with !$loki small-kernels) should NOT be\n"
        f"inside a block loop.\n"
        f"Mid kernel code:\n{mid_code}"
    )
    assert len(lattes_calls_in_block) == 0, (
        f"The lattes-like call (with !$loki small-kernels) should NOT be\n"
        f"inside a block loop.\n"
        f"Mid kernel code:\n{mid_code}"
    )


# ---------------------------------------------------------------------------
# Test: Pool allocator ISTSZ/ZSTACK generation with non-standard block
#       loop variable.
#
# Models the lapineb_drv -> lapineb pattern: the driver uses
# ``DO JSTGLO = 1, NGPTOT_CAP, NPROMA`` as its block loop.
# ``LowerBlockIndexTransformation`` propagates this loop (including
# the ``JSTGLO`` variable) to the kernel.  Later, the pool allocator's
# ``_filter_block_dim_loops`` checks if the loop variable matches
# ``block_dim.indices``.  The test fixture has
# ``block_dim.index = ('ibl', 'bnds%kbl')``.  ``JSTGLO`` is NOT in
# this set, so the loop is rejected, ``driver_loops`` is empty, and
# the ``if driver_loops:`` guard prevents ISTSZ/ZSTACK creation.
#
# Parametrized:
#   loop_var = 'ibl'     → should PASS (ibl IS in block_dim.indices)
#   loop_var = 'jstglo'  → should FAIL (jstglo NOT in block_dim.indices)
#
# BUILD4: lapineb.scc_stack.F90 has no ISTSZ, ZSTACK, ALLOCATE,
#   or YLSTACK_L = LOC(...) inside block loop.  Loop var is JSTGLO.
# Working: lapineb.scc_stack.F90 has ISTSZ (line 263), ZSTACK (line 264),
#   170-line MAX expression, per-block LOC assignment.
# ---------------------------------------------------------------------------

def _fcode_driver_pool_alloc(loop_var):
    """Generate driver Fortran with a parameterized block loop variable."""
    return f"""
module driver_pool_alloc_mod
  implicit none
contains
  subroutine driver_pool_alloc(ngpblks, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_pool_alloc.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(:,:,:)

    integer :: {loop_var}

    do {loop_var} = 1, ngpblks
      bnds%kbl = {loop_var}
      bnds%kstglo = 1 + ({loop_var} - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call kernel_pool_alloc(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,{loop_var}))
    end do
  end subroutine driver_pool_alloc
end module driver_pool_alloc_mod
""".strip()


FCODE_KERNEL_POOL_ALLOC = """
subroutine kernel_pool_alloc(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "sub_kernel_pool_alloc.intfb.h"
  integer, intent(in)  :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: pfield(klon, klev)

  ! Local temporaries that trigger pool allocation
  real :: ztmp1(klon, klev)
  real :: ztmp2(klon)

  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp1(jrof, jk) = pfield(jrof, jk) * 2.0
    end do
  end do

  do jrof = bnds%kidia, bnds%kfdia
    ztmp2(jrof) = 0.0
  end do

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      ztmp2(jrof) = ztmp2(jrof) + ztmp1(jrof, jk)
    end do
  end do

  do jrof = bnds%kidia, bnds%kfdia
    pfield(jrof, 1) = ztmp2(jrof)
  end do

  ! Call sub-kernel — NO !$loki small-kernels pragma.
  ! This models the real IFS pattern (e.g. lapineb calling LARCINB,
  ! VERDISINT, etc.) where compute calls are NOT separators.
  ! After the pipeline, this call ends up INSIDE the block loop,
  ! which makes find_driver_loops recognise the block loop as a
  ! driver loop and triggers pool allocator infrastructure.
  call sub_kernel_pool_alloc(klon, klev, bnds, opts, pfield)

end subroutine kernel_pool_alloc
""".strip()

FCODE_SUB_KERNEL_POOL_ALLOC = """
subroutine sub_kernel_pool_alloc(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in)  :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: pfield(klon, klev)

  ! Local temporary that models larcinb's ZGMVF: a 2D scratch array.
  ! Since this sub-kernel has NO !$loki small-kernels calls,
  ! LowerBlockIndexTransformation exits early and does NOT promote
  ! zlocal to 3D.  The pool allocator's block_dim.sizes filter
  ! does NOT remove it (last dim is klev, not ngpblks).  So zlocal
  ! gets a Cray pointer and contributes non-zero stack_size, which
  ! propagates up to the parent kernel's ISTSZ calculation.
  real :: zlocal(klon, klev)
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      zlocal(jrof, jk) = pfield(jrof, jk) * 0.5
    end do
  end do

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      pfield(jrof, jk) = pfield(jrof, jk) + zlocal(jrof, jk)
    end do
  end do
end subroutine sub_kernel_pool_alloc
""".strip()


@pytest.mark.parametrize('loop_var', ['ibl', 'jstglo'])
@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_pool_allocator_with_nonstandard_block_loop_variable(frontend, horizontal, block_dim, tmp_path, loop_var):
    """
    Pool allocator ISTSZ/ZSTACK must be generated even when the driver's
    block loop variable is not in ``block_dim.indices``.

    This test models the ``lapineb_drv`` -> ``lapineb`` -> ``larcinb``
    pattern:

    * The driver uses ``DO {loop_var} = 1, NGPBLKS`` as its block loop.
    * ``LowerBlockIndexTransformation`` propagates this loop to the kernel.
    * The kernel calls a sub-kernel (modelling ``larcinb``) WITHOUT
      ``!$loki small-kernels``, so the call ends up INSIDE the block loop.
    * The sub-kernel has a local 2D temporary (modelling ``ZGMVF``).
      Since the sub-kernel has no ``!$loki small-kernels`` calls,
      ``LowerBlockIndexTransformation`` exits early and does NOT promote
      the temporary to 3D.  The pool allocator converts it to a Cray
      pointer with non-zero ``stack_size``.
    * The kernel's ``_determine_stack_size`` picks up the sub-kernel's
      non-zero ``stack_size`` and computes a non-zero ``ISTSZ``.
    * ``_filter_block_dim_loops`` then checks if the kernel's block loop
      variable matches ``block_dim.indices``.

    With ``loop_var='ibl'``, the loop variable IS in ``block_dim.indices``
    and the pool allocator works correctly: ISTSZ includes the sub-kernel's
    contribution, ZSTACK is allocated, and ``YLSTACK_L = LOC(ZSTACK(1, ...))``
    is inserted per-block inside the block loop.

    With ``loop_var='jstglo'``, the loop variable is NOT in
    ``block_dim.indices = ('ibl', 'bnds%%kbl')``, so
    ``_filter_block_dim_loops`` rejects it, ``driver_loops`` is empty,
    and the ``if driver_loops:`` guard prevents ISTSZ/ZSTACK creation.
    But ``YDSTACK_L``/``YLSTACK_L`` arguments ARE still injected and
    passed to callees with a stale, non-per-block value.

    BUILD4 bug: ``lapineb.scc_stack.F90`` has no ``ISTSZ``, no ``ZSTACK``,
    no ``ALLOCATE``, no ``YLSTACK_L = LOC(ZSTACK(1, local_IBL))``.
    Working: all present — 170+ lines of ISTSZ MAX expression, ZSTACK
    allocation, per-block LOC assignment.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_fcode = _fcode_driver_pool_alloc(loop_var)

    driver_source = Sourcefile.from_source(
        driver_fcode, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_POOL_ALLOC, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    sub_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_POOL_ALLOC, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver_pool_alloc']
    kernel_routine = kernel_source['kernel_pool_alloc']
    sub_routine = sub_source['sub_kernel_pool_alloc']

    # Enrich call graph
    driver_routine.enrich(kernel_routine)
    kernel_routine.enrich(sub_routine)

    # Create items
    driver_item = ProcedureItem(name='driver_pool_alloc_mod#driver_pool_alloc', source=driver_source)
    kernel_item = ProcedureItem(name='#kernel_pool_alloc', source=kernel_source)
    sub_item = ProcedureItem(name='#sub_kernel_pool_alloc', source=sub_source)

    sgraph = SGraph.from_dict({
        driver_item: [kernel_item],
        kernel_item: [sub_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, ['kernel_pool_alloc']),
        (kernel_routine, 'kernel', kernel_item, ['sub_kernel_pool_alloc']),
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

    driver_code = fgen(driver_routine)
    kernel_code = fgen(kernel_routine)
    sub_code = fgen(sub_routine)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"Pool allocator with block loop variable '{loop_var}' test")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"\n--- SUB KERNEL ---\n{sub_code}")
    print(f"{'='*70}\n")

    kernel_lower = kernel_code.lower()
    sub_lower = sub_code.lower()

    # ===================================================================
    # A. Sub-kernel verification (should pass for BOTH ibl and jstglo)
    # ===================================================================
    # The sub-kernel's local temporary zlocal(klon, klev) must be pool-
    # allocated via a Cray pointer.  This happens regardless of the
    # parent's block loop variable because apply_pool_allocator_to_temporaries
    # runs on the sub-kernel independently.
    # ===================================================================

    # A1. Sub-kernel must have a Cray pointer for zlocal
    sub_intrinsics = FindNodes(Intrinsic).visit(sub_routine.spec)
    cray_ptr_decls = [i for i in sub_intrinsics
                      if 'pointer' in i.text.lower()
                      and 'zlocal' in i.text.lower()]

    print(f"\nSub-kernel intrinsics (Cray pointers): {[i.text for i in sub_intrinsics]}")
    print(f"Cray pointer declarations for zlocal: {[i.text for i in cray_ptr_decls]}")

    assert len(cray_ptr_decls) >= 1, (
        f"Sub-kernel must have a Cray pointer declaration for zlocal!\n"
        f"POINTER(IP_zlocal, zlocal) should be generated by the pool\n"
        f"allocator because zlocal(klon, klev) stays 2D — its last dim\n"
        f"is klev (not ngpblks), so the block_dim.sizes filter does NOT\n"
        f"remove it.\n"
        f"\n"
        f"This models larcinb's ZGMVF which is pool-allocated via Cray\n"
        f"pointer in both BUILD3 and BUILD4 outputs.\n"
        f"\n"
        f"Sub-kernel intrinsics: {[i.text for i in sub_intrinsics]}\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # A2. Sub-kernel must have IP_zlocal = YLSTACK_L (stack pointer assignment)
    sub_assigns = FindNodes(Assignment).visit(sub_routine.body)
    ip_assigns = [a for a in sub_assigns
                  if 'ip_' in str(a.lhs).lower()
                  and 'zlocal' in str(a.lhs).lower()
                  and 'ylstack_l' in str(a.rhs).lower()]

    print(f"IP_zlocal assignments: {[fgen(a) for a in ip_assigns]}")

    assert len(ip_assigns) >= 1, (
        f"Sub-kernel must have IP_zlocal = YLSTACK_L assignment!\n"
        f"This assigns the Cray pointer to the current stack position,\n"
        f"placing zlocal on the caller-provided stack.\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # A3. Sub-kernel must advance the stack pointer after zlocal
    stack_advance_assigns = [a for a in sub_assigns
                             if 'ylstack_l' in str(a.lhs).lower()
                             and 'ylstack_l' in str(a.rhs).lower()
                             and 'ishft' in str(a.rhs).lower()]

    print(f"Stack advance assignments: {[fgen(a) for a in stack_advance_assigns]}")

    assert len(stack_advance_assigns) >= 1, (
        f"Sub-kernel must advance YLSTACK_L after zlocal allocation!\n"
        f"YLSTACK_L = YLSTACK_L + ISHFT(...) bumps the stack pointer\n"
        f"past the space used by zlocal.\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # ===================================================================
    # B. Kernel verification (ibl passes, jstglo FAILS)
    # ===================================================================

    # B1. Kernel must have a block loop
    kernel_loops = FindNodes(Loop).visit(kernel_routine.body)
    block_loop_vars = [str(l.variable).lower().replace('local_', '')
                       for l in kernel_loops]
    has_block_loop = any(
        v == loop_var.lower() for v in block_loop_vars
    )
    assert has_block_loop, (
        f"Kernel should have a block loop with variable '{loop_var}'.\n"
        f"Loop variables found: {[str(l.variable) for l in kernel_loops]}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # B2. Kernel must have ISTSZ declared
    #     THIS IS THE FIRST ASSERTION EXPECTED TO FAIL FOR loop_var='jstglo'
    kernel_var_names = [str(v).lower() for v in kernel_routine.variables]
    has_istsz = 'istsz' in kernel_var_names

    print(f"\nKernel variable names: {kernel_var_names}")
    print(f"Has ISTSZ: {has_istsz}")

    assert has_istsz, (
        f"Kernel must have ISTSZ declared for pool allocator!\n"
        f"\n"
        f"This is the pool allocator bug: the driver uses block loop\n"
        f"variable '{loop_var}', which is NOT in block_dim.indices\n"
        f"= {block_dim.indices}.  _filter_block_dim_loops rejects the\n"
        f"loop, driver_loops is empty, and the 'if driver_loops:' guard\n"
        f"prevents ISTSZ/ZSTACK creation.\n"
        f"\n"
        f"The sub-kernel has a pool-allocated temporary (zlocal via Cray\n"
        f"pointer), so its stack_size is non-zero.  The kernel's\n"
        f"_determine_stack_size should propagate this up to create a\n"
        f"non-zero ISTSZ.  But _filter_block_dim_loops rejects the loop\n"
        f"before create_pool_allocator_drv_loop is ever called.\n"
        f"\n"
        f"Kernel variables: {kernel_var_names}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # B3. ISTSZ must include the sub-kernel's stack contribution
    #     The sub-kernel has zlocal(klon, klev), so ISTSZ must contain
    #     a term proportional to klon*klev (the size of zlocal).
    kernel_assigns = FindNodes(Assignment).visit(kernel_routine.body)
    istsz_assigns = [a for a in kernel_assigns
                     if str(a.lhs).lower() == 'istsz']

    print(f"\nISTSZ assignments: {[fgen(a) for a in istsz_assigns]}")

    assert len(istsz_assigns) >= 1, (
        f"Kernel must have an ISTSZ assignment!\n"
        f"Kernel code:\n{kernel_code}"
    )

    # Check that the ISTSZ RHS contains klon*klev (from the sub-kernel's zlocal)
    istsz_rhs = str(istsz_assigns[0].rhs).lower()
    has_klon_klev = 'klon' in istsz_rhs and 'klev' in istsz_rhs

    print(f"ISTSZ RHS: {istsz_rhs}")
    print(f"ISTSZ RHS contains klon*klev contribution: {has_klon_klev}")

    assert has_klon_klev, (
        f"ISTSZ must include a term proportional to klon*klev!\n"
        f"\n"
        f"The sub-kernel has zlocal(klon, klev) which is pool-allocated.\n"
        f"_determine_stack_size should propagate this as\n"
        f"ISHFT(7 + C_SIZEOF(REAL(1))*klon*klev, -3) or similar.\n"
        f"The kernel's ISTSZ should include this term.\n"
        f"\n"
        f"ISTSZ assignment: {fgen(istsz_assigns[0])}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # B4. Kernel must have ZSTACK allocated
    kernel_allocs = FindNodes(Allocation).visit(kernel_routine.body)
    zstack_allocs = [a for a in kernel_allocs
                     if any('zstack' in str(v).lower() for v in a.variables)]
    assert len(zstack_allocs) >= 1, (
        f"Kernel must have ZSTACK allocated for pool allocator!\n"
        f"Allocations found: {[fgen(a) for a in kernel_allocs]}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # B5. Kernel must have YLSTACK_L = LOC(ZSTACK(1, ...)) inside block loop
    kernel_block_loops = [l for l in kernel_loops
                         if str(l.variable).lower().replace('local_', '') == loop_var.lower()]
    assert len(kernel_block_loops) >= 1, (
        f"Expected at least one block loop with variable '{loop_var}'.\n"
        f"Kernel code:\n{kernel_code}"
    )

    all_loc_assigns_in_blocks = []
    for bl in kernel_block_loops:
        block_loop_code = fgen(bl)
        assignments_in_block = FindNodes(Assignment).visit(bl.body)
        loc_assigns = [a for a in assignments_in_block
                       if 'ylstack_l' in str(a.lhs).lower()
                       and 'loc' in str(a.rhs).lower()]
        all_loc_assigns_in_blocks.extend(loc_assigns)
        print(f"\nBlock loop code:\n{block_loop_code}")
        print(f"LOC assigns in block loop: {[fgen(a) for a in loc_assigns]}")

    assert len(all_loc_assigns_in_blocks) >= 1, (
        f"Kernel must have YLSTACK_L = LOC(ZSTACK(1, ...)) inside block loop!\n"
        f"This per-block LOC assignment ensures each block gets its own\n"
        f"stack pointer into the pre-allocated pool.  Without it, the\n"
        f"sub-kernel's Cray pointer (IP_zlocal = YLSTACK_L) would point\n"
        f"to a stale, non-per-block address.\n"
        f"Kernel code:\n{kernel_code}"
    )


# ===================================================================
# Test: promoted local arrays must get !$acc enter data create
# ===================================================================
#
# Models the lapineb pattern: the driver (lapineb_drv) calls the
# kernel (lapineb) with !$loki small-kernels.  The kernel has local
# 2D arrays (e.g. ZSPT1) and calls sub-kernels WITHOUT !$loki
# small-kernels (the pragmas are commented out in the real source).
#
# LowerBlockIndexTransformation flow:
# 1. process_driver (on the driver) sees the !$loki small-kernels call
#    and promotes the kernel's local arrays to 3D (appending
#    block_dim size).  However, process_driver does NOT add
#    !$loki unstructured-data create/delete for those locals.
# 2. process_kernel (on the kernel) exits early because the kernel
#    has no !$loki small-kernels calls — so the code at lines 748-755
#    that adds !$loki unstructured-data create is never reached.
#
# Result: local arrays are promoted to 3D (host-allocated) but have
# no !$acc enter data create to allocate them on the GPU.  Any
# GPU kernel that touches them hits an OpenACC runtime error.
#
# Contrast with cpg_gp_hyd: it HAS !$loki small-kernels calls, so
# process_kernel runs fully and adds the unstructured-data pragmas.
#
# BUILD output: lapineb.scc_stack.F90 has promoted locals (ZSPT1,
# ZUT1, ZVT1, …) with NGPBLKS dimension but no !$acc enter data
# create for them.  cpg_gp_hyd.scc_stack.F90 correctly has
# !$acc enter data create(Z_DPHYCTY_T0, Z_DVER_T0, …).
# ===================================================================

FCODE_DRIVER_PROMOTED_LOCALS = """
module driver_promoted_mod
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
contains
  subroutine driver_promoted(ngpblks, bnds, opts, pfield)
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(:,:,:)
    integer :: ibl

#include "kernel_promoted.intfb.h"

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

!$loki small-kernels
      call kernel_promoted(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
  end subroutine driver_promoted
end module driver_promoted_mod
""".strip()

# Kernel with local arrays but NO !$loki small-kernels on the call
# to the sub-kernel.  This models lapineb which has commented-out
# !$loki small-kernels pragmas.
FCODE_KERNEL_PROMOTED_LOCALS = """
subroutine kernel_promoted(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: pfield(klon, klev)

  ! Local temporaries — these are 2D in the source but get promoted
  ! to 3D by LowerBlockIndexTransformation.process_driver.  They
  ! MUST get !$acc enter data create on the GPU.
  real :: ztmp(klon, klev)
  real :: zout(klon)

  integer :: jrof, jk

#include "sub_kernel_promoted.intfb.h"

  ! Compute section using local arrays
  do jrof = bnds%kidia, bnds%kfdia
    do jk = 1, klev
      ztmp(jrof, jk) = pfield(jrof, jk) * 2.0
    end do
  end do

  do jrof = bnds%kidia, bnds%kfdia
    zout(jrof) = 0.0
    do jk = 1, klev
      zout(jrof) = zout(jrof) + ztmp(jrof, jk)
    end do
    pfield(jrof, 1) = zout(jrof)
  end do

  ! Call sub-kernel WITHOUT !$loki small-kernels — models lapineb
  ! calling VERDISINT, LARCINB etc. with commented-out pragmas.
  ! This means process_kernel exits early for this kernel.
  call sub_kernel_promoted(klon, klev, bnds, opts, pfield)

end subroutine kernel_promoted
""".strip()

# Simple sub-kernel — just does some work on pfield.
FCODE_SUB_KERNEL_PROMOTED_LOCALS = """
subroutine sub_kernel_promoted(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: pfield(klon, klev)

  integer :: jrof, jk

  do jrof = bnds%kidia, bnds%kfdia
    do jk = 1, klev
      pfield(jrof, jk) = pfield(jrof, jk) * 1.5
    end do
  end do
end subroutine sub_kernel_promoted
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_promoted_locals_get_acc_enter_data_create(frontend, horizontal, block_dim, tmp_path):
    """
    Promoted local arrays in a kernel must get ``!$acc enter data create``
    / ``!$acc exit data delete`` directives (or their ``!$loki`` equivalents)
    so that they are allocated on the GPU.

    This test models the ``lapineb`` pattern:

    * The driver calls the kernel with ``!$loki small-kernels``.
    * ``LowerBlockIndexTransformation.process_driver`` promotes the
      kernel's local arrays from 2D to 3D (appending ``block_dim`` size).
    * The kernel has NO ``!$loki small-kernels`` calls itself, so
      ``process_kernel`` exits early and never reaches the code that
      adds ``!$loki unstructured-data create`` (block_index_transformations.py
      lines 748-755).

    Result: the local arrays (e.g. ``ztmp``, ``zout``) are promoted to
    3D (``ztmp(klon, klev, ngpblks)``) but have no ``!$acc enter data
    create`` to allocate them on the GPU.

    Contrast with ``cpg_gp_hyd`` which HAS ``!$loki small-kernels``
    calls: ``process_kernel`` runs fully and adds the data directives.

    BUILD output: ``lapineb.scc_stack.F90`` has promoted locals (ZSPT1,
    ZUT1, ZVT1, ...) with NGPBLKS dimension but no
    ``!$acc enter data create``.  ``cpg_gp_hyd.scc_stack.F90`` correctly
    has ``!$acc enter data create(Z_DPHYCTY_T0, ...)``.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_PROMOTED_LOCALS, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_PROMOTED_LOCALS, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    sub_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_PROMOTED_LOCALS, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver_promoted']
    kernel_routine = kernel_source['kernel_promoted']
    sub_routine = sub_source['sub_kernel_promoted']

    # Enrich call graph
    driver_routine.enrich(kernel_routine)
    kernel_routine.enrich(sub_routine)

    # Create items
    driver_item = ProcedureItem(name='driver_promoted_mod#driver_promoted', source=driver_source)
    kernel_item = ProcedureItem(name='#kernel_promoted', source=kernel_source)
    sub_item = ProcedureItem(name='#sub_kernel_promoted', source=sub_source)

    sgraph = SGraph.from_dict({
        driver_item: [kernel_item],
        kernel_item: [sub_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, ['kernel_promoted']),
        (kernel_routine, 'kernel', kernel_item, ['sub_kernel_promoted']),
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

    kernel_code = fgen(kernel_routine)
    driver_code = fgen(driver_routine)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"Promoted locals !$acc enter data create test")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"{'='*70}\n")

    kernel_lower = kernel_code.lower()

    # ===================================================================
    # 1. Verify that local arrays were promoted to 3D
    #    (ztmp and zout should have ngpblks or block_dim size as last dim)
    # ===================================================================
    kernel_vars = {v.name.lower(): v for v in kernel_routine.variables
                   if hasattr(v, 'name')}

    promoted_locals = []
    for name in ('ztmp', 'zout'):
        if name in kernel_vars:
            v = kernel_vars[name]
            if hasattr(v, 'shape') and v.shape:
                promoted_locals.append(v)

    print(f"\nKernel variables: {list(kernel_vars.keys())}")
    print(f"Promoted locals: {[(v.name, v.shape) for v in promoted_locals]}")

    # Check at least one local was promoted to 3D
    locals_3d = [v for v in promoted_locals if len(v.shape) >= 3]
    # Also accept 2D locals that gained ngpblks
    locals_with_block = [v for v in promoted_locals
                         if any(str(d).lower() in [s.lower() for s in block_dim.size_expressions]
                                for d in v.shape)]

    has_promoted = len(locals_3d) > 0 or len(locals_with_block) > 0
    print(f"3D locals: {[(str(v), v.shape) for v in locals_3d]}")
    print(f"Locals with block dim: {[(str(v), v.shape) for v in locals_with_block]}")

    assert has_promoted, (
        f"Expected local arrays to be promoted to 3D with block dimension!\n"
        f"process_driver should have promoted ztmp(klon, klev) to\n"
        f"ztmp(klon, klev, ngpblks).  If this assertion fails, the test\n"
        f"setup may be wrong — the driver must call the kernel with\n"
        f"!$loki small-kernels for promotion to happen.\n"
        f"Kernel variables: {list(kernel_vars.keys())}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # ===================================================================
    # 2. Verify that promoted locals appear in !$acc enter data create
    #    THIS IS THE ASSERTION EXPECTED TO FAIL (the bug)
    # ===================================================================

    # Collect all pragma nodes in the kernel
    all_pragmas = FindNodes(Pragma).visit(kernel_routine.body)
    all_spec_pragmas = FindNodes(Pragma).visit(kernel_routine.spec)
    all_pragmas_combined = all_pragmas + all_spec_pragmas

    # Look for !$acc enter data create OR !$loki unstructured-data create
    enter_data_pragmas = [
        p for p in all_pragmas_combined
        if (p.keyword.lower() == 'acc'
            and 'enter' in p.content.lower()
            and 'data' in p.content.lower()
            and 'create' in p.content.lower())
        or (p.keyword.lower() == 'loki'
            and 'unstructured-data' in p.content.lower()
            and 'create' in p.content.lower())
    ]

    print(f"\nAll pragmas in kernel: {[fgen(p) for p in all_pragmas_combined]}")
    print(f"Enter data create pragmas: {[fgen(p) for p in enter_data_pragmas]}")

    # Get the names of all promoted locals
    promoted_names = set()
    for v in promoted_locals:
        if any(str(d).lower() in [s.lower() for s in block_dim.size_expressions]
               for d in v.shape):
            promoted_names.add(v.name.lower())

    print(f"Promoted local names needing !$acc enter data create: {promoted_names}")

    # Check that each promoted local appears in an enter data create pragma
    found_in_pragma = set()
    for p in enter_data_pragmas:
        content = p.content.lower()
        for name in promoted_names:
            if name in content:
                found_in_pragma.add(name)

    missing = promoted_names - found_in_pragma

    assert len(missing) == 0, (
        f"Promoted local arrays must appear in !$acc enter data create!\n"
        f"\n"
        f"This is the promoted-locals bug: LowerBlockIndexTransformation\n"
        f".process_driver promotes the kernel's local arrays from 2D to 3D\n"
        f"(appending block_dim size) but does NOT add !$loki unstructured-data\n"
        f"create/delete for them.  When the kernel itself has no !$loki\n"
        f"small-kernels calls, process_kernel exits early and never reaches\n"
        f"the code (block_index_transformations.py lines 748-755) that adds\n"
        f"the data directives.\n"
        f"\n"
        f"Result: arrays like ztmp(klon, klev, ngpblks) are allocated on the\n"
        f"host but never on the GPU.  Any GPU kernel touching them will crash.\n"
        f"\n"
        f"BUILD: lapineb.scc_stack.F90 has promoted locals (ZSPT1, ZUT1, ...)\n"
        f"with NGPBLKS dimension but no !$acc enter data create.\n"
        f"cpg_gp_hyd.scc_stack.F90 correctly has !$acc enter data create\n"
        f"because cpg_gp_hyd has !$loki small-kernels calls, so process_kernel\n"
        f"runs fully and adds the directives.\n"
        f"\n"
        f"Missing from !$acc enter data create: {missing}\n"
        f"Promoted locals: {[(v.name, str(v.shape)) for v in promoted_locals]}\n"
        f"Enter data pragmas found: {[fgen(p) for p in enter_data_pragmas]}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # ===================================================================
    # 3. Verify matching !$acc exit data delete
    #    (only check if enter data was found — otherwise the enter data
    #    assertion above already failed)
    # ===================================================================
    exit_data_pragmas = [
        p for p in all_pragmas_combined
        if (p.keyword.lower() == 'acc'
            and 'exit' in p.content.lower()
            and 'data' in p.content.lower()
            and 'delete' in p.content.lower())
        or (p.keyword.lower() == 'loki'
            and 'exit' in p.content.lower()
            and 'unstructured-data' in p.content.lower()
            and 'delete' in p.content.lower())
    ]

    print(f"Exit data delete pragmas: {[fgen(p) for p in exit_data_pragmas]}")

    if enter_data_pragmas:
        assert len(exit_data_pragmas) >= 1, (
            f"Kernel has !$acc enter data create but no matching\n"
            f"!$acc exit data delete for promoted local arrays!\n"
            f"Kernel code:\n{kernel_code}"
        )


# ===================================================================
# Test: nested sub-kernel without block loop must NOT get its own
#       ISTSZ/ZSTACK/ALLOCATE (pool allocator stack allocation)
# ===================================================================
#
# Models the lapineb -> larcinb pattern:
#
# * driver_nested_stack  (driver)
#     └─ DO ibl = 1, ngpblks
#          !$loki small-kernels
#          call kernel_nested_stack(...)
#
# * kernel_nested_stack  (kernel, targets=['sub_kernel_nested_stack'])
#     ├─ ztmp1(klon, klev) — local temp
#     ├─ call sub_kernel_nested_stack(...)  ← NO !$loki small-kernels
#     └─ gets block loop from LowerBlockIndexTransformation
#
# * sub_kernel_nested_stack  (kernel, targets=['sub_sub_kernel_nested_stack'])
#     ├─ zlocal(klon, klev) — local temp (pool-allocated to Cray pointer)
#     ├─ DO jfld = 1, nfld      ← regular loop, NOT a block loop
#     │     call sub_sub_kernel_nested_stack(...)
#     │   END DO
#     └─ No block loop, no !$loki small-kernels
#
# * sub_sub_kernel_nested_stack  (kernel, targets=[])
#     └─ ztemp(klon, klev) — local temp (pool-allocated to Cray pointer)
#
# After removing _filter_block_dim_loops, find_driver_loops finds the
# DO jfld loop in sub_kernel_nested_stack (because it contains a call
# to sub_sub_kernel_nested_stack, a target).  The pool allocator then
# creates a spurious ISTSZ/ZSTACK/ALLOCATE for the sub-kernel.
#
# In the working reference, larcinb's ISTSZ/ZSTACK/ALLOCATE are
# COMMENTED OUT — the sub-kernel should use the parent's stack via
# YLSTACK_L, not allocate its own.
# ===================================================================

FCODE_DRIVER_NESTED_STACK = """
module driver_nested_stack_mod
  implicit none
contains
  subroutine driver_nested_stack(ngpblks, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_nested_stack.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(:,:,:)
    integer :: ibl
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)
      !$loki small-kernels
      call kernel_nested_stack(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
  end subroutine driver_nested_stack
end module driver_nested_stack_mod
""".strip()

FCODE_KERNEL_NESTED_STACK = """
module kernel_nested_stack_mod
  implicit none
contains
  subroutine kernel_nested_stack(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "sub_kernel_nested_stack.intfb.h"
    integer, intent(in)  :: klon, klev
    type(bnds_type), intent(in) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(klon, klev)
    real :: ztmp1(klon, klev)
    real :: ztmp2(klon)
    integer :: jrof, jk
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        ztmp1(jrof, jk) = pfield(jrof, jk) * 2.0
      end do
    end do
    do jrof = bnds%kidia, bnds%kfdia
      ztmp2(jrof) = 0.0
    end do
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        ztmp2(jrof) = ztmp2(jrof) + ztmp1(jrof, jk)
      end do
    end do
    do jrof = bnds%kidia, bnds%kfdia
      pfield(jrof, 1) = ztmp2(jrof)
    end do
    ! Call sub-kernel -- NO !$loki small-kernels pragma.
    ! This models lapineb calling larcinb without the pragma.
    ! ngpblks is passed because larcinb accesses NGPBLKS via YDGEOMETRY.
    call sub_kernel_nested_stack(klon, klev, ngpblks, bnds, opts, pfield)
  end subroutine kernel_nested_stack
end module kernel_nested_stack_mod
""".strip()

FCODE_SUB_KERNEL_NESTED_STACK = """
module sub_kernel_nested_stack_mod
  implicit none
contains
  subroutine sub_kernel_nested_stack(klon, klev, ngpblks, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "sub_sub_kernel_nested_stack.intfb.h"
    integer, intent(in)  :: klon, klev, ngpblks
    type(bnds_type), intent(in) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(klon, klev)
    real :: zlocal(klon, klev)
    integer :: jrof, jk, jfld, nfld
    nfld = 3
    ! Local compute on zlocal — models larcinb's ZGMVF usage
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        zlocal(jrof, jk) = pfield(jrof, jk) * 0.5
      end do
    end do
    ! Loop over fields calling a sub-sub-kernel (target) — models
    ! larcinb's DO JFLD loop with calls to LAITRE_GMV_LOKI etc.
    ! This is a REGULAR loop (not a block loop), but find_driver_loops
    ! will return it because it contains a call to a target.
    do jfld = 1, nfld
      call sub_sub_kernel_nested_stack(klon, klev, bnds, opts, pfield)
    end do
    ! More local compute
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        pfield(jrof, jk) = pfield(jrof, jk) + zlocal(jrof, jk)
      end do
    end do
  end subroutine sub_kernel_nested_stack
end module sub_kernel_nested_stack_mod
""".strip()

FCODE_SUB_SUB_KERNEL_NESTED_STACK = """
module sub_sub_kernel_nested_stack_mod
  implicit none
contains
  subroutine sub_sub_kernel_nested_stack(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in)  :: klon, klev
    type(bnds_type), intent(in) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(klon, klev)
    real :: ztemp(klon, klev)
    integer :: jrof, jk
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        ztemp(jrof, jk) = pfield(jrof, jk) * 1.1
      end do
    end do
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        pfield(jrof, jk) = ztemp(jrof, jk)
      end do
    end do
  end subroutine sub_sub_kernel_nested_stack
end module sub_sub_kernel_nested_stack_mod
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_no_stack_allocation_in_nested_sub_kernel_without_block_loop(
    frontend, horizontal, block_dim, tmp_path
):
    """
    A nested sub-kernel called WITHOUT ``!$loki small-kernels`` that has
    no block loop must NOT get its own ISTSZ/ZSTACK/ALLOCATE.  It should
    use the parent kernel's stack pointer passed via YLSTACK_L.

    This test models the ``lapineb`` -> ``larcinb`` pattern:

    * ``kernel_nested_stack`` (modelling ``lapineb``) calls
      ``sub_kernel_nested_stack`` (modelling ``larcinb``) WITHOUT
      ``!$loki small-kernels``.
    * ``sub_kernel_nested_stack`` has a local 2D temporary ``zlocal``
      (modelling ``ZGMVF``) and a ``DO jfld = 1, nfld`` loop that
      calls ``sub_sub_kernel_nested_stack`` (a target, modelling
      ``LAITRE_GMV_LOKI``).
    * After removing ``_filter_block_dim_loops`` (the Bug 2 fix),
      ``find_driver_loops`` finds the ``DO jfld`` loop because it
      contains a call to a target.  The pool allocator then creates a
      spurious ISTSZ/ZSTACK/ALLOCATE for the sub-kernel.
    * In the working reference, ``larcinb``'s ISTSZ/ZSTACK/ALLOCATE are
      all COMMENTED OUT — the sub-kernel should use the parent's stack.

    BUILD double-check-1 bug:
      ``larcinb.scc_stack.F90`` has ACTIVE ``ISTSZ = MAX(...)``,
      ``ALLOCATE(ZSTACK(...))``, ``!$acc data create(ZSTACK)``

    Working reference:
      ``larcinb.scc_stack.F90`` has them all COMMENTED OUT:
      ``! ISTSZ = MAX(...)``, ``! ALLOCATE(ZSTACK(...))``, etc.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_NESTED_STACK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_source = Sourcefile.from_source(
        FCODE_KERNEL_NESTED_STACK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    sub_source = Sourcefile.from_source(
        FCODE_SUB_KERNEL_NESTED_STACK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    sub_sub_source = Sourcefile.from_source(
        FCODE_SUB_SUB_KERNEL_NESTED_STACK, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver_nested_stack']
    kernel_routine = kernel_source['kernel_nested_stack']
    sub_routine = sub_source['sub_kernel_nested_stack']
    sub_sub_routine = sub_sub_source['sub_sub_kernel_nested_stack']

    # Enrich call graph: 4-level hierarchy
    driver_routine.enrich(kernel_routine)
    kernel_routine.enrich(sub_routine)
    sub_routine.enrich(sub_sub_routine)

    # Create items
    driver_item = ProcedureItem(
        name='driver_nested_stack_mod#driver_nested_stack', source=driver_source
    )
    kernel_item = ProcedureItem(
        name='kernel_nested_stack_mod#kernel_nested_stack', source=kernel_source
    )
    sub_item = ProcedureItem(
        name='sub_kernel_nested_stack_mod#sub_kernel_nested_stack', source=sub_source
    )
    sub_sub_item = ProcedureItem(
        name='sub_sub_kernel_nested_stack_mod#sub_sub_kernel_nested_stack', source=sub_sub_source
    )

    sgraph = SGraph.from_dict({
        driver_item: [kernel_item],
        kernel_item: [sub_item],
        sub_item: [sub_sub_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item, ['kernel_nested_stack']),
        (kernel_routine, 'kernel', kernel_item, ['sub_kernel_nested_stack']),
        (sub_routine, 'kernel', sub_item, ['sub_sub_kernel_nested_stack']),
        (sub_sub_routine, 'kernel', sub_sub_item, []),
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

    driver_code = fgen(driver_routine)
    kernel_code = fgen(kernel_routine)
    sub_code = fgen(sub_routine)
    sub_sub_code = fgen(sub_sub_routine)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"Nested sub-kernel stack allocation test")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL ---\n{kernel_code}")
    print(f"\n--- SUB KERNEL ---\n{sub_code}")
    print(f"\n--- SUB SUB KERNEL ---\n{sub_sub_code}")
    print(f"{'='*70}\n")

    # ===================================================================
    # A. Sub-sub-kernel verification: local temp must be pool-allocated
    # ===================================================================
    sub_sub_intrinsics = FindNodes(Intrinsic).visit(sub_sub_routine.spec)
    sub_sub_cray_ptrs = [i for i in sub_sub_intrinsics
                         if 'pointer' in i.text.lower()
                         and 'ztemp' in i.text.lower()]

    print(f"\nSub-sub-kernel Cray pointer decls: {[i.text for i in sub_sub_cray_ptrs]}")

    assert len(sub_sub_cray_ptrs) >= 1, (
        f"Sub-sub-kernel must have a Cray pointer for ztemp!\n"
        f"ztemp(klon, klev) stays 2D and should be pool-allocated.\n"
        f"Sub-sub-kernel code:\n{sub_sub_code}"
    )

    # ===================================================================
    # B. Sub-kernel verification: zlocal must be pool-allocated
    # ===================================================================
    sub_intrinsics = FindNodes(Intrinsic).visit(sub_routine.spec)
    sub_cray_ptrs = [i for i in sub_intrinsics
                     if 'pointer' in i.text.lower()
                     and 'zlocal' in i.text.lower()]

    print(f"\nSub-kernel Cray pointer decls for zlocal: {[i.text for i in sub_cray_ptrs]}")

    assert len(sub_cray_ptrs) >= 1, (
        f"Sub-kernel must have a Cray pointer for zlocal!\n"
        f"zlocal(klon, klev) stays 2D and should be pool-allocated.\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # B2. Sub-kernel must have IP_zlocal = YLSTACK_L
    sub_assigns = FindNodes(Assignment).visit(sub_routine.body)
    ip_assigns = [a for a in sub_assigns
                  if 'ip_' in str(a.lhs).lower()
                  and 'zlocal' in str(a.lhs).lower()
                  and 'ylstack_l' in str(a.rhs).lower()]

    print(f"Sub-kernel IP_zlocal assignments: {[fgen(a) for a in ip_assigns]}")

    assert len(ip_assigns) >= 1, (
        f"Sub-kernel must have IP_zlocal = YLSTACK_L assignment!\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # ===================================================================
    # C. Sub-kernel must NOT have its own ISTSZ/ZSTACK/ALLOCATE
    #    THIS IS THE KEY BUG 4 ASSERTION — EXPECTED TO FAIL
    # ===================================================================
    #
    # The sub-kernel (modelling larcinb) has no block loop, no
    # !$loki small-kernels calls.  It should use the parent kernel's
    # stack via YLSTACK_L, NOT allocate its own.
    #
    # After removing _filter_block_dim_loops, find_driver_loops finds
    # the DO jfld loop (because it contains a call to
    # sub_sub_kernel_nested_stack, a target).  The pool allocator then
    # creates a spurious ISTSZ/ZSTACK/ALLOCATE.
    # ===================================================================

    sub_var_names = [str(v).lower() for v in sub_routine.variables]
    has_istsz = 'istsz' in sub_var_names

    print(f"\nSub-kernel variable names: {sub_var_names}")
    print(f"Sub-kernel has ISTSZ: {has_istsz}")

    assert not has_istsz, (
        f"Sub-kernel must NOT have ISTSZ declared!\n"
        f"\n"
        f"This is the nested-stack-allocation bug: the sub-kernel has no\n"
        f"block loop, no !$loki small-kernels calls.  It should use the\n"
        f"parent kernel's stack via YLSTACK_L, not allocate its own.\n"
        f"\n"
        f"After removing _filter_block_dim_loops, find_driver_loops finds\n"
        f"the DO jfld loop (because it contains a call to\n"
        f"sub_sub_kernel_nested_stack, a target).  The pool allocator then\n"
        f"creates a spurious ISTSZ/ZSTACK/ALLOCATE for the sub-kernel.\n"
        f"\n"
        f"BUILD double-check-1: larcinb.scc_stack.F90 has ACTIVE\n"
        f"ISTSZ = MAX(...), ALLOCATE(ZSTACK(...)), !$acc data create(ZSTACK)\n"
        f"Working: all COMMENTED OUT.\n"
        f"\n"
        f"Sub-kernel variables: {sub_var_names}\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # C2. Sub-kernel must NOT have ZSTACK allocated
    sub_allocs = FindNodes(Allocation).visit(sub_routine.body)
    zstack_allocs = [a for a in sub_allocs
                     if any('zstack' in str(v).lower() for v in a.variables)]

    print(f"Sub-kernel ZSTACK allocations: {[fgen(a) for a in zstack_allocs]}")

    assert len(zstack_allocs) == 0, (
        f"Sub-kernel must NOT have ZSTACK allocated!\n"
        f"The sub-kernel should use the parent's stack, not its own.\n"
        f"ZSTACK allocations found: {[fgen(a) for a in zstack_allocs]}\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # C3. Sub-kernel must NOT have YLSTACK_L = LOC(ZSTACK(...))
    loc_assigns = [a for a in sub_assigns
                   if 'ylstack_l' in str(a.lhs).lower()
                   and 'loc' in str(a.rhs).lower()
                   and 'zstack' in str(a.rhs).lower()]

    print(f"Sub-kernel LOC(ZSTACK) assignments: {[fgen(a) for a in loc_assigns]}")

    assert len(loc_assigns) == 0, (
        f"Sub-kernel must NOT have YLSTACK_L = LOC(ZSTACK(...))!\n"
        f"The sub-kernel should receive its stack pointer from the parent\n"
        f"via YDSTACK_L argument, not compute its own from a local ZSTACK.\n"
        f"LOC(ZSTACK) assignments: {[fgen(a) for a in loc_assigns]}\n"
        f"Sub-kernel code:\n{sub_code}"
    )

    # ===================================================================
    # D. Kernel verification: kernel SHOULD have ISTSZ/ZSTACK
    #    (it has a block loop from LowerBlockIndexTransformation)
    # ===================================================================
    kernel_var_names = [str(v).lower() for v in kernel_routine.variables]
    kernel_has_istsz = 'istsz' in kernel_var_names

    print(f"\nKernel variable names: {kernel_var_names}")
    print(f"Kernel has ISTSZ: {kernel_has_istsz}")

    assert kernel_has_istsz, (
        f"Kernel must have ISTSZ declared for pool allocator!\n"
        f"The kernel has a block loop and calls sub-kernels with\n"
        f"pool-allocated temporaries — it SHOULD get ISTSZ/ZSTACK.\n"
        f"Kernel variables: {kernel_var_names}\n"
        f"Kernel code:\n{kernel_code}"
    )

    # D2. Kernel must have ZSTACK allocated
    kernel_allocs = FindNodes(Allocation).visit(kernel_routine.body)
    kernel_zstack_allocs = [a for a in kernel_allocs
                            if any('zstack' in str(v).lower() for v in a.variables)]

    assert len(kernel_zstack_allocs) >= 1, (
        f"Kernel must have ZSTACK allocated for pool allocator!\n"
        f"Allocations found: {[fgen(a) for a in kernel_allocs]}\n"
        f"Kernel code:\n{kernel_code}"
    )


# ===================================================================
# Test: driver with multiple block loops, only one annotated with
#       !$loki small-kernels — pool allocator ISTSZ/ZSTACK must
#       still be generated for the driver
# ===================================================================
#
# Models the ecphys_setup_layer_loki pattern:
#
# * driver_multi_loop  (driver)
#     ├─ DO ibl = 1, ngpblks
#     │     call kernel_a_multi(...)       ← NO !$loki small-kernels
#     │   END DO
#     ├─ DO ibl = 1, ngpblks
#     │     !$loki small-kernels
#     │     call kernel_b_multi(...)       ← HAS !$loki small-kernels
#     │   END DO
#     └─ DO ibl = 1, ngpblks
#           call kernel_c_multi(...)       ← NO !$loki small-kernels
#         END DO
#
# * kernel_a_multi  (kernel, targets=[])
#     └─ ztmp_a(klon, klev) — local temp → pool-allocated (Cray pointer)
#
# * kernel_b_multi  (kernel, targets=[])
#     └─ simple compute, no local temps needing pool allocation
#        (gets block loop from LowerBlockIndexTransformation via
#        !$loki small-kernels)
#
# * kernel_c_multi  (kernel, targets=[])
#     └─ ztmp_c(klon, klev) — local temp → pool-allocated (Cray pointer)
#
# In ecphys_setup_layer_loki:
#   Loop 1 → GPMKTEND (no pragma, needs stack)
#   Loop 3 → GPGEO (!$loki small-kernels, block loop hoisted in)
#   Loop 5 → COS_SZA, GPHPRE_EXPL (no pragma, need stack)
#
# The driver must have ISTSZ/ZSTACK created, aggregating stack needs
# from kernel_a and kernel_c.  Each driver loop containing calls to
# stack-needing kernels must have YLSTACK_L = LOC(ZSTACK(1, IBL)).
# ===================================================================

FCODE_DRIVER_MULTI_LOOP = """
module driver_multi_loop_mod
  implicit none
contains
  subroutine driver_multi_loop(ngpblks, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_a_multi.intfb.h"
    #include "kernel_b_multi.intfb.h"
    #include "kernel_c_multi.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(:,:,:)
    integer :: ibl
    ! Loop 1: call kernel_a WITHOUT !$loki small-kernels
    ! Models ecphys_setup_layer's GPMKTEND loop
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)
      call kernel_a_multi(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
    ! Loop 2: call kernel_b WITH !$loki small-kernels
    ! Models ecphys_setup_layer's GPGEO loop
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)
      !$loki small-kernels
      call kernel_b_multi(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
    ! Loop 3: call kernel_c WITHOUT !$loki small-kernels
    ! Models ecphys_setup_layer's COS_SZA/GPHPRE_EXPL loop
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)
      call kernel_c_multi(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
  end subroutine driver_multi_loop
end module driver_multi_loop_mod
""".strip()

FCODE_KERNEL_A_MULTI = """
module kernel_a_multi_mod
  implicit none
contains
  subroutine kernel_a_multi(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in)  :: klon, klev
    type(bnds_type), intent(in) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(klon, klev)
    real :: ztmp_a(klon, klev)
    integer :: jrof, jk
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        ztmp_a(jrof, jk) = pfield(jrof, jk) * 2.0
      end do
    end do
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        pfield(jrof, jk) = ztmp_a(jrof, jk)
      end do
    end do
  end subroutine kernel_a_multi
end module kernel_a_multi_mod
""".strip()

FCODE_KERNEL_B_MULTI = """
module kernel_b_multi_mod
  implicit none
contains
  subroutine kernel_b_multi(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in)  :: klon, klev
    type(bnds_type), intent(in) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(klon, klev)
    integer :: jrof, jk
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        pfield(jrof, jk) = pfield(jrof, jk) + 1.0
      end do
    end do
  end subroutine kernel_b_multi
end module kernel_b_multi_mod
""".strip()

FCODE_KERNEL_C_MULTI = """
module kernel_c_multi_mod
  implicit none
contains
  subroutine kernel_c_multi(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in)  :: klon, klev
    type(bnds_type), intent(in) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(klon, klev)
    real :: ztmp_c(klon, klev)
    integer :: jrof, jk
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        ztmp_c(jrof, jk) = pfield(jrof, jk) * 3.0
      end do
    end do
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        pfield(jrof, jk) = ztmp_c(jrof, jk)
      end do
    end do
  end subroutine kernel_c_multi
end module kernel_c_multi_mod
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_multiple_driver_loops_with_one_small_kernels_call(
    frontend, horizontal, block_dim, tmp_path
):
    """
    A driver with multiple block loops must get ISTSZ/ZSTACK even when
    only ONE of the loops has a ``!$loki small-kernels`` call.

    This test models the ``ecphys_setup_layer_loki`` pattern:

    * Loop 1: calls ``kernel_a`` WITHOUT ``!$loki small-kernels``
      — ``kernel_a`` has a local 2D temporary that needs pool allocation
      (models ``GPMKTEND``)
    * Loop 2: calls ``kernel_b`` WITH ``!$loki small-kernels``
      — ``kernel_b`` is a simple compute kernel, gets its block loop
      hoisted in by ``LowerBlockIndexTransformation`` (models ``GPGEO``)
    * Loop 3: calls ``kernel_c`` WITHOUT ``!$loki small-kernels``
      — ``kernel_c`` has a local 2D temporary that needs pool allocation
      (models ``COS_SZA``, ``GPHPRE_EXPL``)

    In ``ecphys_setup_layer.scc_stack.F90``:
    * ISTSZ is computed as MAX(...) across all loops' stack requirements
    * ZSTACK is allocated with dimensions (MAX(ISTSZ,1), NGPBLKS)
    * Each driver loop has ``YLSTACK_L = LOC(ZSTACK(1, IBL))``

    The driver must have pool allocator infrastructure (ISTSZ, ZSTACK,
    ALLOCATE, per-loop LOC assignments) even though only one of its
    three loops uses ``!$loki small-kernels``.
    """
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    driver_source = Sourcefile.from_source(
        FCODE_DRIVER_MULTI_LOOP, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_a_source = Sourcefile.from_source(
        FCODE_KERNEL_A_MULTI, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_b_source = Sourcefile.from_source(
        FCODE_KERNEL_B_MULTI, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    kernel_c_source = Sourcefile.from_source(
        FCODE_KERNEL_C_MULTI, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver_multi_loop']
    kernel_a_routine = kernel_a_source['kernel_a_multi']
    kernel_b_routine = kernel_b_source['kernel_b_multi']
    kernel_c_routine = kernel_c_source['kernel_c_multi']

    # Enrich call graph
    driver_routine.enrich(kernel_a_routine)
    driver_routine.enrich(kernel_b_routine)
    driver_routine.enrich(kernel_c_routine)

    # Create items
    driver_item = ProcedureItem(
        name='driver_multi_loop_mod#driver_multi_loop', source=driver_source
    )
    kernel_a_item = ProcedureItem(
        name='kernel_a_multi_mod#kernel_a_multi', source=kernel_a_source
    )
    kernel_b_item = ProcedureItem(
        name='kernel_b_multi_mod#kernel_b_multi', source=kernel_b_source
    )
    kernel_c_item = ProcedureItem(
        name='kernel_c_multi_mod#kernel_c_multi', source=kernel_c_source
    )

    sgraph = SGraph.from_dict({
        driver_item: [kernel_a_item, kernel_b_item, kernel_c_item],
    })

    items_in_order = [
        (driver_routine, 'driver', driver_item,
         ['kernel_a_multi', 'kernel_b_multi', 'kernel_c_multi']),
        (kernel_a_routine, 'kernel', kernel_a_item, []),
        (kernel_b_routine, 'kernel', kernel_b_item, []),
        (kernel_c_routine, 'kernel', kernel_c_item, []),
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

    driver_code = fgen(driver_routine)
    kernel_a_code = fgen(kernel_a_routine)
    kernel_b_code = fgen(kernel_b_routine)
    kernel_c_code = fgen(kernel_c_routine)

    # Print all generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"Multiple driver loops with one !$loki small-kernels call test")
    print(f"{'='*70}")
    print(f"\n--- DRIVER ---\n{driver_code}")
    print(f"\n--- KERNEL A ---\n{kernel_a_code}")
    print(f"\n--- KERNEL B ---\n{kernel_b_code}")
    print(f"\n--- KERNEL C ---\n{kernel_c_code}")
    print(f"{'='*70}\n")

    # ===================================================================
    # A. Kernel A verification: ztmp_a must be pool-allocated
    # ===================================================================
    ka_intrinsics = FindNodes(Intrinsic).visit(kernel_a_routine.spec)
    ka_cray_ptrs = [i for i in ka_intrinsics
                    if 'pointer' in i.text.lower()
                    and 'ztmp_a' in i.text.lower()]

    print(f"\nKernel A Cray pointer decls: {[i.text for i in ka_cray_ptrs]}")

    assert len(ka_cray_ptrs) >= 1, (
        f"kernel_a must have a Cray pointer for ztmp_a!\n"
        f"ztmp_a(klon, klev) stays 2D because kernel_a has no\n"
        f"!$loki small-kernels calls — process_kernel exits early,\n"
        f"no promotion to 3D.  The pool allocator should convert it\n"
        f"to a Cray pointer.\n"
        f"Kernel A code:\n{kernel_a_code}"
    )

    # A2. Kernel A must have IP_ztmp_a = YLSTACK_L
    ka_assigns = FindNodes(Assignment).visit(kernel_a_routine.body)
    ka_ip_assigns = [a for a in ka_assigns
                     if 'ip_' in str(a.lhs).lower()
                     and 'ztmp_a' in str(a.lhs).lower()
                     and 'ylstack_l' in str(a.rhs).lower()]

    print(f"Kernel A IP_ztmp_a assignments: {[fgen(a) for a in ka_ip_assigns]}")

    assert len(ka_ip_assigns) >= 1, (
        f"kernel_a must have IP_ztmp_a = YLSTACK_L assignment!\n"
        f"Kernel A code:\n{kernel_a_code}"
    )

    # ===================================================================
    # B. Kernel C verification: ztmp_c must be pool-allocated
    # ===================================================================
    kc_intrinsics = FindNodes(Intrinsic).visit(kernel_c_routine.spec)
    kc_cray_ptrs = [i for i in kc_intrinsics
                    if 'pointer' in i.text.lower()
                    and 'ztmp_c' in i.text.lower()]

    print(f"\nKernel C Cray pointer decls: {[i.text for i in kc_cray_ptrs]}")

    assert len(kc_cray_ptrs) >= 1, (
        f"kernel_c must have a Cray pointer for ztmp_c!\n"
        f"ztmp_c(klon, klev) stays 2D — same reasoning as kernel_a.\n"
        f"Kernel C code:\n{kernel_c_code}"
    )

    # B2. Kernel C must have IP_ztmp_c = YLSTACK_L
    kc_assigns = FindNodes(Assignment).visit(kernel_c_routine.body)
    kc_ip_assigns = [a for a in kc_assigns
                     if 'ip_' in str(a.lhs).lower()
                     and 'ztmp_c' in str(a.lhs).lower()
                     and 'ylstack_l' in str(a.rhs).lower()]

    print(f"Kernel C IP_ztmp_c assignments: {[fgen(a) for a in kc_ip_assigns]}")

    assert len(kc_ip_assigns) >= 1, (
        f"kernel_c must have IP_ztmp_c = YLSTACK_L assignment!\n"
        f"Kernel C code:\n{kernel_c_code}"
    )

    # ===================================================================
    # C. Driver verification: must have ISTSZ/ZSTACK
    # ===================================================================
    #
    # The driver has three block loops.  Loops 1 and 3 call kernels
    # with pool-allocated temporaries.  The pool allocator must create
    # ISTSZ/ZSTACK for the driver, aggregating stack needs from all
    # driver loops.
    #
    # In ecphys_setup_layer.scc_stack.F90:
    #   ISTSZ = MAX(0, <terms from GPMKTEND path>, ...,
    #               <terms from COS_SZA path>, <terms from GPHPRE_EXPL path>)
    #   ALLOCATE(ZSTACK(MAX(ISTSZ, 1), NGPBLKS))
    #   !$acc data create(ZSTACK)
    # ===================================================================

    driver_var_names = [str(v).lower() for v in driver_routine.variables]
    has_istsz = 'istsz' in driver_var_names

    print(f"\nDriver variable names: {driver_var_names}")
    print(f"Driver has ISTSZ: {has_istsz}")

    assert has_istsz, (
        f"Driver must have ISTSZ declared for pool allocator!\n"
        f"\n"
        f"The driver has multiple block loops.  Loops 1 and 3 call\n"
        f"kernels with pool-allocated temporaries (kernel_a and kernel_c\n"
        f"have ztmp_a/ztmp_c converted to Cray pointers with non-zero\n"
        f"stack_size).  The pool allocator must create ISTSZ/ZSTACK for\n"
        f"the driver.\n"
        f"\n"
        f"This models ecphys_setup_layer_loki where loops 1 and 5\n"
        f"(GPMKTEND, COS_SZA/GPHPRE_EXPL) need stack allocation,\n"
        f"while loop 3 (GPGEO with !$loki small-kernels) gets its\n"
        f"block loop hoisted into the callee.\n"
        f"\n"
        f"Driver variables: {driver_var_names}\n"
        f"Driver code:\n{driver_code}"
    )

    # C2. Driver must have ZSTACK allocated
    driver_allocs = FindNodes(Allocation).visit(driver_routine.body)
    driver_zstack_allocs = [a for a in driver_allocs
                            if any('zstack' in str(v).lower()
                                   for v in a.variables)]

    print(f"Driver ZSTACK allocations: {[fgen(a) for a in driver_zstack_allocs]}")

    assert len(driver_zstack_allocs) >= 1, (
        f"Driver must have ZSTACK allocated!\n"
        f"Allocations found: {[fgen(a) for a in driver_allocs]}\n"
        f"Driver code:\n{driver_code}"
    )

    # C3. ISTSZ must include klon*klev contribution (from kernel_a
    #     and/or kernel_c's ztmp_a/ztmp_c)
    driver_assigns = FindNodes(Assignment).visit(driver_routine.body)
    istsz_assigns = [a for a in driver_assigns
                     if str(a.lhs).lower() == 'istsz']

    print(f"Driver ISTSZ assignments: {[fgen(a) for a in istsz_assigns]}")

    assert len(istsz_assigns) >= 1, (
        f"Driver must have an ISTSZ assignment!\n"
        f"Driver code:\n{driver_code}"
    )

    istsz_rhs = str(istsz_assigns[0].rhs).lower()
    has_size_terms = 'klon' in istsz_rhs or 'opts%klon' in istsz_rhs

    print(f"ISTSZ RHS: {istsz_rhs}")
    print(f"ISTSZ RHS contains size contribution: {has_size_terms}")

    assert has_size_terms, (
        f"ISTSZ must include stack size contributions from\n"
        f"kernel_a and kernel_c's pool-allocated temporaries!\n"
        f"ISTSZ RHS: {istsz_rhs}\n"
        f"Driver code:\n{driver_code}"
    )

    # ===================================================================
    # D. Each driver loop with stack-needing calls must have
    #    YLSTACK_L = LOC(ZSTACK(1, ...)) inside
    # ===================================================================
    driver_loops = FindNodes(Loop).visit(driver_routine.body)

    print(f"\nDriver loops: {[(str(l.variable), str(l.bounds)) for l in driver_loops]}")

    # Find all LOC(ZSTACK) assignments across all loops
    all_loc_assigns = []
    for dl in driver_loops:
        loop_assigns = FindNodes(Assignment).visit(dl.body)
        loc_assigns = [a for a in loop_assigns
                       if 'ylstack_l' in str(a.lhs).lower()
                       and 'loc' in str(a.rhs).lower()
                       and 'zstack' in str(a.rhs).lower()]
        if loc_assigns:
            all_loc_assigns.extend(loc_assigns)
            print(f"  Loop var={dl.variable}: LOC assigns: {[fgen(a) for a in loc_assigns]}")

    # We expect at least 2 LOC assigns (one in loop 1 for kernel_a,
    # one in loop 3 for kernel_c).  Loop 2 (kernel_b with
    # !$loki small-kernels) may or may not have one depending on
    # whether the pool allocator also adds LOC to that loop.
    assert len(all_loc_assigns) >= 2, (
        f"At least 2 driver loops must have YLSTACK_L = LOC(ZSTACK(1, ...))!\n"
        f"Loop 1 (kernel_a) and Loop 3 (kernel_c) both call kernels with\n"
        f"pool-allocated temporaries and must set up the stack pointer\n"
        f"per-block.\n"
        f"\n"
        f"This models ecphys_setup_layer_loki where loops 1 and 5 each\n"
        f"have YLSTACK_L = LOC(ZSTACK(1, IBL)).\n"
        f"\n"
        f"LOC(ZSTACK) assignments found: {[fgen(a) for a in all_loc_assigns]}\n"
        f"Driver code:\n{driver_code}"
    )


# ---------------------------------------------------------------------------
# Bug 6: Pointer derived types spuriously privatised in driver loop
# ---------------------------------------------------------------------------
#
# In annotate.py:annotate_driver_loop, the array-filtering path correctly
# excludes POINTER variables (line 351):
#     arrays = [v for v in arrays if not v.type.pointer]
#
# But the derived-type (struct) filtering path (lines 360-387) has NO
# equivalent pointer exclusion.  This means TYPE(...), POINTER locals
# — which are aliases to device-resident data (e.g. YDASTRO => ...)
# — get spuriously added to private(...).
#
# Real-world example:
#   rad_transfer_radiative_fluxes_layer.sccs_stack.F90:
#     Bug:     !$acc parallel loop gang private( YLSTACK_L, ydastro, yrephy, ydslphy )
#     Correct: !$acc parallel loop gang private( YLSTACK_L )
#
#   gpgeo_expl.scc_stack.F90:
#     Correct: !$acc parallel loop gang private( local_YLCPG_BNDS )
#     (local_YLCPG_BNDS is a non-pointer, truly local derived-type → correctly privatised)
# ---------------------------------------------------------------------------

# Fortran code for directly testing annotate_driver_loop.
# Models rad_transfer_radiative_fluxes_layer:
#   - block loop with !$loki loop driver vector_length(klon)
#   - TYPE(...), POINTER :: zptr_opts — should NOT be privatised
#   - TYPE(bnds_type) :: zlocal_bnds — SHOULD be privatised
#   - Call to a target (leaf_kernel) inside the loop
FCODE_ANNOTATE_PTR_KERNEL = """
module annotate_ptr_kernel_mod
  implicit none
  contains
  subroutine annotate_ptr_kernel(klon, klev, kst, kend, field_in, field_out)
    integer, intent(in) :: klon, klev, kst, kend
    real, intent(in) :: field_in(klon, klev)
    real, intent(inout) :: field_out(klon, klev)
    integer :: jrof, jlev

    do jlev = 1, klev
      do jrof = kst, kend
        field_out(jrof, jlev) = field_in(jrof, jlev) * 2.0
      end do
    end do
  end subroutine annotate_ptr_kernel
end module annotate_ptr_kernel_mod
""".strip()

FCODE_ANNOTATE_PTR_ROUTINE = """
module annotate_ptr_routine_mod
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  contains
  subroutine annotate_ptr_routine(klon, klev, ngpblks, ydopts, field_in, field_out)
    integer, intent(in) :: klon, klev, ngpblks
    type(opts_type), intent(in), target :: ydopts
    real, intent(in) :: field_in(klon, klev, ngpblks)
    real, intent(inout) :: field_out(klon, klev, ngpblks)

    type(opts_type), pointer :: zptr_opts
    type(bnds_type) :: zlocal_bnds
    integer :: ibl, kst, kend

    zptr_opts => ydopts

    !$loki loop driver vector_length(klon)
    do ibl = 1, ngpblks
      kst = 1
      kend = klon
      zlocal_bnds%kidia = kst
      zlocal_bnds%kfdia = kend
      zlocal_bnds%kbl = ibl

      ! Use zptr_opts inside loop so FindVariables sees it
      call annotate_ptr_kernel(zptr_opts%klon, klev, kst, kend, &
        & field_in(:,:,ibl), field_out(:,:,ibl))
    end do
  end subroutine annotate_ptr_routine
end module annotate_ptr_routine_mod
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI fails to parse pointer declarations')]
))
def test_pointer_derived_types_not_privatised_in_driver_loop(
        tmp_path, frontend, horizontal, block_dim):
    """
    Test that TYPE(...), POINTER locals are NOT added to the private(...)
    clause of !$acc parallel loop gang, while non-pointer local derived-type
    variables ARE correctly privatised.

    This test directly invokes SCCAnnotateTransformation on a routine
    that has a block loop with !$loki loop driver, containing both:
      - TYPE(opts_type), POINTER :: zptr_opts   (should NOT be privatised)
      - TYPE(bnds_type)          :: zlocal_bnds (SHOULD be privatised)
      - a call to a target kernel

    Models: rad_transfer_radiative_fluxes_layer where YDASTRO, YREPHY,
    YDSLPHY are TYPE(...), POINTER aliases to device-resident data and
    get spuriously added to private(...).

    Bug location: annotate.py:annotate_driver_loop, lines 356-387 —
    the structs path has no v.type.pointer filter.
    """
    from loki.transformations.single_column.annotate import SCCAnnotateTransformation

    # Parse shared type modules
    bnds_mod = Module.from_source(FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])
    opts_mod = Module.from_source(FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path])

    # Parse the kernel (target) and the routine under test
    kernel_source = Sourcefile.from_source(
        FCODE_ANNOTATE_PTR_KERNEL, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )
    routine_source = Sourcefile.from_source(
        FCODE_ANNOTATE_PTR_ROUTINE, frontend=frontend,
        definitions=[bnds_mod, opts_mod], xmods=[tmp_path]
    )

    routine = routine_source['annotate_ptr_routine']
    kernel = kernel_source['annotate_ptr_kernel']

    # Enrich the call graph so find_driver_loops can find targets
    routine.enrich(kernel)

    # Apply SCCAnnotateTransformation directly in kernel role
    # (matching how rad_transfer_radiative_fluxes_layer is processed —
    # it's a kernel with an internal block loop containing target calls).
    # privatise_derived_types=True matches the pipeline default.
    annotate = SCCAnnotateTransformation(
        block_dim=block_dim, privatise_derived_types=True
    )
    annotate.transform_subroutine(
        routine, role='kernel', targets=['annotate_ptr_kernel']
    )

    code = fgen(routine)

    # Print generated code for diagnostic purposes
    print(f"\n{'='*70}")
    print(f"Pointer derived types privatisation test")
    print(f"{'='*70}")
    print(f"\n--- ROUTINE ---\n{code}")
    print(f"{'='*70}\n")

    # ===================================================================
    # Extract private(...) clause from the generated code
    # ===================================================================
    private_match = re.search(r'private\(([^)]+)\)', code, re.IGNORECASE)

    print(f"private(...) match: {private_match}")
    if private_match:
        private_vars_str = private_match.group(1)
        private_var_list = [v.strip().lower() for v in private_vars_str.split(',')]
        print(f"Private variables: {private_var_list}")
    else:
        private_var_list = []
        print("No private(...) clause found in generated code!")

    # ===================================================================
    # A. zlocal_bnds SHOULD be in private(...)
    # ===================================================================
    assert 'zlocal_bnds' in private_var_list, (
        f"zlocal_bnds SHOULD be in private(...)!\n"
        f"\n"
        f"zlocal_bnds is a non-pointer, non-argument, truly local\n"
        f"derived-type variable modified per-block inside the loop.\n"
        f"It models local_YLCPG_BNDS in gpgeo_expl.scc_stack.F90\n"
        f"which is correctly privatised.\n"
        f"\n"
        f"Private clause: {private_vars_str if private_match else 'NONE'}\n"
        f"Generated code:\n{code}"
    )

    # ===================================================================
    # B. zptr_opts should NOT be in private(...)
    # ===================================================================
    # zptr_opts is a TYPE(...), POINTER local — an alias to device-resident
    # data.  Privatising it creates a gang-private copy that doesn't point
    # to the original data.
    #
    # In annotate.py:annotate_driver_loop, arrays correctly exclude pointers:
    #     arrays = [v for v in arrays if not v.type.pointer]
    # But the structs path has NO equivalent filter.
    #
    # Real-world: rad_transfer_radiative_fluxes_layer.sccs_stack.F90
    #   Bug:     private( YLSTACK_L, ydastro, yrephy, ydslphy )
    #   Correct: private( YLSTACK_L )
    #
    # THIS ASSERTION IS EXPECTED TO FAIL — it exposes the bug.
    assert 'zptr_opts' not in private_var_list, (
        f"zptr_opts should NOT be in private(...)!\n"
        f"\n"
        f"zptr_opts is a TYPE(opts_type), POINTER local variable\n"
        f"— an alias to device-resident data (zptr_opts => ydopts).\n"
        f"Privatising it creates a gang-private copy that doesn't\n"
        f"point to the original data.\n"
        f"\n"
        f"The array filtering path correctly excludes pointers:\n"
        f"  arrays = [v for v in arrays if not v.type.pointer]\n"
        f"But the derived-type (struct) path has NO equivalent filter.\n"
        f"\n"
        f"Fix: add 'structs = [v for v in structs if not v.type.pointer]'\n"
        f"in annotate.py:annotate_driver_loop after line 362.\n"
        f"\n"
        f"Real-world: rad_transfer_radiative_fluxes_layer.sccs_stack.F90\n"
        f"  Bug:     private( YLSTACK_L, ydastro, yrephy, ydslphy )\n"
        f"  Correct: private( YLSTACK_L )\n"
        f"\n"
        f"Private clause: {private_vars_str if private_match else 'NONE'}\n"
        f"Generated code:\n{code}"
    )
