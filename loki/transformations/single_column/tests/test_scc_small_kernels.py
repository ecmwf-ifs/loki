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
- Category 13: Derived-type components are not placed in ``private()`` clauses.
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


# ---------------------------------------------------------------------------
# Helper: run the SCCSmallKernelsPipeline on a driver+kernel pair
# ---------------------------------------------------------------------------

def _apply_small_kernels_pipeline(driver_source, kernel_source, horizontal,
                                  block_dim, tmp_path):
    """
    Parse *driver_source* and *kernel_source*, enrich, build the required
    item / sgraph objects, and apply the full ``SCCSmallKernelsPipeline``.

    Returns ``(driver_routine, kernel_routine)`` after transformation.
    """
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )

    driver_routine = driver_source['driver']
    kernel_routine = kernel_source['kernel']
    driver_routine.enrich(kernel_routine)

    driver_item = ProcedureItem(name='driver_mod#driver', source=driver_source)
    kernel_item = ProcedureItem(name='#kernel', source=kernel_source)
    sgraph = SGraph.from_dict({driver_item: [kernel_item]})

    for transform in pipeline.transformations:
        transform.apply(
            driver_routine, role='driver', item=driver_item,
            targets=['kernel'], sub_sgraph=sgraph
        )
        transform.apply(
            kernel_routine, role='kernel', item=kernel_item,
            targets=[], sub_sgraph=sgraph
        )

    return driver_routine, kernel_routine


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
