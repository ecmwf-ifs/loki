# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

"""
End-to-end tests for ``SCCSmallKernelsPipeline``.

Each test runs the full 12-step pipeline on a small Fortran call tree
that mimics production-style transformation shapes and verifies the
structural properties of the transformed IR.
"""

import re

import pytest

from loki import Dimension, Module, Sourcefile, fgen
from loki.batch import Pipeline, ProcedureItem, SGraph
from loki.frontend import OMNI, available_frontends
from loki.ir import nodes as ir, FindNodes
from loki.transformations import (
    BlockViewToFieldViewTransformation, DataOffloadDeepcopyAnalysis,
    DataOffloadDeepcopyTransformation, InlineTransformation,
    InjectBlockIndexTransformation, RemoveCodeTransformation, ReplaceKernels,
    SanitiseTransformation, SanitiseUnusedRoutineTransformation,
)
from loki.transformations.single_column import SCCSmallKernelsPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
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


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _parse_modules(frontend, tmp_path):
    """Parse and return the shared type modules."""
    bnds_mod = Module.from_source(
        FCODE_BNDS_TYPE_MOD, frontend=frontend, xmods=[tmp_path]
    )
    opts_mod = Module.from_source(
        FCODE_OPTS_TYPE_MOD, frontend=frontend, xmods=[tmp_path]
    )
    return [bnds_mod, opts_mod]


def _apply_pipeline(pipeline, items_in_order, sgraph):
    """
    Apply all transformations in the pipeline to the given items,
    respecting ``reverse_traversal`` ordering.
    """
    for transform in pipeline.transformations:
        order = items_in_order
        if getattr(transform, 'reverse_traversal', False):
            order = list(reversed(items_in_order))
        for routine, role, item, targets in order:
            transform.apply(
                routine, role=role, item=item,
                targets=targets, sub_sgraph=sgraph
            )


def _build_scc_stack_pipeline(horizontal, block_dim):
    """Build the full SCC stack pipeline, excluding packaging transforms."""
    replace_map = {
        'verdisint': {
            'routine': 'verdisint_gpu_opt',
            'ignore': True,
            'args': {
                'kst': {'position': 1},
                'ldacc': '.true.',
                'kend': {
                    'map_to': 'kend_in',
                    'placeholders': {
                        'geom_total': 'ydgeometry',
                        'geom_dim': 'ydgeometry',
                    },
                    'expr': 'MOD({geom_total}%YRGEM%NGPTOT, {geom_dim}%YRDIM%NPROMA)',
                },
                'ydgeometry': {
                    'map_to': 'kgpblks',
                    'member': 'YRDIM%NGPBLKS',
                },
            },
        }
    }
    pipeline = Pipeline()
    pipeline.append(SanitiseTransformation(
        resolve_associate_mappings=True, resolve_sequence_association=False
    ))
    pipeline.append(InlineTransformation(
        inline_internals=True, inline_marked=True, remove_dead_code=True,
        allowed_aliases='JROF', resolve_sequence_association=False
    ))
    pipeline.append(RemoveCodeTransformation(
        remove_marked_regions=True, remove_dead_code=False,
        call_names=('dr_hook', 'abor1'),
        intrinsic_names=('write(nulout', 'write(nulerr'),
        kernel_only=True, remove_unused_args=True
    ))
    pipeline.append(BlockViewToFieldViewTransformation(
        horizontal=horizontal, global_gfl_ptr=False
    ))
    pipeline.append(InjectBlockIndexTransformation(block_dim=block_dim))
    pipeline.append(DataOffloadDeepcopyAnalysis())
    pipeline.append(DataOffloadDeepcopyTransformation(mode='offload'))
    pipeline.extend(SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc',
        check_bounds=False, trim_vector_sections=True,
        privatise_derived_types=True, remove_unused_vars=True
    ))
    pipeline.append(ReplaceKernels(replace_kernels_map=replace_map))
    pipeline.append(SanitiseUnusedRoutineTransformation(
        routines=('verdisint',), stub_kind='error_stop'
    ))
    return pipeline


def _build_and_apply_2level(
        driver_source, kernel_source, horizontal, block_dim,
        driver_name, kernel_name, driver_item_name, kernel_item_name):
    """Build 2-level call tree, apply pipeline, return (driver, kernel)."""
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )
    driver = driver_source[driver_name]
    kernel = kernel_source[kernel_name]
    driver.enrich(kernel)

    d_item = ProcedureItem(name=driver_item_name, source=driver_source)
    k_item = ProcedureItem(name=kernel_item_name, source=kernel_source)
    sgraph = SGraph.from_dict({d_item: [k_item]})

    items = [
        (driver, 'driver', d_item, [kernel_name]),
        (kernel, 'kernel', k_item, []),
    ]
    _apply_pipeline(pipeline, items, sgraph)
    return driver, kernel


def _build_and_apply_3level(
        driver_source, mid_source, sub_source,
        horizontal, block_dim,
        driver_name, mid_name, sub_name,
        driver_item_name, mid_item_name, sub_item_name,
        mid_targets=None):
    """Build 3-level call tree, apply pipeline, return (driver, mid, sub)."""
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )
    driver = driver_source[driver_name]
    mid = mid_source[mid_name]
    sub = sub_source[sub_name]
    driver.enrich(mid)
    mid.enrich(sub)

    d_item = ProcedureItem(name=driver_item_name, source=driver_source)
    m_item = ProcedureItem(name=mid_item_name, source=mid_source)
    s_item = ProcedureItem(name=sub_item_name, source=sub_source)
    sgraph = SGraph.from_dict({d_item: [m_item], m_item: [s_item]})

    if mid_targets is None:
        mid_targets = [sub_name]
    items = [
        (driver, 'driver', d_item, [mid_name]),
        (mid, 'kernel', m_item, mid_targets),
        (sub, 'kernel', s_item, []),
    ]
    _apply_pipeline(pipeline, items, sgraph)
    return driver, mid, sub


def _build_and_apply_4level(
        driver_source, mid_source, sub_source, leaf_source,
        horizontal, block_dim,
        driver_name, mid_name, sub_name, leaf_name,
        driver_item_name, mid_item_name, sub_item_name, leaf_item_name,
        mid_targets=None, sub_targets=None):
    """Build 4-level call tree, apply pipeline, return routines."""
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )
    driver = driver_source[driver_name]
    mid = mid_source[mid_name]
    sub = sub_source[sub_name]
    leaf = leaf_source[leaf_name]
    driver.enrich(mid)
    mid.enrich(sub)
    sub.enrich(leaf)

    d_item = ProcedureItem(name=driver_item_name, source=driver_source)
    m_item = ProcedureItem(name=mid_item_name, source=mid_source)
    s_item = ProcedureItem(name=sub_item_name, source=sub_source)
    l_item = ProcedureItem(name=leaf_item_name, source=leaf_source)
    sgraph = SGraph.from_dict({d_item: [m_item], m_item: [s_item], s_item: [l_item]})

    if mid_targets is None:
        mid_targets = [sub_name]
    if sub_targets is None:
        sub_targets = [leaf_name]
    items = [
        (driver, 'driver', d_item, [mid_name]),
        (mid, 'kernel', m_item, mid_targets),
        (sub, 'kernel', s_item, sub_targets),
        (leaf, 'kernel', l_item, []),
    ]
    _apply_pipeline(pipeline, items, sgraph)
    return driver, mid, sub, leaf


def _build_and_apply_4level_full_pipeline(
        driver_source, mid_source, sub_source, leaf_source,
        horizontal, block_dim,
        driver_name, mid_name, sub_name, leaf_name,
        driver_item_name, mid_item_name, sub_item_name, leaf_item_name,
        mid_targets=None, sub_targets=None):
    """Build a 4-level call tree, apply the full SCC stack pipeline, and return routines."""
    pipeline = _build_scc_stack_pipeline(horizontal, block_dim)
    driver = driver_source[driver_name]
    mid = mid_source[mid_name]
    sub = sub_source[sub_name]
    leaf = leaf_source[leaf_name]
    driver.enrich(mid)
    mid.enrich(sub)
    sub.enrich(leaf)

    d_item = ProcedureItem(name=driver_item_name, source=driver_source)
    m_item = ProcedureItem(name=mid_item_name, source=mid_source)
    s_item = ProcedureItem(name=sub_item_name, source=sub_source)
    l_item = ProcedureItem(name=leaf_item_name, source=leaf_source)
    sgraph = SGraph.from_dict({d_item: [m_item], m_item: [s_item], s_item: [l_item]})

    if mid_targets is None:
        mid_targets = [sub_name]
    if sub_targets is None:
        sub_targets = [leaf_name]
    items = [
        (driver, 'driver', d_item, [mid_name]),
        (mid, 'kernel', m_item, mid_targets),
        (sub, 'kernel', s_item, sub_targets),
        (leaf, 'kernel', l_item, []),
    ]
    _apply_pipeline(pipeline, items, sgraph)
    return driver, mid, sub, leaf


def _has_block_loop(routine, block_dim):
    """Return True if ``routine`` contains a loop over a block_dim index."""
    for loop in FindNodes(ir.Loop).visit(routine.body):
        var = str(loop.variable).lower().replace('local_', '')
        if var in [idx.lower() for idx in block_dim.indices]:
            return True
    return False


def _get_block_loops(routine, block_dim):
    """Return all loops whose variable matches a block_dim index."""
    result = []
    for loop in FindNodes(ir.Loop).visit(routine.body):
        var = str(loop.variable).lower().replace('local_', '')
        if var in [idx.lower() for idx in block_dim.indices]:
            result.append(loop)
    return result


def _routine_var_names(routine):
    """Return set of lower-cased variable names from routine."""
    return {v.name.lower() for v in routine.variables}


def _find_calls_by_name(routine, callee_name):
    """Find all CallStatements in routine.body matching callee_name."""
    return [c for c in FindNodes(ir.CallStatement).visit(routine.body)
            if callee_name.lower() in str(c.name).lower()]


# ===================================================================
# Group 1: Top-level driver
# ===================================================================

FCODE_TOP_DRIVER = """
module top_driver_mod
  implicit none
contains
  subroutine top_driver(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "nested_driver.intfb.h"
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
      call nested_driver(opts%klon, opts%kflevg, bnds, opts, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine top_driver
end module top_driver_mod
""".strip()

FCODE_NESTED_DRIVER = """
subroutine nested_driver(klon, klev, bnds, opts, t, q)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "leaf_kernel.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do

  !$loki small-kernels
  call leaf_kernel(klon, klev, bnds, opts, t, q)
end subroutine nested_driver
""".strip()

FCODE_LEAF_KERNEL = """
subroutine leaf_kernel(klon, klev, bnds, opts, t, q)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
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


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_top_driver_pool_allocator_removed(frontend, horizontal, block_dim, tmp_path):
    """
    Top-level driver: after the pipeline, the
    driver must NOT have ISTSZ/ZSTACK/ALLOCATE/DEALLOCATE — pool
    allocator infrastructure is moved down into nested kernels.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    driver, _, _ = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    var_names = _routine_var_names(driver)
    assert 'istsz' not in var_names
    assert 'zstack' not in var_names

    allocs = FindNodes(ir.Allocation).visit(driver.body)
    deallocs = FindNodes(ir.Deallocation).visit(driver.body)
    assert not any('zstack' in str(a).lower() for a in allocs)
    assert not any('zstack' in str(d).lower() for d in deallocs)


# ===================================================================
# Group 2: Nested driver
# ===================================================================

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_nested_driver_gets_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    Nested driver kernel: after the pipeline, the
    nested driver must have a block-dimension loop with
    ``!$acc parallel loop gang``.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, nested, _ = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    assert _has_block_loop(nested, block_dim), (
        f"Nested driver should have a block-dimension loop.\n"
        f"Code:\n{fgen(nested)}"
    )

    # Check for !$acc parallel loop gang pragma on or near the block loop
    code_lower = fgen(nested).lower()
    assert 'acc parallel loop gang' in code_lower, (
        f"Nested driver block loop should have !$acc parallel loop gang.\n"
        f"Code:\n{fgen(nested)}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_nested_driver_gets_stack_args(frontend, horizontal, block_dim, tmp_path):
    """
    Nested driver kernel: in the per-driver-loop
    pool allocator, the nested driver receives stack args (YDSTACK_L)
    from the top-level driver and forwards them via YLSTACK_L.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, nested, _ = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    var_names = _routine_var_names(nested)
    # Nested driver receives stack args from parent (not its own ISTSZ/ZSTACK)
    assert 'ydstack_l' in var_names or 'ylstack_l' in var_names, (
        f"Nested driver should have YDSTACK_L or YLSTACK_L.\n"
        f"Vars: {sorted(var_names)}\nCode:\n{fgen(nested)}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_nested_driver_local_copies(frontend, horizontal, block_dim, tmp_path):
    """
    Nested driver: block loop must contain local
    copies (``local_bnds``, ``local_ibl``, etc.) and these must
    appear in the ``private(...)`` clause.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, nested, _ = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    code = fgen(nested)
    code_lower = code.lower()

    # Should have at least one local_ variable
    assert 'local_' in code_lower, (
        f"Nested driver should have local_ copies.\nCode:\n{code}"
    )

    # Check for private clause containing local_ variables
    private_match = re.search(r'private\(([^)]+)\)', code_lower)
    assert private_match, (
        f"Nested driver should have a private(...) clause.\nCode:\n{code}"
    )
    private_vars = private_match.group(1)
    assert 'local_' in private_vars, (
        f"private(...) clause should contain local_ variables.\n"
        f"private clause: {private_vars}\nCode:\n{code}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_nested_driver_passes_kwargs(frontend, horizontal, block_dim, tmp_path):
    """
    Nested driver: calls to leaf kernels must include
    BNDS and OPTS as keyword arguments after the pipeline.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, nested, _ = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    calls = _find_calls_by_name(nested, 'leaf_kernel')
    assert calls, f"Nested driver should still call leaf_kernel.\nCode:\n{fgen(nested)}"

    for call in calls:
        kwarg_names = [kw[0].lower() for kw in (call.kwarguments or ())]
        arg_strs = [str(a).lower() for a in call.arguments]
        all_args = kwarg_names + arg_strs
        # BNDS may be passed as positional (local_bnds) or kwarg
        has_bnds = any('bnds' in a for a in all_args)
        assert has_bnds, (
            f"Call to leaf_kernel should pass BNDS (positional or kwarg).\n"
            f"Args: {arg_strs}\nKwargs: {kwarg_names}\n"
            f"Call: {fgen(call)}\nCode:\n{fgen(nested)}"
        )


# ===================================================================
# Group 3: Leaf kernel
# ===================================================================

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_leaf_kernel_parallel_regions(frontend, horizontal, block_dim, tmp_path):
    """
    Leaf kernel: after the pipeline, the leaf
    kernel must have ``!$acc parallel loop gang`` regions with local
    copies of block indices.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, _, leaf = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    code = fgen(leaf)
    code_lower = code.lower()

    # Leaf kernel should have block loop(s) with acc parallel loop gang
    assert _has_block_loop(leaf, block_dim), (
        f"Leaf kernel should have a block-dimension loop.\nCode:\n{code}"
    )
    assert 'acc parallel loop gang' in code_lower, (
        f"Leaf kernel should have !$acc parallel loop gang.\nCode:\n{code}"
    )

    # Should have local_ copies
    assert 'local_' in code_lower, (
        f"Leaf kernel should have local_ copies of block indices.\n"
        f"Code:\n{code}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_leaf_kernel_temp_promotion(frontend, horizontal, block_dim, tmp_path):
    """
    Leaf kernel: local temporary arrays must get
    ``!$acc enter data create`` / ``!$acc exit data delete`` (or
    ``!$loki`` equivalents) after promotion.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, _, leaf = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    code = fgen(leaf)
    code_lower = code.lower()

    # Check for data management pragmas for ztmp
    has_enter_data = ('acc enter data' in code_lower or
                      'loki unstructured-data' in code_lower)
    # ztmp should be promoted and have data directives
    # (The pool allocator may convert it to Cray pointer instead;
    # either approach is valid)
    all_pragmas = FindNodes(ir.Pragma).visit(leaf.body)
    cray_ptrs = FindNodes(ir.GenericStmt).visit(leaf.spec)
    has_cray = any('ztmp' in i.text.lower() for i in cray_ptrs
                    if 'pointer' in i.text.lower())

    assert has_enter_data or has_cray, (
        f"Leaf kernel's ztmp should either have !$acc enter data create\n"
        f"or be pool-allocated via Cray pointer.\n"
        f"Pragmas: {[fgen(p) for p in all_pragmas]}\n"
        f"Cray pointers: {[i.text for i in cray_ptrs]}\n"
        f"Code:\n{code}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_leaf_kernel_acc_routine_removed(frontend, horizontal, block_dim, tmp_path):
    """
    Leaf kernel that gets ``!$acc parallel loop gang`` regions must
    NOT have ``!$acc routine seq`` — it now owns parallel regions.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_TOP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_NESTED_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, _, leaf = _build_and_apply_3level(
        driver_src, mid_src, sub_src, horizontal, block_dim,
        'top_driver', 'nested_driver', 'leaf_kernel',
        'top_driver_mod#top_driver', '#nested_driver', '#leaf_kernel'
    )

    code = fgen(leaf)
    code_lower = code.lower()

    # If it has acc parallel loop gang, it must NOT have acc routine seq
    if 'acc parallel loop gang' in code_lower:
        assert 'acc routine seq' not in code_lower, (
            f"Leaf kernel with !$acc parallel loop gang should NOT\n"
            f"have !$acc routine seq.\nCode:\n{code}"
        )


# ===================================================================
# Group 4: Thin wrapper
# ===================================================================

FCODE_WRAPPER_DRIVER = """
module wrapper_driver_mod
  implicit none
contains
  subroutine wrapper_driver(ngpblks, bnds, opts, phi, pt)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "thin_wrapper.intfb.h"
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
      call thin_wrapper(opts%klon, opts%kflevg, bnds%kidia, bnds%kfdia, phi(:,:,ibl), pt(:,:,ibl))
    end do
  end subroutine wrapper_driver
end module wrapper_driver_mod
""".strip()

FCODE_THIN_WRAPPER = """
subroutine thin_wrapper(klon, klev, kst, kend, phi, pt)
  implicit none
  #include "inner_compute.intfb.h"
  integer, intent(in) :: klon, klev, kst, kend
  real, intent(inout) :: phi(klon, klev)
  real, intent(in) :: pt(klon, klev)

  !$loki small-kernels
  call inner_compute(klon, klev, kst, kend, phi, pt)
end subroutine thin_wrapper
""".strip()

FCODE_INNER_COMPUTE = """
subroutine inner_compute(klon, klev, kst, kend, phi, pt)
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
end subroutine inner_compute
""".strip()

FCODE_DEEP_DRIVER = """
module deep_driver_mod
  implicit none
contains
  subroutine deep_driver(ngpblks, bnds, opts, phi, pt)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
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
      call mid_kernel(opts%klon, opts%kflevg, bnds, opts, phi(:,:,ibl), pt(:,:,ibl))
    end do
  end subroutine deep_driver
end module deep_driver_mod
""".strip()

FCODE_MID_KERNEL = """
subroutine mid_kernel(klon, klev, bnds, opts, phi, pt)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: phi(klon, klev)
  real, intent(in) :: pt(klon, klev)

  !$loki small-kernels
  call sub_kernel(klon, klev, bnds, opts, phi, pt)
end subroutine mid_kernel
""".strip()

FCODE_SUB_KERNEL = """
subroutine sub_kernel(klon, klev, bnds, opts, phi, pt)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: phi(klon, klev)
  real, intent(in) :: pt(klon, klev)
  real :: zsum(klon, klev)
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      zsum(jrof, jk) = pt(jrof, jk) + 1.0
    end do
  end do

  zsum(bnds%kidia:bnds%kfdia, 1) = 0.0

  !$loki small-kernels
  call leaf_kernel_deep(klon, klev, bnds, opts, phi, zsum)

  do jk = klev, 1, -1
    do jrof = bnds%kidia, bnds%kfdia
      phi(jrof, jk) = phi(jrof, jk) + zsum(jrof, jk)
    end do
  end do
end subroutine sub_kernel
""".strip()

FCODE_LEAF_KERNEL_DEEP = """
subroutine leaf_kernel_deep(klon, klev, bnds, opts, phi, zsum)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: phi(klon, klev)
  real, intent(in) :: zsum(klon, klev)
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      phi(jrof, jk) = phi(jrof, jk) + zsum(jrof, jk)
    end do
  end do
end subroutine leaf_kernel_deep
""".strip()

FCODE_CONDITIONAL_ACC_PROLOGUE_SUB_KERNEL = """
subroutine conditional_acc_prologue_sub_kernel(klon, klev, kst, kend, lvertfe, mode_intg, phi, phif, pt, pr, plnpr)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  logical, intent(in) :: lvertfe, mode_intg
  real, intent(inout) :: phi(klon, klev + 1)
  real, intent(out) :: phif(klon, klev)
  real, intent(in) :: pt(klon, klev)
  real, intent(in) :: pr(klon, klev)
  real, intent(in) :: plnpr(klon, klev)
  real :: zphi(klon, klev + 2)
  character(len=4) :: mode_flag
  integer :: jrof, jk

  !$acc data present(phi, phif, pt, pr, plnpr)
  !$acc enter data create(zphi)

  mode_flag = 'IBOT'
  if (mode_intg) mode_flag = 'INTG'

  if (lvertfe) then

    do jk = 1, klev
      do jrof = kst, kend
        zphi(jrof, jk + 1) = -pr(jrof, jk) * pt(jrof, jk) * plnpr(jrof, jk)
      end do
    end do

    zphi(kst:kend, 1) = 0.0
    zphi(kst:kend, klev + 2) = 0.0

    !$loki small-kernels
    call conditional_acc_prologue_leaf_kernel(klon, klev, kst, kend, zphi, phif, pedge=phi(:, klev + 1), mode_flag=mode_flag)

    do jk = klev, 1, -1
      do jrof = kst, kend
        phi(jrof, jk) = phi(jrof, jk + 1) + pr(jrof, jk) * pt(jrof, jk) * plnpr(jrof, jk)
      end do
    end do
  else
    do jk = klev, 1, -1
      do jrof = kst, kend
        phi(jrof, jk) = phi(jrof, jk + 1) + pr(jrof, jk) * pt(jrof, jk) * plnpr(jrof, jk)
        phif(jrof, jk) = phi(jrof, jk + 1) + pr(jrof, jk) * pt(jrof, jk)
      end do
    end do
  end if
  !$acc exit data delete(zphi)
  !$acc end data
end subroutine conditional_acc_prologue_sub_kernel
""".strip()


FCODE_CONDITIONAL_ACC_PROLOGUE_MID_KERNEL = """
subroutine conditional_acc_prologue_mid_kernel(klon, klev, kst, kend, lvertfe, mode_intg, phi, phif, pt, pr, plnpr)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  logical, intent(in) :: lvertfe, mode_intg
  real, intent(inout) :: phi(klon, klev + 1)
  real, intent(out) :: phif(klon, klev)
  real, intent(in) :: pt(klon, klev)
  real, intent(in) :: pr(klon, klev)
  real, intent(in) :: plnpr(klon, klev)

  !$loki small-kernels
  call conditional_acc_prologue_sub_kernel(klon, klev, kst, kend, lvertfe, mode_intg, phi, phif, pt, pr, plnpr)
end subroutine conditional_acc_prologue_mid_kernel
""".strip()


FCODE_CONDITIONAL_ACC_PROLOGUE_DRIVER = """
module conditional_acc_prologue_driver_mod
  implicit none
contains
  subroutine conditional_acc_prologue_driver(ngpblks, bnds, opts, lvertfe, mode_intg, phi, phif, pt, pr, plnpr)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    logical, intent(in) :: lvertfe, mode_intg
    real, intent(inout) :: phi(:,:,:)
    real, intent(out) :: phif(:,:,:)
    real, intent(in) :: pt(:,:,:)
    real, intent(in) :: pr(:,:,:)
    real, intent(in) :: plnpr(:,:,:)
    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call conditional_acc_prologue_mid_kernel(opts%klon, opts%kflevg, bnds%kidia, bnds%kfdia, lvertfe, mode_intg, &
        & phi(:,:,ibl), phif(:,:,ibl), pt(:,:,ibl), pr(:,:,ibl), plnpr(:,:,ibl))
    end do
  end subroutine conditional_acc_prologue_driver
end module conditional_acc_prologue_driver_mod
""".strip()


FCODE_CONDITIONAL_ACC_PROLOGUE_LEAF_KERNEL = """
subroutine conditional_acc_prologue_leaf_kernel(klon, klev, kst, kend, zphi, phif, pedge, mode_flag)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  real, intent(in) :: zphi(klon, klev + 2)
  real, intent(out) :: phif(klon, klev)
  real, intent(in) :: pedge(klon)
  character(len=*), intent(in) :: mode_flag
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = kst, kend
      phif(jrof, jk) = zphi(jrof, jk + 1) + pedge(jrof)
      if (mode_flag == 'INTG') phif(jrof, jk) = phif(jrof, jk) + 1.0
    end do
  end do
end subroutine conditional_acc_prologue_leaf_kernel
""".strip()

FCODE_EXPLICIT_ACC_DATA_DRIVER = """
module explicit_acc_data_driver_mod
  implicit none
contains
  subroutine explicit_acc_data_driver(ngpblks, bnds, opts, lvertfe, phi, phif, pt, pr, plnpr)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    logical, intent(in) :: lvertfe
    real, intent(inout) :: phi(:,:,:)
    real, intent(out) :: phif(:,:,:)
    real, intent(in) :: pt(:,:,:)
    real, intent(in) :: pr(:,:,:)
    real, intent(in) :: plnpr(:,:,:)
    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call explicit_acc_data_mid_kernel(opts%klon, opts%kflevg, bnds%kidia, bnds%kfdia, lvertfe, phi(:,:,ibl), phif(:,:,ibl), pt(:,:,ibl), pr(:,:,ibl), plnpr(:,:,ibl))
    end do
  end subroutine explicit_acc_data_driver
end module explicit_acc_data_driver_mod
""".strip()

FCODE_EXPLICIT_ACC_DATA_MID_KERNEL = """
subroutine explicit_acc_data_mid_kernel(klon, klev, kst, kend, lvertfe, phi, phif, pt, pr, plnpr)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  logical, intent(in) :: lvertfe
  real, intent(inout) :: phi(klon, klev + 1)
  real, intent(out) :: phif(klon, klev)
  real, intent(in) :: pt(klon, klev)
  real, intent(in) :: pr(klon, klev)
  real, intent(in) :: plnpr(klon, klev)

  !$loki small-kernels
  call explicit_acc_data_sub_kernel(klon, klev, kst, kend, lvertfe, phi, phif, pt, pr, plnpr)
end subroutine explicit_acc_data_mid_kernel
""".strip()

FCODE_EXPLICIT_ACC_DATA_SUB_KERNEL = """
subroutine explicit_acc_data_sub_kernel(klon, klev, kst, kend, lvertfe, phi, phif, pt, pr, plnpr)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  logical, intent(in) :: lvertfe
  real, intent(inout) :: phi(klon, klev + 1)
  real, intent(out) :: phif(klon, klev)
  real, intent(in) :: pt(klon, klev)
  real, intent(in) :: pr(klon, klev)
  real, intent(in) :: plnpr(klon, klev)
  real :: zphi(klon, 0:klev + 1)
  integer :: jrof, jk

  !$acc data present(phi, phif, pt, pr, plnpr)
  !$acc enter data create(zphi)
  if (lvertfe) then
    do jk = 1, klev
      do jrof = kst, kend
        zphi(jrof, jk) = -pr(jrof, jk) * pt(jrof, jk) * plnpr(jrof, jk)
      end do
    end do

    zphi(kst:kend, 0) = 0.0
    zphi(kst:kend, klev + 1) = 0.0

    !$loki small-kernels
    call explicit_acc_data_leaf_kernel(klon, klev, kst, kend, zphi, phif, pedge=phi(:, klev + 1))

    do jk = klev, 1, -1
      do jrof = kst, kend
        phi(jrof, jk) = phi(jrof, jk + 1) + pr(jrof, jk) * pt(jrof, jk) * plnpr(jrof, jk)
      end do
    end do
  else
    do jk = klev, 1, -1
      do jrof = kst, kend
        phi(jrof, jk) = phi(jrof, jk + 1) + pr(jrof, jk) * pt(jrof, jk) * plnpr(jrof, jk)
        phif(jrof, jk) = phi(jrof, jk + 1) + pr(jrof, jk) * pt(jrof, jk)
      end do
    end do
  end if
  !$acc exit data delete(zphi)
  !$acc end data
end subroutine explicit_acc_data_sub_kernel
""".strip()



FCODE_EXPLICIT_ACC_DATA_LEAF_KERNEL = """
subroutine explicit_acc_data_leaf_kernel(klon, klev, kst, kend, zphi, phif, pedge)
  implicit none
  integer, intent(in) :: klon, klev, kst, kend
  real, intent(in) :: zphi(klon, 0:klev + 1)
  real, intent(out) :: phif(klon, klev)
  real, intent(in) :: pedge(klon)
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = kst, kend
      phif(jrof, jk) = zphi(jrof, jk) + pedge(jrof)
    end do
  end do
end subroutine explicit_acc_data_leaf_kernel
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_thin_wrapper_no_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    Thin wrapper kernel: must NOT have a block loop.
    The block loop belongs inside the inner kernel, not in the wrapper.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_WRAPPER_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    wrapper_src = Sourcefile.from_source(
        FCODE_THIN_WRAPPER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    inner_src = Sourcefile.from_source(
        FCODE_INNER_COMPUTE, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, wrapper, inner = _build_and_apply_3level(
        driver_src, wrapper_src, inner_src, horizontal, block_dim,
        'wrapper_driver', 'thin_wrapper', 'inner_compute',
        'wrapper_driver_mod#wrapper_driver', '#thin_wrapper', '#inner_compute'
    )

    # Wrapper must NOT have a block loop
    assert not _has_block_loop(wrapper, block_dim), (
        f"Thin wrapper should NOT have a block loop.\n"
        f"Code:\n{fgen(wrapper)}"
    )

    # Wrapper must still call inner_compute
    calls = _find_calls_by_name(wrapper, 'inner_compute')
    assert calls, (
        f"Wrapper should still call inner_compute.\nCode:\n{fgen(wrapper)}"
    )

    # Inner kernel MUST have a block loop
    assert _has_block_loop(inner, block_dim), (
        f"Inner kernel should have a block-dimension loop.\n"
        f"Code:\n{fgen(inner)}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_block_section_propagates_through_intermediate_kernel(
        frontend, horizontal, block_dim, tmp_path):
    """
    A kernel with horizontal work around a ``!$loki small-kernels`` call
    must still propagate block-section processing to its leaf callee.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_DEEP_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_MID_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_SUB_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    leaf_src = Sourcefile.from_source(
        FCODE_LEAF_KERNEL_DEEP, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, _, sub, leaf = _build_and_apply_4level(
        driver_src, mid_src, sub_src, leaf_src, horizontal, block_dim,
        'deep_driver', 'mid_kernel', 'sub_kernel', 'leaf_kernel_deep',
        'deep_driver_mod#deep_driver', '#mid_kernel', '#sub_kernel', '#leaf_kernel_deep'
    )

    assert _has_block_loop(sub, block_dim), (
        f"Intermediate kernel should have a block-dimension loop.\nCode:\n{fgen(sub)}"
    )
    assert _has_block_loop(leaf, block_dim), (
        f"Leaf kernel should inherit block-section processing through the "
        f"intermediate kernel.\nCode:\n{fgen(leaf)}"
    )

    calls = _find_calls_by_name(sub, 'leaf_kernel_deep')
    assert calls, (
        f"Intermediate kernel should still call leaf_kernel_deep.\n"
        f"Code:\n{fgen(sub)}"
    )
    assert 'acc parallel loop gang' in fgen(leaf).lower(), (
        f"Leaf kernel should own a parallel block loop after propagation.\n"
        f"Code:\n{fgen(leaf)}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_real_pipeline_conditional_acc_prologue_outside_first_block_loop(
        frontend, horizontal, block_dim, tmp_path):
    """
    In the first conditional branch, ``!$acc enter data create`` and the
    mode-flag setup must stay outside
    the first inserted block loop.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_CONDITIONAL_ACC_PROLOGUE_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_CONDITIONAL_ACC_PROLOGUE_MID_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_CONDITIONAL_ACC_PROLOGUE_SUB_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    leaf_src = Sourcefile.from_source(
        FCODE_CONDITIONAL_ACC_PROLOGUE_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, _, sub, _ = _build_and_apply_4level_full_pipeline(
        driver_src, mid_src, sub_src, leaf_src, horizontal, block_dim,
        'conditional_acc_prologue_driver', 'conditional_acc_prologue_mid_kernel',
        'conditional_acc_prologue_sub_kernel', 'conditional_acc_prologue_leaf_kernel',
        'conditional_acc_prologue_driver_mod#conditional_acc_prologue_driver',
        '#conditional_acc_prologue_mid_kernel', '#conditional_acc_prologue_sub_kernel',
        '#conditional_acc_prologue_leaf_kernel'
    )

    block_loops = _get_block_loops(sub, block_dim)
    assert block_loops, f"Expected block loop in transformed sub-kernel.\nCode:\n{fgen(sub)}"
    first_loop = block_loops[0]

    pragmas = FindNodes(ir.Pragma).visit(first_loop.body)
    assignments = FindNodes(ir.Assignment).visit(first_loop.body)
    enter_data_inside = [
        p for p in pragmas
        if p.keyword.lower() == 'acc'
        and 'enter' in p.content.lower()
        and 'data' in p.content.lower()
        and 'create' in p.content.lower()
    ]
    mode_assigns_inside = [a for a in assignments if a.lhs == 'mode_flag']

    assert not enter_data_inside, (
        f"Conditional prologue reproducer: !$acc enter data create(...) should stay outside the first block loop.\n"
        f"Found inside first block loop: {[fgen(p) for p in enter_data_inside]}\n"
        f"First block loop:\n{fgen(first_loop)}\n\nFull sub-kernel:\n{fgen(sub)}"
    )
    assert not mode_assigns_inside, (
        f"Conditional prologue reproducer: mode-flag setup should stay outside the first block loop.\n"
        f"Found inside first block loop: {[fgen(a) for a in mode_assigns_inside]}\n"
        f"First block loop:\n{fgen(first_loop)}\n\nFull sub-kernel:\n{fgen(sub)}"
    )


# ===================================================================
# Group 5: Vector and stack propagation
# ===================================================================

FCODE_VEC_DRIVER = """
module vec_driver_mod
  implicit none
contains
  subroutine vec_driver(ngpblks, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "vec_kernel.intfb.h"
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
      call vec_kernel(opts%klon, opts%kflevg, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine vec_driver
end module vec_driver_mod
""".strip()

# Vector kernel: simple kernel called with !$loki small-kernels.
# It should get !$acc routine seq or vector annotation, but not its own
# pool allocator.
FCODE_VEC_KERNEL = """
subroutine vec_kernel(klon, klev, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: t(klon, klev)
  real, intent(inout) :: q(klon, klev)
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      t(jrof, jk) = t(jrof, jk) * 2.0
      q(jrof, jk) = t(jrof, jk) + 1.0
    end do
  end do
end subroutine vec_kernel
""".strip()

# Stack-propagation pattern: caller -> callee where callee has
# local temporaries that get pool-allocated.
FCODE_STACK_CALLER = """
subroutine stack_caller(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  #include "stack_callee.intfb.h"
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(inout) :: pfield(klon, klev)
  integer :: jrof, jk

  do jk = 1, klev
    do jrof = bnds%kidia, bnds%kfdia
      pfield(jrof, jk) = pfield(jrof, jk) + 1.0
    end do
  end do

  !$loki small-kernels
  call stack_callee(klon, klev, bnds, opts, pfield)
end subroutine stack_caller
""".strip()

FCODE_STACK_DRIVER = """
module stack_driver_mod
  implicit none
contains
  subroutine stack_driver(ngpblks, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "stack_caller.intfb.h"
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
      call stack_caller(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
  end subroutine stack_driver
end module stack_driver_mod
""".strip()

FCODE_STACK_CALLEE = """
subroutine stack_callee(klon, klev, bnds, opts, pfield)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
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
end subroutine stack_callee
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_vector_kernel_no_pool_allocator(frontend, horizontal, block_dim, tmp_path):
    """
    Simple leaf kernel: a kernel without sub-calls
    should NOT get ISTSZ/ZSTACK at the kernel level.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_VEC_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kernel_src = Sourcefile.from_source(
        FCODE_VEC_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, vec_kernel = _build_and_apply_2level(
        driver_src, kernel_src, horizontal, block_dim,
        'vec_driver', 'vec_kernel',
        'vec_driver_mod#vec_driver', '#vec_kernel'
    )

    var_names = _routine_var_names(vec_kernel)
    assert 'istsz' not in var_names, (
        f"Vector kernel should NOT have ISTSZ.\nVars: {sorted(var_names)}\n"
        f"Code:\n{fgen(vec_kernel)}"
    )
    assert 'zstack' not in var_names, (
        f"Vector kernel should NOT have ZSTACK.\nVars: {sorted(var_names)}\n"
        f"Code:\n{fgen(vec_kernel)}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_vector_kernel_acc_routine(frontend, horizontal, block_dim, tmp_path):
    """
    Leaf kernel called with ``!$loki small-kernels``: should get
    ``!$acc routine seq`` or ``!$acc routine vector`` annotation.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_VEC_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kernel_src = Sourcefile.from_source(
        FCODE_VEC_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, vec_kernel = _build_and_apply_2level(
        driver_src, kernel_src, horizontal, block_dim,
        'vec_driver', 'vec_kernel',
        'vec_driver_mod#vec_driver', '#vec_kernel'
    )

    code_lower = fgen(vec_kernel).lower()
    has_acc_routine = ('acc routine' in code_lower or
                       'acc parallel loop gang' in code_lower)
    assert has_acc_routine, (
        f"Kernel should have !$acc routine or !$acc parallel loop gang.\n"
        f"Code:\n{fgen(vec_kernel)}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_stack_arg_propagated_to_callee(frontend, horizontal, block_dim, tmp_path):
    """
    Stack arg propagation: callee kernel with local
    temporaries must get YDSTACK_L in its argument list, and the
    caller's call must pass it.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_STACK_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    caller_src = Sourcefile.from_source(
        FCODE_STACK_CALLER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    callee_src = Sourcefile.from_source(
        FCODE_STACK_CALLEE, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, caller, callee = _build_and_apply_3level(
        driver_src, caller_src, callee_src, horizontal, block_dim,
        'stack_driver', 'stack_caller', 'stack_callee',
        'stack_driver_mod#stack_driver', '#stack_caller', '#stack_callee'
    )

    # Callee should have stack arg (YDSTACK_L)
    callee_arg_names = [str(a).lower() for a in callee.arguments]
    has_stack = any('stack' in a for a in callee_arg_names)
    assert has_stack, (
        f"Callee should have YDSTACK_L in args.\n"
        f"Args: {callee_arg_names}\nCode:\n{fgen(callee)}"
    )

    # Caller's call to callee must pass stack arg
    calls = _find_calls_by_name(caller, 'stack_callee')
    assert calls, f"Caller should call stack_callee.\nCode:\n{fgen(caller)}"
    for call in calls:
        kwarg_names = [kw[0].lower() for kw in (call.kwarguments or ())]
        arg_strs = [str(a).lower() for a in call.arguments]
        assert any('stack' in kw for kw in kwarg_names) or \
               any('stack' in a for a in arg_strs), (
            f"Call to stack_callee must pass YDSTACK_L.\n"
            f"Kwargs: {kwarg_names}\nArgs: {arg_strs}\n"
            f"Call: {fgen(call)}\nCode:\n{fgen(caller)}"
        )


# ===================================================================
# Group 6: Multi-driver-loop
# ===================================================================

FCODE_MULTI_DRIVER = """
module multi_driver_mod
  implicit none
contains
  subroutine multi_driver(ngpblks, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "kernel_a.intfb.h"
    #include "kernel_b.intfb.h"
    #include "kernel_c.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(:,:,:)
    integer :: ibl

    ! Loop 1: kernel_a WITH !$loki small-kernels
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)
      !$loki small-kernels
      call kernel_a(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do

    ! Loop 2: kernel_b WITH !$loki small-kernels
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)
      !$loki small-kernels
      call kernel_b(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do

    ! Loop 3: kernel_c WITH !$loki small-kernels
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)
      !$loki small-kernels
      call kernel_c(opts%klon, opts%kflevg, bnds, opts, pfield(:,:,ibl))
    end do
  end subroutine multi_driver
end module multi_driver_mod
""".strip()

FCODE_KERNEL_A = """
module kernel_a_mod
  implicit none
contains
  subroutine kernel_a(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in) :: klon, klev
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
  end subroutine kernel_a
end module kernel_a_mod
""".strip()

FCODE_KERNEL_B = """
module kernel_b_mod
  implicit none
contains
  subroutine kernel_b(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in) :: klon, klev
    type(bnds_type), intent(in) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(inout) :: pfield(klon, klev)
    integer :: jrof, jk
    do jk = 1, klev
      do jrof = bnds%kidia, bnds%kfdia
        pfield(jrof, jk) = pfield(jrof, jk) + 1.0
      end do
    end do
  end subroutine kernel_b
end module kernel_b_mod
""".strip()

FCODE_KERNEL_C = """
module kernel_c_mod
  implicit none
contains
  subroutine kernel_c(klon, klev, bnds, opts, pfield)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    integer, intent(in) :: klon, klev
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
  end subroutine kernel_c
end module kernel_c_mod
""".strip()


def _build_and_apply_multi_driver(
        driver_source, kernel_sources, horizontal, block_dim,  # pylint: disable=unused-argument
        driver_name, kernel_names, driver_item_name, kernel_item_names,
        kernel_source_objs):
    """
    Build multi-kernel call tree. Only the driver gets 'driver' role;
    kernels get 'kernel' role. All go through the pipeline.
    """
    pipeline = SCCSmallKernelsPipeline(
        horizontal=horizontal, block_dim=block_dim, directive='openacc'
    )
    driver = driver_source[driver_name]
    kernels = [src[name] for src, name in zip(kernel_source_objs, kernel_names)]
    for k in kernels:
        driver.enrich(k)

    d_item = ProcedureItem(name=driver_item_name, source=driver_source)
    k_items = [ProcedureItem(name=iname, source=src)
               for iname, src in zip(kernel_item_names, kernel_source_objs)]
    sgraph = SGraph.from_dict({d_item: k_items})

    items = [(driver, 'driver', d_item, list(kernel_names))]
    for k, ki in zip(kernels, k_items):
        items.append((k, 'kernel', ki, []))
    _apply_pipeline(pipeline, items, sgraph)
    return driver, kernels


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
@pytest.mark.xfail(reason=(
    'Multi-driver-loop pattern requires mixing SK and standard SCC pipelines. '
    'All-SK multi-loop drivers hoist block loops into kernels, removing '
    'ISTSZ/ZSTACK from the driver.'
))
def test_multi_driver_loop_max_istsz(frontend, horizontal, block_dim, tmp_path):
    """
    Driver with multiple block loops:
    ISTSZ must be aggregated via MAX across all driver loops, and
    ZSTACK must be allocated.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_MULTI_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    ka_src = Sourcefile.from_source(
        FCODE_KERNEL_A, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kb_src = Sourcefile.from_source(
        FCODE_KERNEL_B, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kc_src = Sourcefile.from_source(
        FCODE_KERNEL_C, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    driver, _ = _build_and_apply_multi_driver(
        driver_src,
        [ka_src, kb_src, kc_src],
        horizontal, block_dim,
        'multi_driver',
        ['kernel_a', 'kernel_b', 'kernel_c'],
        'multi_driver_mod#multi_driver',
        ['kernel_a_mod#kernel_a', 'kernel_b_mod#kernel_b', 'kernel_c_mod#kernel_c'],
        [ka_src, kb_src, kc_src]
    )

    var_names = _routine_var_names(driver)
    assert 'istsz' in var_names, (
        f"Driver must have ISTSZ.\nVars: {sorted(var_names)}\n"
        f"Code:\n{fgen(driver)}"
    )

    allocs = FindNodes(ir.Allocation).visit(driver.body)
    assert any('zstack' in str(a).lower() for a in allocs), (
        f"Driver must ALLOCATE ZSTACK.\n"
        f"Allocations: {[fgen(a) for a in allocs]}\n"
        f"Code:\n{fgen(driver)}"
    )

    # ISTSZ should use MAX (aggregating across loops)
    assigns = FindNodes(ir.Assignment).visit(driver.body)
    istsz_assigns = [a for a in assigns if str(a.lhs).lower() == 'istsz']
    assert istsz_assigns, f"Driver must have ISTSZ assignment.\nCode:\n{fgen(driver)}"

    istsz_rhs = str(istsz_assigns[0].rhs).lower()
    assert 'max' in istsz_rhs, (
        f"ISTSZ should use MAX to aggregate stack sizes.\n"
        f"ISTSZ RHS: {istsz_rhs}\nCode:\n{fgen(driver)}"
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
@pytest.mark.xfail(reason=(
    'Multi-driver-loop pattern requires mixing SK and standard SCC pipelines.'
))
def test_multi_driver_loop_loc_assigns(frontend, horizontal, block_dim, tmp_path):
    """
    Driver with multiple block loops: each loop with stack-needing
    calls must have ``YLSTACK_L = LOC(ZSTACK(...))`` inside.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_MULTI_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    ka_src = Sourcefile.from_source(
        FCODE_KERNEL_A, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kb_src = Sourcefile.from_source(
        FCODE_KERNEL_B, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kc_src = Sourcefile.from_source(
        FCODE_KERNEL_C, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    driver, _ = _build_and_apply_multi_driver(
        driver_src,
        [ka_src, kb_src, kc_src],
        horizontal, block_dim,
        'multi_driver',
        ['kernel_a', 'kernel_b', 'kernel_c'],
        'multi_driver_mod#multi_driver',
        ['kernel_a_mod#kernel_a', 'kernel_b_mod#kernel_b', 'kernel_c_mod#kernel_c'],
        [ka_src, kb_src, kc_src]
    )

    # Find LOC(ZSTACK) assignments inside driver loops
    driver_loops = FindNodes(ir.Loop).visit(driver.body)
    loc_count = 0
    for dl in driver_loops:
        loop_assigns = FindNodes(ir.Assignment).visit(dl.body)
        loc_assigns = [a for a in loop_assigns
                       if 'ylstack_l' in str(a.lhs).lower()
                       and 'loc' in str(a.rhs).lower()
                       and 'zstack' in str(a.rhs).lower()]
        loc_count += len(loc_assigns)

    # At least loops 1 and 3 need LOC assigns (kernel_a and kernel_c
    # have pool-allocated temporaries)
    assert loc_count >= 2, (
        f"At least 2 driver loops must have YLSTACK_L = LOC(ZSTACK(...)).\n"
        f"Found {loc_count}.\nCode:\n{fgen(driver)}"
    )


# ===================================================================
# Group 7: Edge cases
# ===================================================================

FCODE_PRIV_DRIVER = """
module priv_driver_mod
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
contains
  subroutine priv_driver(ydgeometry, bnds, opts, t, q)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "priv_kernel.intfb.h"
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
      call priv_kernel(ydgeometry%yrdim%nproma, ydgeometry%yrdim%nflevg, &
        & ydgeometry%yrdim%ngpblks, bnds, t(:,:,ibl), q(:,:,ibl))
    end do
  end subroutine priv_driver
end module priv_driver_mod
""".strip()

FCODE_PRIV_KERNEL = """
subroutine priv_kernel(klon, klev, ngpblks, bnds, t, q)
  use bnds_type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: klon, klev, ngpblks
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
end subroutine priv_kernel
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_derived_type_args_not_privatised(frontend, horizontal, block_dim, tmp_path):
    """
    Derived-type argument components (e.g. YDGEOMETRY%YRDIM) must NOT
    appear in the ``private(...)`` clause of ``!$acc parallel loop gang``.
    Only truly local variables should be privatised.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_PRIV_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kernel_src = Sourcefile.from_source(
        FCODE_PRIV_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    driver, _ = _build_and_apply_2level(
        driver_src, kernel_src, horizontal, block_dim,
        'priv_driver', 'priv_kernel',
        'priv_driver_mod#priv_driver', '#priv_kernel'
    )

    code = fgen(driver)
    private_matches = re.findall(r'private\(([^)]+)\)', code, re.IGNORECASE)

    for match in private_matches:
        privates = [p.strip().lower() for p in match.split(',')]
        for priv in privates:
            assert 'ydgeometry' not in priv, (
                f"YDGEOMETRY component '{priv}' should NOT be in\n"
                f"private() clause.\nprivate({match})\nCode:\n{code}"
            )


FCODE_ACC_EXIT_DRIVER = """
module acc_exit_driver_mod
  implicit none
contains
  subroutine acc_exit_driver(ngpblks, bnds, opts, pd, pt, psp)
    use bnds_type_mod, only: bnds_type
    use opts_type_mod, only: opts_type
    implicit none
    #include "acc_exit_kernel.intfb.h"
    integer, intent(in) :: ngpblks
    type(bnds_type), intent(inout) :: bnds
    type(opts_type), intent(in) :: opts
    real, intent(out) :: pd(:,:,:)
    real, intent(in) :: pt(:,:,:)
    real, intent(in) :: psp(:,:)
    integer :: ibl

    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call acc_exit_kernel(opts%klon, opts%kflevg, bnds, opts, pd(:,:,ibl), pt(:,:,ibl), psp(:,ibl))
    end do
  end subroutine acc_exit_driver
end module acc_exit_driver_mod
""".strip()

FCODE_ACC_EXIT_KERNEL = """
subroutine acc_exit_kernel(klon, klev, bnds, opts, pd, pt, psp)
  use bnds_type_mod, only: bnds_type
  use opts_type_mod, only: opts_type
  implicit none
  integer, intent(in) :: klon, klev
  type(bnds_type), intent(in) :: bnds
  type(opts_type), intent(in) :: opts
  real, intent(out) :: pd(klon, klev)
  real, intent(in) :: pt(klon, klev)
  real, intent(in) :: psp(klon)
  real :: ztmp(klon, 0:klev+1)
  real :: zout(klon, klev)
  integer :: jrof, jk

  !$acc enter data create(ztmp, zout)

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
end subroutine acc_exit_kernel
""".strip()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
@pytest.mark.xfail(reason='Minimal SCC helper still sweeps explicit raw ACC enter/exit pragmas into block loops')
def test_acc_exit_data_outside_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    ``!$acc enter/exit data`` directives must remain outside the block
    loop, not be swept inside by extract_block_sections.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_ACC_EXIT_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    kernel_src = Sourcefile.from_source(
        FCODE_ACC_EXIT_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, kernel = _build_and_apply_2level(
        driver_src, kernel_src, horizontal, block_dim,
        'acc_exit_driver', 'acc_exit_kernel',
        'acc_exit_driver_mod#acc_exit_driver', '#acc_exit_kernel'
    )

    # Kernel should have a block loop
    block_loops = _get_block_loops(kernel, block_dim)
    assert block_loops, (
        f"Kernel should have a block loop.\nCode:\n{fgen(kernel)}"
    )

    # Check that !$acc enter/exit data are NOT inside the block loop
    for bl in block_loops:
        block_pragmas = FindNodes(ir.Pragma).visit(bl.body)
        enter_inside = [p for p in block_pragmas
                        if p.keyword.lower() == 'acc'
                        and 'enter' in p.content.lower()
                        and 'data' in p.content.lower()
                        and 'create' in p.content.lower()]
        exit_inside = [p for p in block_pragmas
                       if p.keyword.lower() == 'acc'
                       and 'exit' in p.content.lower()
                       and 'data' in p.content.lower()
                       and 'delete' in p.content.lower()]

        assert not enter_inside, (
            f"!$acc enter data must NOT be inside block loop.\n"
            f"Found: {[fgen(p) for p in enter_inside]}\nCode:\n{fgen(kernel)}"
        )
        assert not exit_inside, (
            f"!$acc exit data must NOT be inside block loop.\n"
            f"Found: {[fgen(p) for p in exit_inside]}\nCode:\n{fgen(kernel)}"
        )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_real_pipeline_explicit_acc_data_regions_not_duplicated(
        frontend, horizontal, block_dim, tmp_path):
    """
    The real scc-stack pipeline must not duplicate explicit ACC data regions.

    This reproduces the explicit-data-managed shape where a kernel already contains
    explicit ``!$acc data`` and ``!$acc enter/exit data`` pragmas before SCC
    annotation and pragma-model lowering are applied.
    """
    defs = _parse_modules(frontend, tmp_path)
    driver_src = Sourcefile.from_source(
        FCODE_EXPLICIT_ACC_DATA_DRIVER, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    mid_src = Sourcefile.from_source(
        FCODE_EXPLICIT_ACC_DATA_MID_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    sub_src = Sourcefile.from_source(
        FCODE_EXPLICIT_ACC_DATA_SUB_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )
    leaf_src = Sourcefile.from_source(
        FCODE_EXPLICIT_ACC_DATA_LEAF_KERNEL, frontend=frontend, definitions=defs, xmods=[tmp_path]
    )

    _, _, sub, _ = _build_and_apply_4level_full_pipeline(
        driver_src, mid_src, sub_src, leaf_src,
        horizontal, block_dim,
        'explicit_acc_data_driver', 'explicit_acc_data_mid_kernel',
        'explicit_acc_data_sub_kernel', 'explicit_acc_data_leaf_kernel',
        'explicit_acc_data_driver_mod#explicit_acc_data_driver', '#explicit_acc_data_mid_kernel',
        '#explicit_acc_data_sub_kernel', '#explicit_acc_data_leaf_kernel'
    )

    acc_pragmas = [p for p in FindNodes(ir.Pragma).visit(sub.body) if p.keyword.lower() == 'acc']
    acc_data = [p for p in acc_pragmas if p.content.lower().startswith('data ')]
    acc_end_data = [p for p in acc_pragmas if p.content.lower() == 'end data']
    acc_enter_data = [p for p in acc_pragmas if p.content.lower().startswith('enter data')]
    acc_exit_data = [p for p in acc_pragmas if p.content.lower().startswith('exit data')]

    assert len(acc_data) == 1, f"Expected exactly one ACC data region.\nCode:\n{fgen(sub)}"
    assert len(acc_end_data) == 1, f"Expected exactly one ACC end data pragma.\nCode:\n{fgen(sub)}"
    assert len(acc_enter_data) == 1, f"Expected exactly one ACC enter data pragma.\nCode:\n{fgen(sub)}"
    assert len(acc_exit_data) == 1, f"Expected exactly one ACC exit data pragma.\nCode:\n{fgen(sub)}"
