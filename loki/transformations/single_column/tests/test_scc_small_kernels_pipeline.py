# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

"""
End-to-end tests for ``SCCSmallKernelsPipeline``.

Each test runs the full pipeline on a small Fortran call tree that
mimics production-style transformation shapes and verifies the
structural properties of the transformed IR.
"""

import pytest

from loki import Dimension, Module, Sourcefile
from loki.batch import Pipeline, ProcedureItem, SGraph
from loki.frontend import OMNI, available_frontends
from loki.ir import nodes as ir, FindNodes, get_pragma_parameters, pragmas_attached
from loki.tools import CaseInsensitiveDict
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

def _sk_pipeline(horizontal, block_dim):
    """Build the small-kernels pipeline with OpenACC directives."""
    return SCCSmallKernelsPipeline(horizontal=horizontal, block_dim=block_dim, directive='openacc')


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


def _apply_pipeline(pipeline, tmp_path, frontend, call_tree, fcodes):
    """
    Parse the given sources, enrich them along :data:`call_tree` and apply
    :data:`pipeline` in call-tree order (or reversed for transformations
    with ``reverse_traversal``), with the first routine as the driver.

    Parameters
    ----------
    call_tree : dict
        Map of routine names to the names of their callees.
    fcodes : dict
        Map of routine names to their (module or subroutine) source.

    Returns
    -------
    dict
        The transformed routines, keyed by name.
    """
    defs = [
        Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
        for fcode in (FCODE_BNDS_TYPE_MOD, FCODE_OPTS_TYPE_MOD)
    ]
    sources = {
        name: Sourcefile.from_source(fcode, frontend=frontend, definitions=defs, xmods=[tmp_path])
        for name, fcode in fcodes.items()
    }
    routines = {name: source[name] for name, source in sources.items()}
    items = {
        name: ProcedureItem(
            name=f'{source.modules[0].name}#{name}' if source.modules else f'#{name}', source=source
        )
        for name, source in sources.items()
    }

    for name, callees in call_tree.items():
        routines[name].enrich(tuple(routines[c] for c in callees))
    sgraph = SGraph.from_dict({items[name]: [items[c] for c in callees] for name, callees in call_tree.items()})

    order = [
        (routines[name], 'driver' if idx == 0 else 'kernel', items[name], list(callees))
        for idx, (name, callees) in enumerate(call_tree.items())
    ]
    for transformation in pipeline.transformations:
        for routine, role, item, targets in (order[::-1] if transformation.reverse_traversal else order):
            transformation.apply(routine, role=role, item=item, targets=targets, sub_sgraph=sgraph)
    return routines


def _block_loops(routine, block_dim):
    """Return all loops over a block-dimension index in :data:`routine`."""
    return [loop for loop in FindNodes(ir.Loop).visit(routine.body) if loop.variable in block_dim.indices]


def _gang_loop_parameters(routine, block_dim):
    """
    Return the parameters of the ``!$acc parallel loop gang`` pragma of
    each block loop in :data:`routine`, or `None` for loops without it.
    """
    with pragmas_attached(routine, ir.Loop):
        return [
            get_pragma_parameters(loop.pragma, starts_with='parallel loop gang', only_loki_pragmas=False)
            if loop.pragma and loop.pragma[0].content.startswith('parallel loop gang') else None
            for loop in _block_loops(routine, block_dim)
        ]


def _acc_pragma_contents(routine):
    """Return the contents of all ``!$acc`` pragmas in the spec and body of :data:`routine`."""
    return [p.content for p in FindNodes(ir.Pragma).visit(routine.ir) if p.keyword == 'acc']


# ===================================================================
# Driver -> nested driver -> leaf kernel
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
def test_sk_pipeline_nested_driver(frontend, horizontal, block_dim, tmp_path):
    """
    Test the pipeline on a driver calling a nested driver with horizontal
    work and a small-kernels call to a leaf kernel with a temporary.

    The block loop is removed from the top-level driver, together with the
    pool allocator infrastructure, and re-created as offloaded block loops
    in the nested driver and the leaf kernel, using local copies of the
    bounds object.
    """
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={'top_driver': ('nested_driver',), 'nested_driver': ('leaf_kernel',), 'leaf_kernel': ()},
        fcodes={
            'top_driver': FCODE_TOP_DRIVER, 'nested_driver': FCODE_NESTED_DRIVER,
            'leaf_kernel': FCODE_LEAF_KERNEL
        }
    )
    driver, nested, leaf = routines['top_driver'], routines['nested_driver'], routines['leaf_kernel']

    # The top-level driver has no block loop and no pool allocator infrastructure
    assert not _block_loops(driver, block_dim)
    assert 'istsz' not in driver.variable_map
    assert 'zstack' not in driver.variable_map
    assert not FindNodes(ir.Allocation).visit(driver.body)
    assert not FindNodes(ir.Deallocation).visit(driver.body)

    # The nested driver and leaf kernel get offloaded block loops, with the
    # host-initialised local copy of the bounds object as firstprivate
    for routine in (nested, leaf):
        gang_params = _gang_loop_parameters(routine, block_dim)
        assert gang_params and all(params is not None for params in gang_params)
        assert all(params['firstprivate'] == 'local_bnds' for params in gang_params)
        assert 'local_bnds' in routine.variable_map

    # The nested driver receives or forwards the stack, and passes the bounds object to the leaf kernel
    assert 'ydstack_l' in nested.variable_map or 'ylstack_l' in nested.variable_map
    calls = [call for call in FindNodes(ir.CallStatement).visit(nested.body) if call.name == 'leaf_kernel']
    assert len(calls) == 1
    assert 'bnds' in calls[0].arguments

    # The leaf kernel temporary is created on the device, and the leaf kernel
    # owns its parallel regions, so it is not a device routine
    acc_pragmas = _acc_pragma_contents(leaf)
    assert 'enter data create(ztmp)' in acc_pragmas
    assert not any(content.startswith('routine') for content in acc_pragmas)


# ===================================================================
# Thin wrapper and intermediate kernels
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

    !$loki data
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call conditional_acc_prologue_mid_kernel(opts%klon, opts%kflevg, bnds%kidia, bnds%kfdia, lvertfe, mode_intg, &
        & phi(:,:,ibl), phif(:,:,ibl), pt(:,:,ibl), pr(:,:,ibl), plnpr(:,:,ibl))
    end do
    !$loki end data
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

    !$loki data
    do ibl = 1, ngpblks
      bnds%kbl = ibl
      bnds%kstglo = 1 + (ibl - 1) * opts%klon
      bnds%kidia = 1
      bnds%kfdia = min(opts%klon, opts%kgpcomp - bnds%kstglo + 1)

      !$loki small-kernels
      call explicit_acc_data_mid_kernel(opts%klon, opts%kflevg, bnds%kidia, bnds%kfdia, lvertfe, phi(:,:,ibl), phif(:,:,ibl), pt(:,:,ibl), pr(:,:,ibl), plnpr(:,:,ibl))
    end do
    !$loki end data
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
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={'wrapper_driver': ('thin_wrapper',), 'thin_wrapper': ('inner_compute',), 'inner_compute': ()},
        fcodes={
            'wrapper_driver': FCODE_WRAPPER_DRIVER, 'thin_wrapper': FCODE_THIN_WRAPPER,
            'inner_compute': FCODE_INNER_COMPUTE
        }
    )
    wrapper, inner = routines['thin_wrapper'], routines['inner_compute']

    assert not _block_loops(wrapper, block_dim)
    assert [call.name for call in FindNodes(ir.CallStatement).visit(wrapper.body)] == ['inner_compute']
    assert _block_loops(inner, block_dim)


@pytest.mark.parametrize('frontend', available_frontends())
def test_block_section_propagates_through_intermediate_kernel(frontend, horizontal, block_dim, tmp_path):
    """
    A kernel with horizontal work around a ``!$loki small-kernels`` call
    must still propagate block-section processing to its leaf callee.
    """
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={
            'deep_driver': ('mid_kernel',), 'mid_kernel': ('sub_kernel',),
            'sub_kernel': ('leaf_kernel_deep',), 'leaf_kernel_deep': ()
        },
        fcodes={
            'deep_driver': FCODE_DEEP_DRIVER, 'mid_kernel': FCODE_MID_KERNEL,
            'sub_kernel': FCODE_SUB_KERNEL, 'leaf_kernel_deep': FCODE_LEAF_KERNEL_DEEP
        }
    )
    sub, leaf = routines['sub_kernel'], routines['leaf_kernel_deep']

    assert _block_loops(sub, block_dim)
    assert 'leaf_kernel_deep' in [call.name for call in FindNodes(ir.CallStatement).visit(sub.body)]
    gang_params = _gang_loop_parameters(leaf, block_dim)
    assert gang_params and all(params is not None for params in gang_params)


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_real_pipeline_conditional_acc_prologue_outside_first_block_loop(
        frontend, horizontal, block_dim, tmp_path):
    """
    In the first conditional branch, ``!$acc enter data create`` and the
    mode-flag setup must stay outside the first inserted block loop.
    """
    routines = _apply_pipeline(
        _build_scc_stack_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={
            'conditional_acc_prologue_driver': ('conditional_acc_prologue_mid_kernel',),
            'conditional_acc_prologue_mid_kernel': ('conditional_acc_prologue_sub_kernel',),
            'conditional_acc_prologue_sub_kernel': ('conditional_acc_prologue_leaf_kernel',),
            'conditional_acc_prologue_leaf_kernel': (),
        },
        fcodes={
            'conditional_acc_prologue_driver': FCODE_CONDITIONAL_ACC_PROLOGUE_DRIVER,
            'conditional_acc_prologue_mid_kernel': FCODE_CONDITIONAL_ACC_PROLOGUE_MID_KERNEL,
            'conditional_acc_prologue_sub_kernel': FCODE_CONDITIONAL_ACC_PROLOGUE_SUB_KERNEL,
            'conditional_acc_prologue_leaf_kernel': FCODE_CONDITIONAL_ACC_PROLOGUE_LEAF_KERNEL,
        }
    )
    sub = routines['conditional_acc_prologue_sub_kernel']

    block_loops = _block_loops(sub, block_dim)
    assert block_loops
    first_loop = block_loops[0]

    assert not [
        p for p in FindNodes(ir.Pragma).visit(first_loop.body)
        if p.keyword == 'acc' and p.content.startswith('enter data create')
    ]
    assert not [a for a in FindNodes(ir.Assignment).visit(first_loop.body) if a.lhs == 'mode_flag']


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_real_pipeline_explicit_acc_data_regions_not_duplicated(frontend, horizontal, block_dim, tmp_path):
    """
    The real scc-stack pipeline must not duplicate explicit ACC data regions.

    This reproduces the explicit-data-managed shape where a kernel already contains
    explicit ``!$acc data`` and ``!$acc enter/exit data`` pragmas before SCC
    annotation and pragma-model lowering are applied.
    """
    routines = _apply_pipeline(
        _build_scc_stack_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={
            'explicit_acc_data_driver': ('explicit_acc_data_mid_kernel',),
            'explicit_acc_data_mid_kernel': ('explicit_acc_data_sub_kernel',),
            'explicit_acc_data_sub_kernel': ('explicit_acc_data_leaf_kernel',),
            'explicit_acc_data_leaf_kernel': (),
        },
        fcodes={
            'explicit_acc_data_driver': FCODE_EXPLICIT_ACC_DATA_DRIVER,
            'explicit_acc_data_mid_kernel': FCODE_EXPLICIT_ACC_DATA_MID_KERNEL,
            'explicit_acc_data_sub_kernel': FCODE_EXPLICIT_ACC_DATA_SUB_KERNEL,
            'explicit_acc_data_leaf_kernel': FCODE_EXPLICIT_ACC_DATA_LEAF_KERNEL,
        }
    )
    acc_pragmas = [p.content.lower() for p in FindNodes(ir.Pragma).visit(routines['explicit_acc_data_sub_kernel'].body)
                   if p.keyword.lower() == 'acc']

    assert len([p for p in acc_pragmas if p.startswith('data ')]) == 1
    assert len([p for p in acc_pragmas if p == 'end data']) == 1
    assert len([p for p in acc_pragmas if p.startswith('enter data')]) == 1
    assert len([p for p in acc_pragmas if p.startswith('exit data')]) == 1


# ===================================================================
# Vector kernels and stack propagation
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
def test_vector_kernel(frontend, horizontal, block_dim, tmp_path):
    """
    A simple leaf kernel called with ``!$loki small-kernels`` gets
    OpenACC annotations, but not its own pool allocator.
    """
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={'vec_driver': ('vec_kernel',), 'vec_kernel': ()},
        fcodes={'vec_driver': FCODE_VEC_DRIVER, 'vec_kernel': FCODE_VEC_KERNEL}
    )
    vec_kernel = routines['vec_kernel']

    assert 'istsz' not in vec_kernel.variable_map
    assert 'zstack' not in vec_kernel.variable_map
    assert any(
        content.startswith(('routine', 'parallel loop gang')) for content in _acc_pragma_contents(vec_kernel)
    )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
def test_stack_arg_propagated_to_callee(frontend, horizontal, block_dim, tmp_path):
    """
    Stack arg propagation: callee kernel with local temporaries must get
    YDSTACK_L in its argument list, and the caller's call must pass it.
    """
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={'stack_driver': ('stack_caller',), 'stack_caller': ('stack_callee',), 'stack_callee': ()},
        fcodes={
            'stack_driver': FCODE_STACK_DRIVER, 'stack_caller': FCODE_STACK_CALLER,
            'stack_callee': FCODE_STACK_CALLEE
        }
    )
    caller, callee = routines['stack_caller'], routines['stack_callee']

    assert 'ydstack_l' in callee.arguments
    calls = [call for call in FindNodes(ir.CallStatement).visit(caller.body) if call.name == 'stack_callee']
    assert len(calls) == 1
    assert 'ydstack_l' in CaseInsensitiveDict(calls[0].kwarguments)


# ===================================================================
# Multiple driver loops
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


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
@pytest.mark.xfail(reason=(
    'Multi-driver-loop pattern requires mixing SK and standard SCC pipelines. '
    'All-SK multi-loop drivers hoist block loops into kernels, removing '
    'ISTSZ/ZSTACK from the driver.'
))
def test_multi_driver_loop_pool_allocator(frontend, horizontal, block_dim, tmp_path):
    """
    Driver with multiple block loops: ISTSZ must be aggregated via MAX
    across all driver loops, ZSTACK must be allocated, and each loop with
    stack-needing calls must have ``YLSTACK_L = LOC(ZSTACK(...))`` inside.
    """
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={
            'multi_driver': ('kernel_a', 'kernel_b', 'kernel_c'),
            'kernel_a': (), 'kernel_b': (), 'kernel_c': ()
        },
        fcodes={
            'multi_driver': FCODE_MULTI_DRIVER, 'kernel_a': FCODE_KERNEL_A,
            'kernel_b': FCODE_KERNEL_B, 'kernel_c': FCODE_KERNEL_C
        }
    )
    driver = routines['multi_driver']

    # ISTSZ is aggregated with MAX and ZSTACK is allocated
    assert 'istsz' in driver.variable_map
    assert any(var == 'zstack' for alloc in FindNodes(ir.Allocation).visit(driver.body) for var in alloc.variables)
    istsz_assigns = [a for a in FindNodes(ir.Assignment).visit(driver.body) if a.lhs == 'istsz']
    assert istsz_assigns
    assert 'max' in str(istsz_assigns[0].rhs).lower()

    # At least the loops calling kernel_a and kernel_c, which have pool-allocated
    # temporaries, set the stack pointer
    loc_assigns = [
        a for loop in FindNodes(ir.Loop).visit(driver.body) for a in FindNodes(ir.Assignment).visit(loop.body)
        if a.lhs == 'ylstack_l' and 'loc(zstack' in str(a.rhs).lower()
    ]
    assert len(loc_assigns) >= 2


# ===================================================================
# Edge cases
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
def test_derived_type_args_not_privatised(frontend, horizontal, block_dim, tmp_path):
    """
    Derived-type argument components (e.g. YDGEOMETRY%YRDIM) must NOT
    appear in the ``private(...)`` or ``firstprivate(...)`` clauses of
    ``!$acc parallel loop gang``. Only truly local variables should be
    privatised.
    """
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={'priv_driver': ('priv_kernel',), 'priv_kernel': ()},
        fcodes={'priv_driver': FCODE_PRIV_DRIVER, 'priv_kernel': FCODE_PRIV_KERNEL}
    )

    gang_params = _gang_loop_parameters(routines['priv_kernel'], block_dim)
    assert gang_params and all(params is not None for params in gang_params)
    for routine in routines.values():
        for params in _gang_loop_parameters(routine, block_dim):
            for clause in ('private', 'firstprivate'):
                assert not any(
                    var.strip().lower().startswith('ydgeometry')
                    for var in params.get(clause, '').split(',')
                )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI fails to import undefined module.')]))
@pytest.mark.xfail(reason='Minimal SCC helper still sweeps explicit raw ACC enter/exit pragmas into block loops')
def test_acc_exit_data_outside_block_loop(frontend, horizontal, block_dim, tmp_path):
    """
    ``!$acc enter/exit data`` directives must remain outside the block
    loop, not be swept inside by extract_block_sections.
    """
    routines = _apply_pipeline(
        _sk_pipeline(horizontal, block_dim), tmp_path, frontend,
        call_tree={'acc_exit_driver': ('acc_exit_kernel',), 'acc_exit_kernel': ()},
        fcodes={'acc_exit_driver': FCODE_ACC_EXIT_DRIVER, 'acc_exit_kernel': FCODE_ACC_EXIT_KERNEL}
    )
    kernel = routines['acc_exit_kernel']

    block_loops = _block_loops(kernel, block_dim)
    assert block_loops
    for loop in block_loops:
        assert not [
            p for p in FindNodes(ir.Pragma).visit(loop.body)
            if p.keyword == 'acc' and p.content.startswith(('enter data', 'exit data'))
        ]
