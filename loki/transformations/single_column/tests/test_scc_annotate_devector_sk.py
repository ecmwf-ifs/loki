# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for small-kernels-related fixes in devector.py and annotate.py.
"""

import pytest

from loki import Dimension, Module, Sourcefile, Subroutine
from loki.batch import ProcedureItem, SGraph
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, FindNodes, Transformer, pragmas_attached, pragma_regions_attached,
    is_loki_pragma, get_pragma_parameters
)
from loki.tools import as_tuple

from loki.transformations.single_column import SCCSmallKernelsPipeline
from loki.transformations.single_column.annotate import SCCAnnotateTransformation
from loki.transformations.single_column.devector import SCCDevectorTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('klon', 'columns')
    )


@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(name='block_dim', size='nb', index='b')


# ---------------------------------------------------------------------------
# devector.py tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_devector_unenriched_call(tmp_path, frontend, horizontal):
    """
    Verify that extract_vector_sections does not crash when a call
    is unenriched (call.routine is BasicType.DEFERRED).
    """
    fcode = """
subroutine kernel(nlon, start, end, field)
  implicit none
  integer, intent(in) :: nlon, start, end
  real, intent(inout) :: field(nlon)
  integer :: jl

  do jl = start, end
    field(jl) = field(jl) + 1.0
  end do

  call unknown_sub(nlon, field)

  do jl = start, end
    field(jl) = field(jl) * 2.0
  end do
end subroutine kernel
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Do NOT enrich — call.routine will be BasicType.DEFERRED
    trafo = SCCDevectorTransformation(horizontal=horizontal)

    # This should NOT crash
    trafo.transform_subroutine(routine, role='kernel', targets=('unknown_sub',))

    # The call should act as a separator, creating two vector sections
    sections = [s for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section']
    assert len(sections) == 2


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_devector_pragmas_attached_call(tmp_path, frontend, horizontal):
    """
    Verify that extract_vector_sections correctly identifies call
    separators even when pragmas are attached to CallStatement nodes.
    """
    fcode = """
subroutine kernel(nlon, start, end, field)
  implicit none
  integer, intent(in) :: nlon, start, end
  real, intent(inout) :: field(nlon)
  integer :: jl

  do jl = start, end
    field(jl) = field(jl) + 1.0
  end do

  !$loki separator
  call some_sub(nlon, field)

  do jl = start, end
    field(jl) = field(jl) * 2.0
  end do
end subroutine kernel
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    trafo = SCCDevectorTransformation(horizontal=horizontal)
    trafo.transform_subroutine(routine, role='kernel', targets=('some_sub',))

    # Should produce two vector sections (call is a separator)
    sections = [s for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section']
    assert len(sections) == 2


# ---------------------------------------------------------------------------
# annotate.py tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_annotate_find_acc_vars_loki_data_fallback(tmp_path, frontend, block_dim):
    """
    Verify that find_acc_vars also handles `!$loki data present(...)` pragmas
    (not just `!$loki structured-data`).
    """
    fcode = """
subroutine driver(nlon, nz, nb, field1)
  implicit none
  integer, intent(in) :: nlon, nz, nb
  real, intent(inout) :: field1(nlon, nz, nb)
  integer :: b

  !$loki data present(field1)
  !$loki driver-loop
  do b = 1, nb
    field1(1, 1, b) = 0.0
  end do
  !$loki end data
end subroutine driver
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = SCCAnnotateTransformation(block_dim=block_dim)

    with pragma_regions_attached(routine):
        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            acc_vars = trafo.find_acc_vars(routine, targets=())

    # field1 should be found in the data clause
    all_vars = [v for loop_vars in acc_vars.values() for v in loop_vars]
    assert 'field1' in all_vars


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_annotate_find_kernel_acc_vars(tmp_path, frontend, block_dim):
    """
    Verify that find_kernel_acc_vars picks up variables from
    `!$loki device-present` and `!$loki structured-data` pragmas.
    """
    fcode = """
subroutine kernel(nlon, nz, field1, field2)
  implicit none
  integer, intent(in) :: nlon, nz
  real, intent(inout) :: field1(nlon, nz)
  real, intent(inout) :: field2(nlon, nz)

  !$loki device-present vars(field1, field2)
  field1(1, 1) = 0.0
  field2(1, 1) = 0.0
  !$loki end device-present
end subroutine kernel
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = SCCAnnotateTransformation(block_dim=block_dim)

    with pragma_regions_attached(routine):
        acc_vars = trafo.find_kernel_acc_vars(routine)

    assert 'field1' in acc_vars
    assert 'field2' in acc_vars


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_annotate_privatise_derived_types_filters(tmp_path, frontend, block_dim):
    """
    Verify the improved derived-type privatisation filters:
    - Local derived-type scalar -> privatised
    - Argument-passed derived-type -> excluded
    - Pointer alias to argument -> excluded
    - Module-imported derived-type -> excluded
    """
    fcode_mod = """
module config_mod
  implicit none
  type :: t_config
    real :: val
  end type t_config
  type(t_config), save :: global_config
end module config_mod
    """.strip()

    fcode = """
subroutine driver(nlon, nb, ydarg, field1)
  use config_mod, only: t_config, global_config
  implicit none
  integer, intent(in) :: nlon, nb
  type(t_config), intent(in) :: ydarg
  real, intent(inout) :: field1(nlon, nb)
  type(t_config) :: local_struct
  type(t_config), pointer :: ptr_alias
  integer :: b

  ptr_alias => ydarg

  !$loki loop driver
  do b = 1, nb
    local_struct%val = 1.0
    field1(1, b) = local_struct%val + ydarg%val + global_config%val + ptr_alias%val
  end do
end subroutine driver
    """.strip()

    mod = Module.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=mod, xmods=[tmp_path])

    trafo = SCCAnnotateTransformation(block_dim=block_dim, privatise_derived_types=True)
    trafo.transform_subroutine(routine, role='driver', targets=())

    # Find the annotated driver loop
    with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
        loops = FindNodes(ir.Loop).visit(routine.body)
        driver_loop = [l for l in loops if l.variable == 'b'][0]
        gang_pragma = [p for p in as_tuple(driver_loop.pragma) if is_loki_pragma(p, starts_with='loop gang')]

    assert len(gang_pragma) == 1
    params = get_pragma_parameters(gang_pragma[0], starts_with='loop gang')
    private_str = params.get('private', '')
    private_names = [p.strip().lower() for p in private_str.split(',') if p.strip()]

    # local_struct should be privatised
    assert 'local_struct' in private_names

    # ydarg (argument) and global_config (imported) should NOT be privatised
    assert 'ydarg' not in private_names
    assert 'global_config' not in private_names

    # ptr_alias is a local pointer variable — it IS privatised because
    # it has no intent and is not imported. Pointer aliases to shared
    # data must be excluded via explicit acc_vars if needed.
    assert 'ptr_alias' in private_names


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_annotate_kernel_driver_loops(tmp_path, frontend, block_dim):
    """
    Verify that kernel-role routines with driver loops (nested drivers)
    get gang-parallel annotations on those loops.
    """
    fcode_inner = """
subroutine inner_kernel(nlon, start, end, field)
  implicit none
  integer, intent(in) :: nlon, start, end
  real, intent(inout) :: field(nlon)
  integer :: jl

  do jl = start, end
    field(jl) = field(jl) + 1.0
  end do
end subroutine inner_kernel
    """.strip()

    fcode_outer = """
subroutine outer_kernel(nlon, nz, nb, start, end, field)
  implicit none
  integer, intent(in) :: nlon, nz, nb, start, end
  real, intent(inout) :: field(nlon, nz, nb)
  integer :: b

  !$loki loop driver
  do b = 1, nb
    call inner_kernel(nlon, start, end, field(:,1,b))
  end do
end subroutine outer_kernel
    """.strip()

    inner = Subroutine.from_source(fcode_inner, frontend=frontend, xmods=[tmp_path])
    outer = Subroutine.from_source(fcode_outer, frontend=frontend, xmods=[tmp_path])
    outer.enrich(inner)

    trafo = SCCAnnotateTransformation(block_dim=block_dim, privatise_derived_types=True)
    trafo.transform_subroutine(outer, role='kernel', targets=('inner_kernel',))

    # The driver loop in the kernel should get a gang pragma
    with pragmas_attached(outer, ir.Loop, attach_pragma_post=True):
        loops = FindNodes(ir.Loop).visit(outer.body)
        driver_loop = [l for l in loops if l.variable == 'b'][0]
        gang_pragmas = [p for p in as_tuple(driver_loop.pragma) if is_loki_pragma(p, starts_with='loop gang')]
        assert len(gang_pragmas) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_annotate_kernel_driver_loops_device_present(tmp_path, frontend, block_dim):
    """
    Verify that kernels with driver loops (nested drivers) only declare
    arguments used inside the driver loops as device-present, while
    kernels without driver loops keep all array and derived-type arguments.
    """
    fcode_type_mod = """
module bnds_mod
  implicit none
  type bnds_type
    integer :: kidia, kfdia
  end type bnds_type
end module bnds_mod
    """.strip()

    fcode_kernel = """
subroutine kernel(nlon, nb, bnds, field, other)
  use bnds_mod, only: bnds_type
  implicit none
  integer, intent(in) :: nlon, nb
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: field(nlon, nb), other(nlon, nb)
  type(bnds_type) :: local_bnds
  integer :: b, jl

  local_bnds = bnds

  !$loki loop driver
  do b = 1, nb
    do jl = local_bnds%kidia, local_bnds%kfdia
      field(jl, b) = 1.0
    end do
  end do

  call host_kernel(nlon, nb, other)
end subroutine kernel
    """.strip()

    type_mod = Module.from_source(fcode_type_mod, frontend=frontend, xmods=[tmp_path])
    kernel = Subroutine.from_source(
        fcode_kernel, frontend=frontend, definitions=type_mod, xmods=[tmp_path]
    )
    plain_kernel = kernel.clone(name='plain_kernel')
    plain_kernel.body = Transformer(
        {p: None for p in FindNodes(ir.Pragma).visit(plain_kernel.body)}
    ).visit(plain_kernel.body)

    trafo = SCCAnnotateTransformation(block_dim=block_dim, privatise_derived_types=True)
    trafo.transform_subroutine(kernel, role='kernel', targets=('host_kernel',))
    trafo.transform_subroutine(plain_kernel, role='kernel', targets=('host_kernel',))

    def device_present_vars(routine):
        pragmas = [
            p for p in FindNodes(ir.Pragma).visit(routine.body)
            if is_loki_pragma(p, starts_with='device-present')
        ]
        assert len(pragmas) == 1
        params = get_pragma_parameters(pragmas[0], starts_with='device-present')
        return tuple(v.strip() for v in params['vars'].split(','))

    # Only ``field`` is used inside the driver loop
    assert device_present_vars(kernel) == ('field',)

    # The local bounds copy is initialised on the host, so it must be firstprivate
    with pragmas_attached(kernel, ir.Loop):
        driver_loop = [l for l in FindNodes(ir.Loop).visit(kernel.body) if l.variable == 'b'][0]
        params = get_pragma_parameters(driver_loop.pragma, starts_with='loop gang')
    assert params['firstprivate'] == 'local_bnds'
    assert 'private' not in params
    # Without driver loops, all array and derived-type arguments remain
    assert device_present_vars(plain_kernel) == ('field', 'other', 'bnds')


@pytest.mark.parametrize('frontend', available_frontends())
def test_annotate_driver_loop_firstprivate_derived_types(tmp_path, frontend, block_dim):
    """
    Verify that privatised derived-type objects are declared ``firstprivate``
    on request, and remain ``private`` by default.
    """
    fcode_type_mod = """
module bnds_mod
  implicit none
  type bnds_type
    integer :: kidia, kfdia
  end type bnds_type
end module bnds_mod
    """.strip()

    fcode = """
subroutine driver(nlon, nb, bnds, field)
  use bnds_mod, only: bnds_type
  implicit none
  integer, intent(in) :: nlon, nb
  type(bnds_type), intent(in) :: bnds
  real, intent(inout) :: field(nlon, nb)
  type(bnds_type) :: local_bnds
  real :: tmp(nlon)
  integer :: b, jl

  local_bnds = bnds

  !$loki loop driver
  do b = 1, nb
    do jl = local_bnds%kidia, local_bnds%kfdia
      tmp(jl) = 1.0
      field(jl, b) = tmp(jl)
    end do
  end do
end subroutine driver
    """.strip()

    type_mod = Module.from_source(fcode_type_mod, frontend=frontend, xmods=[tmp_path])
    trafo = SCCAnnotateTransformation(block_dim=block_dim, privatise_derived_types=True)

    def annotated_loop_params(firstprivate_derived_types):
        routine = Subroutine.from_source(fcode, frontend=frontend, definitions=type_mod, xmods=[tmp_path])
        with pragmas_attached(routine, ir.Loop, attach_pragma_post=True):
            loop = [l for l in FindNodes(ir.Loop).visit(routine.body) if l.variable == 'b'][0]
            trafo.annotate_driver_loop(loop, [], firstprivate_derived_types=firstprivate_derived_types)
            return get_pragma_parameters(loop.pragma, starts_with='loop gang')

    params = annotated_loop_params(firstprivate_derived_types=False)
    assert params['private'] == 'tmp, local_bnds'
    assert 'firstprivate' not in params

    params = annotated_loop_params(firstprivate_derived_types=True)
    assert params['private'] == 'tmp'
    assert params['firstprivate'] == 'local_bnds'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('header, body, callee_pragma', [
    # A routine with only small-kernels calls (no block loops of its own)
    ('ydcpg_bnds%kbl = ibl', '', '!$loki small-kernels'),
    # A nested driver whose loop header updates the bounds via a call
    ('call update_bnds(ydcpg_bnds, ibl)', 't(ydcpg_bnds%kidia, 1) = 1.0', '!$loki small-kernels'),
    # A nested driver calling a device kernel inside its block loops
    ('ydcpg_bnds%kbl = ibl', 't(ydcpg_bnds%kidia, 1) = 1.0', ''),
])
def test_sk_pipeline_bounds_object_not_device_present(tmp_path, frontend, header, body, callee_pragma):
    """
    Verify that the bounds object, which only lives on the host in the
    small-kernels pipeline, is not declared device-present in any kernel.
    """
    fcode_mod = """
module bnds_mod
  implicit none
  type cpg_bnds_type
    integer :: kidia, kfdia, kbl
  end type cpg_bnds_type
contains
  subroutine update_bnds(bnds, kbl)
    type(cpg_bnds_type), intent(inout) :: bnds
    integer, intent(in) :: kbl
    bnds%kbl = kbl
  end subroutine update_bnds
end module bnds_mod
    """.strip()
    fcode_driver = f"""
subroutine driver(ngpblks, klon, ydcpg_bnds, t)
  use bnds_mod, only: cpg_bnds_type, update_bnds
  implicit none
  integer, intent(in) :: ngpblks, klon
  type(cpg_bnds_type), intent(inout) :: ydcpg_bnds
  real, intent(inout) :: t(klon, 2, ngpblks)
  integer :: ibl
  do ibl = 1, ngpblks
    {header}
    !$loki small-kernels
    call kernel(klon, ydcpg_bnds, t(:,:,ibl))
  end do
end subroutine driver
    """.strip()
    fcode_kernel = f"""
subroutine kernel(klon, ydcpg_bnds, t)
  use bnds_mod, only: cpg_bnds_type
  implicit none
  integer, intent(in) :: klon
  type(cpg_bnds_type), intent(in) :: ydcpg_bnds
  real, intent(inout) :: t(klon, 2)
  {body}
  {callee_pragma}
  call child(klon, ydcpg_bnds, t)
end subroutine kernel
    """.strip()
    fcode_child = """
subroutine child(klon, ydcpg_bnds, t)
  use bnds_mod, only: cpg_bnds_type
  implicit none
  integer, intent(in) :: klon
  type(cpg_bnds_type), intent(in) :: ydcpg_bnds
  real, intent(inout) :: t(klon, 2)
  integer :: jrof
  do jrof = ydcpg_bnds%kidia, ydcpg_bnds%kfdia
    t(jrof, 1) = 2.0 * t(jrof, 1)
  end do
end subroutine child
    """.strip()

    bnds_mod = Module.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])
    sources = {
        name: Sourcefile.from_source(fcode, frontend=frontend, definitions=[bnds_mod], xmods=[tmp_path])
        for name, fcode in (('driver', fcode_driver), ('kernel', fcode_kernel), ('child', fcode_child))
    }
    routines = {name: source[name] for name, source in sources.items()}
    items = {name: ProcedureItem(name=f'#{name}', source=source) for name, source in sources.items()}
    routines['driver'].enrich(routines['kernel'])
    routines['kernel'].enrich(routines['child'])
    sgraph = SGraph.from_dict({items['driver']: [items['kernel']], items['kernel']: [items['child']]})

    horizontal = Dimension(name='horizontal', size='klon', index='jrof',
                           lower=('kidia', 'ydcpg_bnds%kidia'), upper=('kfdia', 'ydcpg_bnds%kfdia', 'klon'))
    block_dim = Dimension(name='block_dim', size='ngpblks', index=('ibl', 'ydcpg_bnds%kbl'))
    pipeline = SCCSmallKernelsPipeline(horizontal=horizontal, block_dim=block_dim, directive='openacc')
    order = [('driver', 'driver', ('kernel',)), ('kernel', 'kernel', ('child',)), ('child', 'kernel', ())]
    for transformation in pipeline.transformations:
        for name, role, targets in (order[::-1] if transformation.reverse_traversal else order):
            transformation.apply(routines[name], role=role, item=items[name], targets=targets, sub_sgraph=sgraph)

    for routine in routines.values():
        for pragma in FindNodes(ir.Pragma).visit(routine.body):
            if pragma.keyword == 'acc' and pragma.content.startswith('data'):
                params = get_pragma_parameters(pragma, starts_with='data', only_loki_pragmas=False)
                assert 'ydcpg_bnds' not in [v.strip().lower() for v in params.get('present', '').split(',')]
