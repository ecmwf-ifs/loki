# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`LowerBlockIndexSKTransformation` and the extended
:any:`InjectBlockIndexTransformation.get_block_index` with ``loop=`` parameter.
"""

# pylint: disable=redefined-outer-name

import pytest

from loki import Dimension, Module, Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes, pragmas_attached
from loki.transformations.block_index_transformations import InjectBlockIndexTransformation
from loki.transformations.block_index_transformations_sk import LowerBlockIndexSKTransformation


@pytest.fixture
def block_dim():
    return Dimension(
        name='block_dim', size='nb', index='ibl',
        bounds=('1', 'nb'),
        aliases=('bnds%kbl', 'jkglo'),
    )


@pytest.fixture
def horizontal():
    return Dimension(
        name='horizontal', size='nproma', index='jl',
        bounds=('1', 'nproma'),
    )


# -------------------------------------------------------------------
# Fortran fixtures
# -------------------------------------------------------------------

FCODE_TYPE_MOD = """
module type_mod
  implicit none
  type :: bnds_type
    integer :: kidia
    integer :: kfdia
    integer :: kbl
  end type bnds_type
  type :: geom_type
    integer :: ngpblks
  end type geom_type
end module type_mod
"""

FCODE_PARKIND = """
module parkind1
  implicit none
  integer, parameter :: jpim = selected_int_kind(9)
  integer, parameter :: jprb = selected_real_kind(13, 300)
end module parkind1
"""


def _make_item(routine, role='kernel', targets=(), config=None):
    """Create a minimal ProcedureItem-like mock for testing."""

    class _MockItem:
        def __init__(self, routine, role, targets, config):
            self.routine = routine
            self.role = role
            self.targets = targets
            self.local_name = routine.name.lower()
            self.trafo_data = {}
            self.config = config or {}

    return _MockItem(routine, role, targets, config)


def _build_sgraph(items):
    """Build a minimal successor map from a list of items."""

    class _MockSGraph:
        def __init__(self, items):
            self._map = {items[i]: [items[i+1]] for i in range(len(items)-1)}
            if items:
                self._map.setdefault(items[-1], [])

        def successors(self, item):
            return self._map.get(item, [])

    return _MockSGraph(items)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_driver_passes_relevant_vars(frontend, block_dim):
    """
    Verify that LowerBlockIndexSKTransformation passes driver-loop header
    variables as new arguments to the callee.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field, ydbnds)
  use type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  type(bnds_type), intent(in) :: ydbnds
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  integer :: jl, jk

  do jk = 1, nlev
    do jl = 1, nproma
      field(jl, jk) = 0.0
    end do
  end do
end subroutine kernel
"""
    type_mod = Module.from_source(FCODE_TYPE_MOD, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=[type_mod])
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend, definitions=[type_mod])

    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # Check that nb was added as kwarg to the call
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 1
    call = calls[0]
    kwarg_names = [k for k, _ in call.kwarguments]
    assert 'nb' in [str(k).lower() for k in kwarg_names]

    # Check that nb is now in kernel arguments
    kernel_arg_names = [a.name.lower() for a in kernel.arguments]
    assert 'nb' in kernel_arg_names

    # Check trafo_data was propagated
    assert 'LowerBlockIndex' in kernel_item.trafo_data
    assert 'relevant_vars' in kernel_item.trafo_data['LowerBlockIndex']
    assert 'driver_loop' in kernel_item.trafo_data['LowerBlockIndex']


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_kernel_propagates_to_sub_kernel(frontend, block_dim):
    """
    Verify recursive propagation: driver -> kernel -> sub_kernel.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)

  !$loki small-kernels
  call sub_kernel(nproma, nlev, field)
end subroutine kernel
"""
    fcode_sub_kernel = """
subroutine sub_kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1, 1) = 1.0
end subroutine sub_kernel
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    sub_kernel = Subroutine.from_source(fcode_sub_kernel, frontend=frontend)

    driver.enrich(kernel)
    kernel.enrich(sub_kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=('sub_kernel',))
    sub_kernel_item = _make_item(sub_kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item, sub_kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)

    # Process driver first
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # Now process kernel (it should propagate to sub_kernel)
    trafo.transform_subroutine(
        kernel, role='kernel', targets=('sub_kernel',),
        item=kernel_item, sub_sgraph=sgraph
    )

    # Verify sub_kernel got trafo_data
    assert 'LowerBlockIndex' in sub_kernel_item.trafo_data
    assert 'relevant_vars' in sub_kernel_item.trafo_data['LowerBlockIndex']

    # Verify nb was added to sub_kernel arguments
    sub_kernel_arg_names = [a.name.lower() for a in sub_kernel.arguments]
    assert 'nb' in sub_kernel_arg_names


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_promotes_local_arrays(frontend, block_dim):
    """
    Verify that callee-local arrays are promoted with the block dimension.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  real :: tmp(nproma, nlev)
  integer :: jl, jk

  do jk = 1, nlev
    do jl = 1, nproma
      tmp(jl, jk) = field(jl, jk) * 2.0
      field(jl, jk) = tmp(jl, jk)
    end do
  end do
end subroutine kernel
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # Check tmp was promoted to 3D (nproma, nlev, nb)
    tmp_var = kernel.variable_map.get('tmp')
    assert tmp_var is not None
    assert len(tmp_var.shape) == 3
    assert tmp_var.shape[-1] == 'nb'


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_injects_unstructured_data_pragmas(frontend, block_dim):
    """
    Verify that !$loki unstructured-data create/delete pragmas are injected.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  real :: work(nproma)
  integer :: jl, jk

  do jk = 1, nlev
    do jl = 1, nproma
      work(jl) = field(jl, jk)
      field(jl, jk) = work(jl) + 1.0
    end do
  end do
end subroutine kernel
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # Check pragmas in kernel body
    pragmas = FindNodes(ir.Pragma).visit(kernel.body)
    create_pragmas = [p for p in pragmas if 'unstructured-data create' in p.content]
    delete_pragmas = [p for p in pragmas if 'unstructured-data delete' in p.content]
    assert len(create_pragmas) == 1
    assert len(delete_pragmas) == 1
    assert 'work' in create_pragmas[0].content


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_does_not_duplicate_existing_loki_unstructured_data_pragmas(frontend, block_dim):
    """
    Verify that existing generic ``!$loki unstructured-data`` pragmas
    suppress duplicate injection before ``PragmaModel`` lowering.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  real :: work(nproma)
  integer :: jl, jk

  !$loki unstructured-data create(work)
  do jk = 1, nlev
    do jl = 1, nproma
      work(jl) = field(jl, jk)
      field(jl, jk) = work(jl) + 1.0
    end do
  end do
  !$loki exit unstructured-data delete(work)
end subroutine kernel
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    pragmas = FindNodes(ir.Pragma).visit(kernel.body)
    create_pragmas = [p for p in pragmas if 'unstructured-data create' in p.content]
    delete_pragmas = [p for p in pragmas if 'unstructured-data delete' in p.content]
    assert len(create_pragmas) == 1
    assert len(delete_pragmas) == 1
    assert create_pragmas[0].content == 'unstructured-data create(work)'
    assert delete_pragmas[0].content == 'exit unstructured-data delete(work)'


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_replaces_block_index_with_range(frontend, block_dim):
    """
    Verify that block-index subscripts in call arguments are replaced
    with range indices (:).
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1, 1) = 0.0
end subroutine kernel
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # The call argument field(:,:,ibl) should become field(:,:,:)
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    call = calls[0]
    field_arg = call.arguments[2]  # third positional arg
    assert all(isinstance(d, sym.RangeIndex) for d in field_arg.dimensions)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_imports_derived_types_and_kinds(frontend, block_dim):
    """
    Verify that derived-type and kind-parameter imports are propagated
    to the callee when new arguments with those types are added.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field, ydbnds)
  use type_mod, only: bnds_type
  use parkind1, only: jpim
  implicit none
  integer(kind=jpim), intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  type(bnds_type), intent(in) :: ydbnds
  integer :: ibl, kidia, kfdia
  kidia = ydbnds%kidia
  kfdia = ydbnds%kfdia

  do ibl = 1, nb
    kidia = ydbnds%kidia
    kfdia = ydbnds%kfdia
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1, 1) = 0.0
end subroutine kernel
"""
    type_mod = Module.from_source(FCODE_TYPE_MOD, frontend=frontend)
    parkind = Module.from_source(FCODE_PARKIND, frontend=frontend)
    driver = Subroutine.from_source(
        fcode_driver, frontend=frontend, definitions=[type_mod, parkind]
    )
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend, definitions=[type_mod, parkind])
    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # Check that kernel now imports from type_mod or parkind1
    kernel_imports = list(kernel.imports)
    imported_names = set()
    for imp in kernel_imports:
        for s in imp.symbols:
            imported_names.add(s.name.lower())

    # ydbnds has type bnds_type -> should propagate the import
    # (only if ydbnds was passed as new arg — depends on dtype matching)
    # At minimum, check that some import was added
    kernel_arg_names = [a.name.lower() for a in kernel.arguments]
    # kidia and kfdia are scalars used in the loop header assignments
    # They should be passed (or their parent ydbnds via dtype match)
    assert any(name in kernel_arg_names for name in ('kidia', 'kfdia', 'ydbnds'))


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_only_processes_pragma_annotated_calls(frontend, block_dim):
    """
    Verify that calls without the !$loki small-kernels pragma are not
    modified by the transformation.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    call regular_kernel(nproma, nlev, field(:,:,ibl))
    !$loki small-kernels
    call sk_kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_regular = """
subroutine regular_kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1,1) = 0.0
end subroutine regular_kernel
"""
    fcode_sk = """
subroutine sk_kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1,1) = 0.0
end subroutine sk_kernel
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    regular = Subroutine.from_source(fcode_regular, frontend=frontend)
    sk = Subroutine.from_source(fcode_sk, frontend=frontend)
    driver.enrich(regular)
    driver.enrich(sk)

    regular_item = _make_item(regular, role='kernel', targets=())
    sk_item = _make_item(sk, role='kernel', targets=())
    driver_item = _make_item(driver, role='driver', targets=('regular_kernel', 'sk_kernel'))

    class _MockSGraph:
        def successors(self, item):
            if item is driver_item:
                return [regular_item, sk_item]
            return []

    sgraph = _MockSGraph()

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('regular_kernel', 'sk_kernel'),
        item=driver_item, sub_sgraph=sgraph
    )

    # regular_kernel should be unchanged (still 2 args)
    assert len(regular.arguments) == 3  # nproma, nlev, field

    # sk_kernel should have gained nb as argument
    sk_arg_names = [a.name.lower() for a in sk.arguments]
    assert 'nb' in sk_arg_names


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_inject_block_index_per_loop(frontend, block_dim):
    """
    Verify InjectBlockIndexTransformation.get_block_index resolves
    block-index aliases from loop-body assignments.
    """
    fcode = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: jkglo, ibl

  do jkglo = 1, nb
    ibl = jkglo
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = InjectBlockIndexTransformation(block_dim=block_dim)

    variable_map = routine.variable_map
    # Without loop context, should find ibl directly
    result = trafo.get_block_index(routine, variable_map)
    assert result == 'ibl'

    # With loop context providing assignment ibl = jkglo, should still resolve
    with pragmas_attached(routine, ir.Loop):
        loops = FindNodes(ir.Loop).visit(routine.body)
        assert len(loops) == 1
        result_with_loop = trafo.get_block_index(routine, variable_map, loop=loops[0])
        assert result_with_loop == 'ibl'


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_dtype_matching_kwargs(frontend, block_dim):
    """
    Verify that when a derived-type member variable (e.g. ``ydbnds%kidia``)
    appears in the driver loop header, its root parent (``ydbnds``) is
    matched by dtype to an existing callee argument (``ydcpg_bnds``) and
    passed as a keyword argument using the callee's argument name.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field, ydbnds)
  use type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  type(bnds_type), intent(in) :: ydbnds
  integer :: ibl, kidia

  do ibl = 1, nb
    kidia = ydbnds%kidia
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field, ydcpg_bnds)
  use type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  type(bnds_type), intent(in) :: ydcpg_bnds
  field(1, 1) = 0.0
end subroutine kernel
"""
    type_mod = Module.from_source(FCODE_TYPE_MOD, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=[type_mod])
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend, definitions=[type_mod])
    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # ydbnds is NOT passed in the original call, but ydbnds%kidia appears
    # in the loop body. The root parent ydbnds has dtype bnds_type which
    # matches the callee's ydcpg_bnds argument — so ydbnds should be passed
    # as a kwarg with key 'ydcpg_bnds' (the callee's arg name).
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    call = calls[0]
    kwarg_dict = {str(k).lower(): v for k, v in call.kwarguments}

    # ydbnds passed as kwarg matching callee's ydcpg_bnds by dtype
    assert 'ydcpg_bnds' in kwarg_dict
    assert kwarg_dict['ydcpg_bnds'] == 'ydbnds'

    # ydbnds should NOT appear as a new positional argument in the kernel
    kernel_arg_names = [a.name.lower() for a in kernel.arguments]
    assert 'ydbnds' not in kernel_arg_names
    assert 'ydcpg_bnds' in kernel_arg_names  # existing arg kept
    assert 'nb' in kernel_arg_names  # nb added as new arg


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_argument_dims_updated(frontend, block_dim):
    """
    Verify that callee argument array declarations are promoted from
    2D to 3D when the call-site argument has higher rank.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1, 1) = 0.0
end subroutine kernel
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver.enrich(kernel)

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # field in kernel should now be 3D: (nproma, nlev, nb)
    field_var = kernel.variable_map['field']
    assert len(field_var.shape) == 3
    assert field_var.shape[-1] == 'nb'

    # Verify dimensions match shape
    assert len(field_var.dimensions) == 3


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_new_derived_dummy_declared_before_bounds_use(frontend):
    """
    Verify that newly-added derived-type dummies are declared before any
    existing dummy-array declarations whose bounds are updated to use them.
    """
    fcode_type_mod = """
module type_mod
  implicit none
  type :: yrdim_type
    integer :: ngpblks
  end type yrdim_type
  type :: geometry
    type(yrdim_type) :: yrdim
  end type geometry
end module type_mod
"""
    fcode_driver = """
subroutine driver(nproma, nlev, nb, geom, field)
  use type_mod, only: geometry
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  type(geometry), intent(in) :: geom
  real, intent(inout) :: field(nproma, nlev, geom%yrdim%ngpblks)
  integer :: ibl, jkglo

  do ibl = 1, nb
    jkglo = geom%yrdim%ngpblks
    !$loki small-kernels
    call kernel(nproma, nlev, field(:, :, ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1, 1) = 0.0
end subroutine kernel
"""
    type_mod = Module.from_source(fcode_type_mod, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=[type_mod])
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend, definitions=[type_mod])
    driver.enrich(kernel)

    geom_block_dim = Dimension(
        name='block_dim', size='geom%yrdim%ngpblks', index='ibl',
        bounds=('1', 'geom%yrdim%ngpblks'), aliases=('jkglo',)
    )

    driver_item = _make_item(driver, role='driver', targets=('kernel',))
    kernel_item = _make_item(kernel, role='kernel', targets=())
    sgraph = _build_sgraph([driver_item, kernel_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=geom_block_dim)
    trafo.transform_subroutine(
        driver, role='driver', targets=('kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    decls = [decl for decl in FindNodes(ir.VariableDeclaration).visit(kernel.spec) if len(decl.symbols) == 1]
    decl_pos = {decl.symbols[0].name.lower(): idx for idx, decl in enumerate(decls)}

    assert 'geom' in [arg.name.lower() for arg in kernel.arguments]
    assert kernel.variable_map['field'].shape[-1] == 'geom%yrdim%ngpblks'
    assert decl_pos['geom'] < decl_pos['field']


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI cannot import undefined modules')]))
def test_sk_deferred_call_skipped(frontend, block_dim):
    """
    Verify that un-enriched (DEFERRED) calls are skipped gracefully
    with a warning, and the routine is not modified.
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field)
  implicit none
  integer, intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call unknown_kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    # Do NOT enrich — call remains DEFERRED

    driver_item = _make_item(driver, role='driver', targets=('unknown_kernel',))
    sgraph = _build_sgraph([driver_item])

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    # Should not raise — just warn and skip
    trafo.transform_subroutine(
        driver, role='driver', targets=('unknown_kernel',),
        item=driver_item, sub_sgraph=sgraph
    )

    # Driver should be unchanged — call still has original arguments
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert len(calls[0].arguments) == 3  # nproma, nlev, field(:,:,ibl)
