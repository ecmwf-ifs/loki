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

from loki import Dimension, Module, Sourcefile, Subroutine
from loki.batch import ProcedureItem, SGraph
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes, pragmas_attached
from loki.transformations.block_index_transformations import InjectBlockIndexTransformation
from loki.transformations.block_index_transformations_sk import LowerBlockIndexSKTransformation
from loki.types import BasicType


@pytest.fixture
def block_dim():
    return Dimension(
        name='block_dim', size='nb', index='ibl',
        bounds=('1', 'nb'),
        aliases=('bnds%kbl', 'jkglo'),
    )


FCODE_TYPE_MOD = """
module type_mod
  implicit none
  type :: bnds_type
    integer :: kidia
    integer :: kfdia
    integer :: kbl
  end type bnds_type
end module type_mod
"""

FCODE_PARKIND = """
module parkind1
  implicit none
  integer, parameter :: jpim = selected_int_kind(9)
end module parkind1
"""


def _apply_sk_lowering(tmp_path, frontend, block_dim, call_tree, fcodes, definitions=()):
    """
    Parse the routines in :data:`fcodes`, enrich them along :data:`call_tree`
    and apply :any:`LowerBlockIndexSKTransformation` in call-tree order,
    with the first routine in :data:`call_tree` as the driver.

    Callees without source in :data:`fcodes` remain un-enriched.

    Returns
    -------
    tuple of dict
        The parsed routines and their scheduler items, keyed by name.
    """
    sources = {
        name: Sourcefile.from_source(fcode, frontend=frontend, definitions=definitions, xmods=[tmp_path])
        for name, fcode in fcodes.items()
    }
    routines = {name: source[name] for name, source in sources.items()}
    items = {name: ProcedureItem(name=f'#{name}', source=source) for name, source in sources.items()}

    for name, callees in call_tree.items():
        routines[name].enrich(tuple(routines[c] for c in callees if c in routines))
    sgraph = SGraph.from_dict({
        items[name]: [items[c] for c in callees if c in items] for name, callees in call_tree.items()
    })

    trafo = LowerBlockIndexSKTransformation(block_dim=block_dim)
    for idx, (name, callees) in enumerate(call_tree.items()):
        trafo.transform_subroutine(
            routines[name], role='driver' if idx == 0 else 'kernel', targets=callees,
            item=items[name], sub_sgraph=sgraph
        )
    return routines, items


def _calls(routine):
    """Return the calls in :data:`routine`, keyed by callee name."""
    return {str(call.name).lower(): call for call in FindNodes(ir.CallStatement).visit(routine.body)}


def _pragma_contents(routine):
    """Return the contents of all pragmas in the body of :data:`routine`."""
    return [p.content for p in FindNodes(ir.Pragma).visit(routine.body)]


@pytest.mark.parametrize('frontend', available_frontends())
def test_sk_lower_block_index_driver(tmp_path, frontend, block_dim):
    """
    Test the lowering of small-kernels calls in a driver loop, including
    new arguments from the loop header, promotion of kernel temporaries,
    data pragmas, imports, dtype-matched keyword arguments, and calls that
    are not lowered (no small-kernels pragma or not enriched).
    """
    fcode_driver = """
subroutine driver(nproma, nlev, nb, field, coef, ydbnds)
  use type_mod, only: bnds_type
  use parkind1, only: jpim
  implicit none
  integer(kind=jpim), intent(in) :: nproma, nlev, nb
  real, intent(inout) :: field(nproma, nlev, nb)
  real, intent(in) :: coef(5)
  type(bnds_type), intent(in) :: ydbnds
  integer :: ibl, kidia

  do ibl = 1, nb
    kidia = ydbnds%kidia
    call regular_kernel(nproma, nlev, field(:,:,ibl))
    !$loki small-kernels
    call kernel(nproma, nlev, field(:,:,ibl), coef(:))
    !$loki small-kernels
    call kernel_bnds(nproma, nlev, field(:,:,ibl))
    !$loki small-kernels
    call unknown_kernel(nproma, nlev, field(:,:,ibl))
  end do
end subroutine driver
"""
    fcode_regular_kernel = """
subroutine regular_kernel(nproma, nlev, field)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  field(1, 1) = 0.0
end subroutine regular_kernel
"""
    fcode_kernel = """
subroutine kernel(nproma, nlev, field, coef)
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  real, intent(in) :: coef(5)
  real :: tmp(nproma, nlev)
  integer :: jl, jk

  do jk = 1, nlev
    do jl = 1, nproma
      tmp(jl, jk) = field(jl, jk) * coef(1)
      field(jl, jk) = tmp(jl, jk)
    end do
  end do
end subroutine kernel
"""
    fcode_kernel_bnds = """
subroutine kernel_bnds(nproma, nlev, field, ydcpg_bnds)
  use type_mod, only: bnds_type
  implicit none
  integer, intent(in) :: nproma, nlev
  real, intent(inout) :: field(nproma, nlev)
  type(bnds_type), intent(in) :: ydcpg_bnds
  real :: work(nproma)
  integer :: jl, jk

  !$loki unstructured-data create(work)
  do jk = 1, nlev
    do jl = ydcpg_bnds%kidia, ydcpg_bnds%kfdia
      work(jl) = field(jl, jk)
      field(jl, jk) = work(jl) + 1.0
    end do
  end do
  !$loki exit unstructured-data delete(work)
end subroutine kernel_bnds
"""
    definitions = (
        Module.from_source(FCODE_TYPE_MOD, frontend=frontend, xmods=[tmp_path]),
        Module.from_source(FCODE_PARKIND, frontend=frontend, xmods=[tmp_path]),
    )
    routines, items = _apply_sk_lowering(
        tmp_path, frontend, block_dim,
        call_tree={
            'driver': ('regular_kernel', 'kernel', 'kernel_bnds', 'unknown_kernel'),
            'regular_kernel': (), 'kernel': (), 'kernel_bnds': ()
        },
        fcodes={
            'driver': fcode_driver, 'regular_kernel': fcode_regular_kernel,
            'kernel': fcode_kernel, 'kernel_bnds': fcode_kernel_bnds
        },
        definitions=definitions
    )
    calls = _calls(routines['driver'])

    # Calls without small-kernels pragma or without enrichment are not modified
    for name in ('regular_kernel', 'unknown_kernel'):
        assert calls[name].arguments == ('nproma', 'nlev', 'field(:, :, ibl)')
        assert not calls[name].kwarguments
    assert routines['regular_kernel'].arguments == ('nproma', 'nlev', 'field(nproma, nlev)')
    assert not items['regular_kernel'].trafo_data

    # Block-index subscripts are replaced, unblocked arrays and the loop variable are not
    # passed on, and loop-header variables (incl. derived-type roots) become new arguments
    kernel = routines['kernel']
    assert calls['kernel'].arguments == ('nproma', 'nlev', 'field(:, :, :)', 'coef(:)')
    assert calls['kernel'].kwarguments == (('kidia', 'kidia'), ('nb', 'nb'), ('ydbnds', 'ydbnds'))
    assert kernel.arguments == (
        'nproma', 'nlev', 'field(nproma, nlev, nb)', 'coef(5)', 'kidia', 'nb', 'ydbnds'
    )
    assert kernel.variable_map['field'].shape == ('nproma', 'nlev', 'nb')
    assert kernel.variable_map['ibl'].type.intent is None

    # Imports for the derived type and kind of new arguments are added to the kernel
    # (OMNI resolves kind parameters to their value, which requires no import)
    assert 'bnds_type' in kernel.imported_symbols
    if frontend == OMNI:
        assert kernel.variable_map['nb'].type.kind == 4
    else:
        assert 'jpim' in kernel.imported_symbols

    # Kernel temporaries are promoted and wrapped in unstructured data pragmas
    assert kernel.variable_map['tmp'].shape == ('nproma', 'nlev', 'nb')
    assert _pragma_contents(kernel) == ['unstructured-data create(tmp)', 'exit unstructured-data delete(tmp)']

    # The bounds object is passed to the existing argument of the same type, and
    # existing unstructured data pragmas are not duplicated
    kernel_bnds = routines['kernel_bnds']
    assert ('ydcpg_bnds', 'ydbnds') in calls['kernel_bnds'].kwarguments
    assert 'ydbnds' not in kernel_bnds.arguments
    assert 'ydcpg_bnds' in kernel_bnds.arguments
    assert 'nb' in kernel_bnds.arguments
    assert kernel_bnds.variable_map['work'].shape == ('nproma', 'nb')
    assert _pragma_contents(kernel_bnds) == [
        'unstructured-data create(work)', 'exit unstructured-data delete(work)'
    ]

    # Transformation state is stored for nested lowering in the kernels
    for name in ('kernel', 'kernel_bnds'):
        assert 'relevant_vars' in items[name].trafo_data['LowerBlockIndex']
        assert 'driver_loop' in items[name].trafo_data['LowerBlockIndex']


@pytest.mark.parametrize('frontend', available_frontends())
def test_sk_lower_block_index_nested(tmp_path, frontend, block_dim):
    """
    Test the recursive lowering of small-kernels calls in kernels, where
    arrays promoted at the previous level (with or without subscripts) span
    the block dimension, while unblocked arrays and the driver-loop
    variable are not passed on.
    """
    fcode_driver = """
subroutine driver(nproma, nb, field, coef)
  implicit none
  integer, intent(in) :: nproma, nb
  real, intent(inout) :: field(nproma, nb)
  real, intent(in) :: coef(5)
  integer :: ibl, kbl

  do ibl = 1, nb
    kbl = ibl
    !$loki small-kernels
    call kernel(nproma, field(:, ibl), coef(:))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, field, coef)
  implicit none
  integer, intent(in) :: nproma
  real, intent(inout) :: field(nproma)
  real, intent(in) :: coef(5)
  real :: ztmp(nproma)

  ztmp(1) = coef(1)
  !$loki small-kernels
  call sub_kernel(nproma, ztmp(:), coef(:), field)
  field(1) = ztmp(1)
end subroutine kernel
"""
    fcode_sub_kernel = """
subroutine sub_kernel(nproma, ztmp, coef, field)
  implicit none
  integer, intent(in) :: nproma
  real, intent(inout) :: ztmp(nproma), field(nproma)
  real, intent(in) :: coef(5)
  ztmp(1) = coef(2)
  field(1) = ztmp(1)
end subroutine sub_kernel
"""
    routines, items = _apply_sk_lowering(
        tmp_path, frontend, block_dim,
        call_tree={'driver': ('kernel',), 'kernel': ('sub_kernel',), 'sub_kernel': ()},
        fcodes={'driver': fcode_driver, 'kernel': fcode_kernel, 'sub_kernel': fcode_sub_kernel}
    )
    driver, kernel, sub_kernel = routines['driver'], routines['kernel'], routines['sub_kernel']

    # The driver call spans the block dimension, but passes neither unblocked
    # arrays nor the driver-loop variable
    call = _calls(driver)['kernel']
    assert call.arguments == ('nproma', 'field(:, :)', 'coef(:)')
    assert call.kwarguments == (('kbl', 'kbl'), ('nb', 'nb'))

    # The promoted kernel temporary and argument span the block dimension in the nested call
    assert kernel.variable_map['ztmp'].shape == ('nproma', 'nb')
    call = _calls(kernel)['sub_kernel']
    assert call.arguments == ('nproma', 'ztmp(:, :)', 'coef(:)', 'field(:, :)')
    assert call.kwarguments == (('kbl', 'kbl'), ('nb', 'nb'))

    # The sub-kernel receives the propagated arguments and transformation state
    assert sub_kernel.arguments == ('nproma', 'ztmp(nproma, nb)', 'coef(5)', 'field(nproma, nb)', 'kbl', 'nb')
    assert 'relevant_vars' in items['sub_kernel'].trafo_data['LowerBlockIndex']

    # Each callee declares the driver-loop variable locally, and unblocked arrays keep their shape
    for routine in (kernel, sub_kernel):
        assert routine.variable_map['ibl'].type.intent is None
        assert routine.variable_map['ibl'].type.dtype == BasicType.INTEGER
        assert routine.variable_map['coef'].shape == (5,)


@pytest.mark.parametrize('frontend', available_frontends())
def test_sk_lower_block_index_derived_type_bounds(tmp_path, frontend):
    """
    Test the lowering with a derived-type block size and derived-type
    loop-header variables, where the root object is passed once together
    with indices used in its parents, and declared before its first use
    in the bounds of other dummy arguments.
    """
    fcode_type_mod = """
module geom_mod
  implicit none
  type :: bnds_type
    integer :: kidia
  end type bnds_type
  type :: yrdim_type
    integer :: ngpblks
  end type yrdim_type
  type :: geometry
    type(yrdim_type) :: yrdim
    type(bnds_type) :: bnds(4)
  end type geometry
end module geom_mod
"""
    fcode_driver = """
subroutine driver(nproma, nb, geom, field, jj)
  use geom_mod, only: geometry
  implicit none
  integer, intent(in) :: nproma, nb, jj
  type(geometry), intent(in) :: geom
  real, intent(inout) :: field(nproma, geom%yrdim%ngpblks)
  integer :: ibl, jkglo, kidia

  do ibl = 1, nb
    jkglo = geom%yrdim%ngpblks
    kidia = geom%bnds(jj)%kidia
    !$loki small-kernels
    call kernel(nproma, field(:, ibl))
  end do
end subroutine driver
"""
    fcode_kernel = """
subroutine kernel(nproma, field)
  implicit none
  integer, intent(in) :: nproma
  real, intent(inout) :: field(nproma)
  field(1) = 0.0
end subroutine kernel
"""
    geom_block_dim = Dimension(
        name='block_dim', size='geom%yrdim%ngpblks', index='ibl',
        bounds=('1', 'geom%yrdim%ngpblks'), aliases=('jkglo',)
    )
    routines, _ = _apply_sk_lowering(
        tmp_path, frontend, geom_block_dim,
        call_tree={'driver': ('kernel',), 'kernel': ()},
        fcodes={'driver': fcode_driver, 'kernel': fcode_kernel},
        definitions=(Module.from_source(fcode_type_mod, frontend=frontend, xmods=[tmp_path]),)
    )
    kernel = routines['kernel']

    # The root object is passed once, together with the index used in its parents
    call = _calls(routines['driver'])['kernel']
    assert call.kwarguments == (
        ('geom', 'geom'), ('jj', 'jj'), ('jkglo', 'jkglo'), ('kidia', 'kidia'), ('nb', 'nb')
    )
    assert kernel.arguments == (
        'nproma', 'field(nproma, geom%yrdim%ngpblks)', 'geom', 'jj', 'jkglo', 'kidia', 'nb'
    )
    assert kernel.variable_map['field'].shape == ('nproma', 'geom%yrdim%ngpblks')

    # The new derived-type dummy is declared before its use in the bounds of ``field``
    decls = [decl for decl in FindNodes(ir.VariableDeclaration).visit(kernel.spec) if len(decl.symbols) == 1]
    decl_pos = {decl.symbols[0].name.lower(): idx for idx, decl in enumerate(decls)}
    assert decl_pos['geom'] < decl_pos['field']


@pytest.mark.parametrize('frontend', available_frontends())
def test_inject_block_index_per_loop(tmp_path, frontend, block_dim):
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
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
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
