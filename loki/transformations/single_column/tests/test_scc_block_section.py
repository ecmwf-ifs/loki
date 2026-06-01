# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :mod:`loki.transformations.single_column.block`.
"""

import pytest

from loki import Dimension, Sourcefile, fgen
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, Transformer, pragmas_attached,
)
from loki.tools import CaseInsensitiveDict

from loki.transformations.single_column.block import (
    SCCBlockSectionTransformation,
    SCCBlockSectionToLoopTransformation,
    ReblockSectionTransformer,
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture
def block_dim():
    return Dimension(
        name='block_dim', size='nb', index='ibl',
        bounds=('1', 'nb'),
        aliases=('bounds%kbl', 'block_global'),
    )


@pytest.fixture
def horizontal():
    return Dimension(
        name='horizontal', size='nproma', index='jl',
        bounds=('1', 'nproma'),
    )


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _make_item(routine, role='kernel', targets=(), config=None):
    """Create a minimal ProcedureItem-like mock for testing."""

    class _MockItem:
        # pylint: disable=too-few-public-methods
        def __init__(self, routine, role, targets, config):
            self.routine = routine
            self.role = role
            self.targets = targets
            self.local_name = routine.name.lower()
            self.trafo_data = {}
            self.config = config or {}

    return _MockItem(routine, role, targets, config)


# ----------------------------------------------------------------
# Tests for SCCBlockSectionTransformation
# ----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_block_section_extraction(frontend, block_dim):
    """
    Verify that extract_block_sections correctly identifies contiguous
    sections containing block-dimension variables, split at calls
    annotated with ``!$loki small-kernels``.
    """
    fcode_kernel = """
subroutine compute_phys(bounds, nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl, jl
  type(bounds_type), intent(in) :: bounds
  real :: zfld(nproma, nb)
  real :: ztmp(nproma)

  ! Pre-computation (references block dim)
  zfld(1, ibl) = 1.0

  ! Non-block-dim computation (should be excluded)
  ztmp(1) = 2.0

  ! Post-computation (references block dim)
  zfld(2, ibl) = 3.0
end subroutine compute_phys
    """.strip()

    source = Sourcefile.from_source(fcode_kernel, frontend=frontend)
    routine = source['compute_phys']

    successor_map = CaseInsensitiveDict()
    sections = SCCBlockSectionTransformation.extract_block_sections(
        routine.body.body, block_dim, successor_map
    )

    # All block-dim-referencing nodes should be captured in at least
    # one section (the extraction captures contiguous regions that
    # contain block-dim references; non-block-dim nodes like ztmp may
    # be included in the same contiguous region)
    all_nodes = [n for sec in sections for n in sec]
    assigns = [n for n in all_nodes if isinstance(n, ir.Assignment)]
    assign_lhs_names = [str(a.lhs) for a in assigns]
    assert any('zfld' in name.lower() for name in assign_lhs_names)

    # There should be exactly one section (no separators to split on)
    assert len(sections) == 1


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_block_section_trimming(frontend, block_dim):
    """
    Verify that get_trimmed_sections trims extracted sections to only
    include nodes that reference block-dimension symbols.
    """
    fcode = """
subroutine trim_test(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: za(nproma, nb)
  real :: ztmp

  ! This should be trimmed (before first block-dim ref)
  ztmp = 42.0

  ! This should remain (references ibl)
  za(1, ibl) = 1.0

  ! This should remain (references ibl)
  za(2, ibl) = 2.0

  ! This should be trimmed (after last block-dim ref)
  ztmp = 99.0
end subroutine trim_test
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['trim_test']

    # Build a single section from the full body
    sections = [routine.body.body]
    trimmed = SCCBlockSectionTransformation.get_trimmed_sections(
        routine, block_dim, sections
    )

    assert len(trimmed) >= 1
    # The trimmed section should span from the first to the last
    # block-dim node, excluding leading/trailing non-block-dim nodes
    all_nodes = [n for sec in trimmed for n in sec]
    assigns = [n for n in all_nodes if isinstance(n, ir.Assignment)]
    assign_lhs = [str(a.lhs).lower() for a in assigns]
    # Both za assignments should be present
    assert assign_lhs.count('za(1, ibl)') == 1
    assert assign_lhs.count('za(2, ibl)') == 1
    # The leading ztmp=42 and trailing ztmp=99 should be trimmed away
    assert assign_lhs.count('ztmp') == 0


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_block_section_trimming_keeps_setup_and_data_pragmas_outside(frontend, block_dim):
    """
    Leading scalar setup and explicit data-management pragmas must stay
    outside trimmed block sections.
    """
    fcode = """
subroutine trim_setup(nproma, nb, flag)
  implicit none
  integer, intent(in) :: nproma, nb
  logical, intent(in) :: flag
  integer :: ibl
  character(len=4) :: mode_flag
  real :: za(nproma, nb)
  real :: work(nproma)

  mode_flag = 'IBOT'
  if (flag) mode_flag = 'INTG'
  !$loki unstructured-data create(work)
  za(1, ibl) = 1.0
  !$loki exit unstructured-data delete(work)
end subroutine trim_setup
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['trim_setup']

    trimmed = SCCBlockSectionTransformation.get_trimmed_sections(
        routine, block_dim, [routine.body.body]
    )

    assert len(trimmed) == 1
    nodes = trimmed[0]
    assert len(nodes) == 1
    assert isinstance(nodes[0], ir.Assignment)
    assert nodes[0].lhs == 'za(1, ibl)'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_driver_loop_unwrap(frontend, block_dim):
    """
    Verify that driver loops containing ``!$loki small-kernels``
    pragma calls are unwrapped (replaced with body + comment markers).
    """
    fcode_kernel = """
subroutine kernel_phys(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  real :: z(nproma)
  z(1) = 1.0
end subroutine kernel_phys
    """.strip()

    fcode_driver = """
subroutine driver_phys(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: z(nproma, nb)

  do ibl = 1, nb
    z(1, ibl) = 0.0
    !$loki small-kernels
    call kernel_phys(nproma, nb)
  end do
end subroutine driver_phys
    """.strip()

    source = Sourcefile.from_source(
        fcode_driver + '\n' + fcode_kernel, frontend=frontend
    )
    driver = source['driver_phys']
    kernel = source['kernel_phys']

    driver.enrich(kernel)

    driver_item = _make_item(
        driver, role='driver', targets=('kernel_phys',)
    )
    kernel_item = _make_item(kernel, role='kernel')

    successor_map = CaseInsensitiveDict({
        'kernel_phys': kernel_item,
    })

    trafo = SCCBlockSectionTransformation(block_dim=block_dim)
    trafo.process_driver(
        driver, driver_item, successor_map,
        targets=('kernel_phys',)
    )

    # The driver loop should have been unwrapped
    loops = FindNodes(ir.Loop).visit(driver.body)
    assert len(loops) == 0

    # Comment markers should be present
    comments = FindNodes(ir.Comment).visit(driver.body)
    comment_texts = [c.text for c in comments]
    assert any('former driver loop' in t for t in comment_texts)
    assert any('END: former driver loop' in t for t in comment_texts)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_driver_marks_successors(frontend, block_dim):
    """
    Verify that ``BlockSectionTrafo`` flag is set in the successor's
    ``trafo_data`` when the driver processes ``!$loki small-kernels``
    pragma calls.
    """
    fcode_kernel = """
subroutine cpg_gp(nproma)
  implicit none
  integer, intent(in) :: nproma
end subroutine cpg_gp
    """.strip()

    fcode_driver = """
subroutine cpg_0(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl

  do ibl = 1, nb
    !$loki small-kernels
    call cpg_gp(nproma)
  end do
end subroutine cpg_0
    """.strip()

    source = Sourcefile.from_source(
        fcode_driver + '\n' + fcode_kernel, frontend=frontend
    )
    driver = source['cpg_0']
    kernel = source['cpg_gp']

    driver.enrich(kernel)

    driver_item = _make_item(
        driver, role='driver', targets=('cpg_gp',)
    )
    kernel_item = _make_item(kernel, role='kernel')

    successor_map = CaseInsensitiveDict({
        'cpg_gp': kernel_item,
    })

    trafo = SCCBlockSectionTransformation(block_dim=block_dim)
    trafo.process_driver(
        driver, driver_item, successor_map,
        targets=('cpg_gp',)
    )

    assert kernel_item.trafo_data.get('BlockSectionTrafo') is True


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_kernel_skipped_without_trafo_data(frontend, block_dim):
    """
    Verify that kernel routines without ``BlockSectionTrafo`` in
    ``trafo_data`` are left untouched.
    """
    fcode = """
subroutine untouched_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: za(nproma, nb)
  za(1, ibl) = 1.0
end subroutine untouched_kernel
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['untouched_kernel']
    item = _make_item(routine, role='kernel')

    # No BlockSectionTrafo set
    successor_map = CaseInsensitiveDict()

    trafo = SCCBlockSectionTransformation(block_dim=block_dim)
    body_before = routine.body

    trafo.process_kernel(routine, item, successor_map)

    # Body should be unchanged
    assert routine.body is body_before


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_reblock_wraps_with_driver_loop(frontend, horizontal):
    """
    Verify that ReblockSectionTransformer wraps ``block_section``
    labelled sections with a cloned driver loop and
    ``!$loki loop driver`` pragma.
    """
    fcode = """
subroutine reblock_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: za(nproma, nb)
  za(1, ibl) = 1.0
end subroutine reblock_kernel
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['reblock_kernel']

    # Create a mock driver loop
    ibl_var = routine.variable_map.get('ibl')
    nb_var = routine.variable_map.get('nb')
    driver_loop = ir.Loop(
        variable=ibl_var,
        bounds=sym.LoopRange(
            (sym.IntLiteral(1), nb_var)
        ),
        body=()
    )

    item = _make_item(routine, role='kernel')
    item.trafo_data['LowerBlockIndex'] = {'driver_loop': driver_loop}

    # Manually wrap an assignment in a block_section
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) >= 1

    section = ir.Section(body=(assigns[0],), label='block_section')
    routine.body = Transformer({assigns[0]: section}).visit(routine.body)

    # Apply ReblockSectionTransformer
    routine.body = ReblockSectionTransformer(
        routine, item, horizontal
    ).visit(routine.body)

    # Should now have a loop
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) >= 1

    # The loop should have a !$loki loop driver pragma with vector_length
    loop = loops[0]
    assert loop.pragma is not None
    pragma_contents = [p.content for p in loop.pragma]
    assert any('loop driver' in c for c in pragma_contents)
    assert any('vector_length(nproma)' in c for c in pragma_contents)

    # Comment markers should be present
    comments = FindNodes(ir.Comment).visit(routine.body)
    comment_texts = [c.text for c in comments]
    assert any('START of Loki inserted block loop' in t for t in comment_texts)
    assert any('END of Loki inserted block loop' in t for t in comment_texts)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_activate_pragmas(frontend):
    """
    Verify that ``!$loki inactive-small-kernels`` pragmas are
    activated by stripping the prefix.
    """
    fcode = """
subroutine pragma_kernel(nproma)
  implicit none
  integer, intent(in) :: nproma
  real :: z(nproma)

  !$loki inactive-small-kernels data present(z)
  z(1) = 1.0
  !$loki inactive-small-kernels end data
end subroutine pragma_kernel
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['pragma_kernel']

    SCCBlockSectionToLoopTransformation.activate_pragmas(routine)

    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    for pragma in pragmas:
        assert 'inactive-small-kernels' not in pragma.content
    # The content should still have the actual directive
    pragma_contents = [p.content.strip() for p in pragmas]
    assert any('data present' in c for c in pragma_contents)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_create_local_copies_block_indices(frontend, horizontal):
    """
    Verify that local copies are created for derived-type block-dimension
    index variables with ``local_X = X`` assignments prepended.
    """
    # Use a block_dim whose indices include the derived-type member
    block_dim_with_dt = Dimension(
        name='block_dim', size='nb',
        index=('ibl', 'bounds%kbl'),
        bounds=('1', 'nb'),
    )

    fcode = """
subroutine localcopy_kernel(bounds, nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  type :: bounds_t
    integer :: kbl
  end type bounds_t
  type(bounds_t), intent(in) :: bounds
  real :: za(nproma, nb)

  za(1, bounds%kbl) = 1.0
end subroutine localcopy_kernel
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['localcopy_kernel']

    trafo = SCCBlockSectionToLoopTransformation(
        block_dim=block_dim_with_dt, horizontal=horizontal
    )
    trafo._create_local_copies(routine)

    # Should have a local_bounds variable (the parent of the
    # derived-type member bounds%kbl is localized)
    var_names = [v.name.lower() for v in routine.variables]
    assert 'local_bounds' in var_names

    # Should have a local_bounds = bounds assignment at top
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    local_assigns = [
        a for a in assigns
        if 'local_' in str(a.lhs).lower()
    ]
    assert len(local_assigns) >= 1
    assert local_assigns[0].lhs == 'local_bounds'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_block_section_nested_conditional(frontend, block_dim):
    """
    Verify that extract_block_sections recursively processes
    conditional bodies to find nested block sections.
    """
    fcode_kernel = """
subroutine nested_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: za(nproma, nb)
  logical :: lflag

  if (lflag) then
    za(1, ibl) = 1.0
    !$loki small-kernels
    call sub_kernel(nproma, nb)
    za(2, ibl) = 2.0
  end if
end subroutine nested_kernel
    """.strip()

    fcode_sub = """
subroutine sub_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
end subroutine sub_kernel
    """.strip()

    source = Sourcefile.from_source(
        fcode_kernel + '\n' + fcode_sub, frontend=frontend
    )
    routine = source['nested_kernel']
    sub = source['sub_kernel']
    routine.enrich(sub)

    sub_item = _make_item(sub, role='kernel')
    successor_map = CaseInsensitiveDict({
        'sub_kernel': sub_item,
    })

    with pragmas_attached(routine, ir.CallStatement):
        sections = SCCBlockSectionTransformation.extract_block_sections(
            routine.body.body, block_dim, successor_map
        )

    # Should find sections inside the conditional body
    assert len(sections) >= 1
    # sub_kernel should be marked
    assert sub_item.trafo_data.get('BlockSectionTrafo') is True


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_block_section_nested_conditional_array_ref_block_index(frontend, block_dim):
    """
    Reproduce a nested-call pattern where block-index usage appears only
    inside array references in pre/post-call loop regions.
    """
    block_dim_with_members = Dimension(
        name='block_dim',
        size=('GEOMETRY%DIMS%NBLOCKS', 'GEOM_INFO%DIMS%NBLOCKS'),
        index=('IBL', 'BOUNDS%KBL', 'LOCAL_BOUNDS%KBL', 'BLOCK_GLOBAL')
    )

    fcode_kernel = """
subroutine array_ref_block_kernel(nproma, kst, kend, klev, bounds)
  implicit none
  type bounds_type
    integer :: kbl
  end type bounds_type
  integer, intent(in) :: nproma, kst, kend, klev
  type(bounds_type), intent(in) :: bounds
  integer :: jrof, jlev
  logical :: lflag
  real :: zphi(nproma, 0:klev+1, 10)
  real :: phi(nproma, 0:klev, 10)
  real :: phif(nproma, klev, 10)
  real :: pt(nproma, klev, 10)
  real :: pr(nproma, klev, 10)
  real :: plnpr(nproma, klev, 10)

  if (lflag) then
    do jrof = kst, kend
      do jlev = 1, klev
        zphi(jrof, jlev, bounds%kbl) = pr(jrof, jlev, bounds%kbl) * pt(jrof, jlev, bounds%kbl)
      end do
      zphi(jrof, 0, bounds%kbl) = 0.0
      zphi(jrof, klev + 1, bounds%kbl) = 0.0
    end do

    !$loki small-kernels
    call child_kernel(nproma, klev, zphi, phif)

    do jrof = kst, kend
      do jlev = klev, 1, -1
        phi(jrof, jlev - 1, bounds%kbl) = phi(jrof, jlev, bounds%kbl) + pr(jrof, jlev, bounds%kbl) * pt(jrof, jlev, bounds%kbl) * plnpr(jrof, jlev, bounds%kbl)
      end do
    end do
  end if
end subroutine array_ref_block_kernel
    """.strip()

    fcode_child = """
subroutine child_kernel(nproma, klev, zphi, phif)
  implicit none
  integer, intent(in) :: nproma, klev
  real, intent(in) :: zphi(nproma, 0:klev+1, 10)
  real, intent(out) :: phif(nproma, klev, 10)
end subroutine child_kernel
    """.strip()

    source = Sourcefile.from_source(
        fcode_kernel + '\n' + fcode_child, frontend=frontend
    )
    routine = source['array_ref_block_kernel']
    child = source['child_kernel']
    routine.enrich(child)

    child_item = _make_item(child, role='kernel')
    successor_map = CaseInsensitiveDict({
        'child_kernel': child_item,
    })

    with pragmas_attached(routine, ir.CallStatement):
        sections = SCCBlockSectionTransformation.extract_block_sections(
            routine.body.body, block_dim_with_members, successor_map
        )

    assert len(sections) == 2
    assert child_item.trafo_data.get('BlockSectionTrafo') is True

    for sec in sections:
        vars_in_sec = tuple(FindVariables().visit(sec))
        assert 'BOUNDS%KBL' in vars_in_sec
        assert any('bounds%kbl' in str(var).lower() for var in vars_in_sec)

    section_code = [tuple(fgen(node).lower() for node in sec) for sec in sections]
    assert any(any('zphi(' in node for node in sec) for sec in section_code)
    assert any(any('phi(' in node for node in sec) for sec in section_code)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_block_section_separator_at_call(frontend, block_dim):
    """
    Verify that calls annotated with ``!$loki small-kernels`` act as
    separator nodes, splitting the IR into separate block sections.
    """
    fcode_kernel = """
subroutine split_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: za(nproma, nb), zb(nproma, nb)

  ! Section 1
  za(1, ibl) = 1.0

  !$loki small-kernels
  call child_kernel(nproma, nb)

  ! Section 2
  zb(1, ibl) = 2.0
end subroutine split_kernel
    """.strip()

    fcode_child = """
subroutine child_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
end subroutine child_kernel
    """.strip()

    source = Sourcefile.from_source(
        fcode_kernel + '\n' + fcode_child, frontend=frontend
    )
    routine = source['split_kernel']
    child = source['child_kernel']
    routine.enrich(child)

    child_item = _make_item(child, role='kernel')
    successor_map = CaseInsensitiveDict({
        'child_kernel': child_item,
    })

    with pragmas_attached(routine, ir.CallStatement):
        sections = SCCBlockSectionTransformation.extract_block_sections(
            routine.body.body, block_dim, successor_map
        )

    # The call should split the body into at least 2 sections
    assert len(sections) >= 2

    # Each section should contain block-dim references
    for sec in sections:
        all_vars = list(FindVariables().visit(sec))
        var_names = [str(v).lower() for v in all_vars]
        assert any('ibl' in name for name in var_names)


# ----------------------------------------------------------------
# Additional coverage tests
# ----------------------------------------------------------------

@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_process_kernel_wraps_block_sections(frontend, block_dim):
    """
    End-to-end test for ``process_kernel``: verify that block-dimension
    computation regions are wrapped in ``Section(label='block_section')``
    nodes in the resulting IR.
    """
    fcode_kernel = """
subroutine phys_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: za(nproma, nb), zb(nproma, nb)

  za(1, ibl) = 1.0
  !$loki small-kernels
  call sub_phys(nproma, nb)
  zb(1, ibl) = 2.0
end subroutine phys_kernel
    """.strip()

    fcode_sub = """
subroutine sub_phys(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
end subroutine sub_phys
    """.strip()

    source = Sourcefile.from_source(
        fcode_kernel + '\n' + fcode_sub, frontend=frontend
    )
    routine = source['phys_kernel']
    sub = source['sub_phys']
    routine.enrich(sub)

    item = _make_item(routine, role='kernel')
    item.trafo_data['BlockSectionTrafo'] = True
    sub_item = _make_item(sub, role='kernel')

    successor_map = CaseInsensitiveDict({
        'sub_phys': sub_item,
    })

    trafo = SCCBlockSectionTransformation(block_dim=block_dim)
    trafo.process_kernel(routine, item, successor_map)


    # There should be Section nodes labelled 'block_section'
    sections = [
        s for s in FindNodes(ir.Section).visit(routine.body)
        if s.label == 'block_section'
    ]
    assert len(sections) >= 2

    # Each block_section should contain an assignment
    for sec in sections:
        assigns = FindNodes(ir.Assignment).visit(sec)
        assert len(assigns) >= 1


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_process_kernel_removes_loki_routine_pragma(frontend, block_dim):
    """
    Verify that ``!$loki routine seq`` pragmas are removed from the
    kernel spec and body during ``process_kernel``.
    """
    fcode = """
subroutine seq_kernel(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl
  real :: za(nproma, nb)
  !$loki routine seq

  za(1, ibl) = 1.0
end subroutine seq_kernel
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['seq_kernel']

    item = _make_item(routine, role='kernel')
    item.trafo_data['BlockSectionTrafo'] = True
    successor_map = CaseInsensitiveDict()

    # Verify pragma exists before
    pragmas_before = [
        p for p in FindNodes(ir.Pragma).visit(routine.ir)
        if p.keyword.lower() == 'loki' and 'routine' in p.content.lower()
    ]
    assert len(pragmas_before) >= 1

    trafo = SCCBlockSectionTransformation(block_dim=block_dim)
    trafo.process_kernel(routine, item, successor_map)

    # Verify pragma is removed after
    pragmas_after = [
        p for p in FindNodes(ir.Pragma).visit(routine.ir)
        if p.keyword.lower() == 'loki' and 'routine' in p.content.lower()
    ]
    assert len(pragmas_after) == 0


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_reblock_raises_without_trafo_data(frontend, horizontal):
    """
    Verify that ``ReblockSectionTransformer`` raises ``RuntimeError``
    when ``LowerBlockIndex`` is missing from ``item.trafo_data``.
    """
    fcode = """
subroutine raise_kernel(nproma)
  implicit none
  integer, intent(in) :: nproma
end subroutine raise_kernel
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['raise_kernel']

    item = _make_item(routine, role='kernel')
    # No LowerBlockIndex in trafo_data

    with pytest.raises(RuntimeError, match='LowerBlockIndex'):
        ReblockSectionTransformer(routine, item, horizontal)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full module header')]
))
def test_driver_preserves_non_sk_loops(frontend, block_dim):
    """
    Verify that driver loops without ``!$loki small-kernels`` pragma
    calls are preserved (not unwrapped).
    """
    fcode_kernel = """
subroutine keep_kernel(nproma)
  implicit none
  integer, intent(in) :: nproma
end subroutine keep_kernel
    """.strip()

    fcode_driver = """
subroutine keep_driver(nproma, nb)
  implicit none
  integer, intent(in) :: nproma, nb
  integer :: ibl

  ! This loop has no small-kernels pragma — should be preserved
  do ibl = 1, nb
    call keep_kernel(nproma)
  end do
end subroutine keep_driver
    """.strip()

    source = Sourcefile.from_source(
        fcode_driver + '\n' + fcode_kernel, frontend=frontend
    )
    driver = source['keep_driver']
    kernel = source['keep_kernel']
    driver.enrich(kernel)

    driver_item = _make_item(
        driver, role='driver', targets=('keep_kernel',)
    )
    kernel_item = _make_item(kernel, role='kernel')
    successor_map = CaseInsensitiveDict({
        'keep_kernel': kernel_item,
    })

    trafo = SCCBlockSectionTransformation(block_dim=block_dim)
    trafo.process_driver(
        driver, driver_item, successor_map,
        targets=('keep_kernel',)
    )

    # The loop should still be there
    loops = FindNodes(ir.Loop).visit(driver.body)
    assert len(loops) == 1

    # No BlockSectionTrafo should be set on the successor
    assert not kernel_item.trafo_data.get('BlockSectionTrafo', False)
