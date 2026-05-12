# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for :any:`CreateLocalCopiesTransformation`.
"""

from unittest.mock import Mock

import pytest

from loki import Dimension, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import FindNodes, FindVariables, Pragma

from loki.transformations.single_column.local_copies import CreateLocalCopiesTransformation


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('klon',)
    )


@pytest.fixture(scope='module', name='horizontal_derived')
def fixture_horizontal_derived():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('bnds%start', 'bnds%end'), aliases=('klon',)
    )


@pytest.fixture(scope='module', name='block_dim')
def fixture_block_dim():
    return Dimension(name='block_dim', size='nb', index='ibl',
                     aliases=('nblks',))


def _make_item(trafo_data=None):
    """Create a mock scheduler Item with the given trafo_data."""
    item = Mock()
    item.trafo_data = trafo_data if trafo_data is not None else {}
    return item


# ---------------------------------------------------------------------------
# get_block_index tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_get_block_index_plain(tmp_path, frontend):
    """get_block_index resolves a plain variable name."""
    fcode = """
subroutine kernel(ibl, field)
  implicit none
  integer, intent(in) :: ibl
  real, intent(inout) :: field(10)
  field(1) = real(ibl)
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    var = CreateLocalCopiesTransformation.get_block_index(
        routine, routine.variable_map, 'ibl'
    )
    assert var is not None
    assert var == 'ibl'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_get_block_index_derived_type(tmp_path, frontend):
    """get_block_index resolves a %-separated derived-type path."""
    fcode = """
subroutine kernel(dims, field)
  implicit none
  type :: dim_t
    integer :: ibl
  end type
  type(dim_t), intent(in) :: dims
  real, intent(inout) :: field(10)
  field(1) = real(dims%ibl)
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    var = CreateLocalCopiesTransformation.get_block_index(
        routine, routine.variable_map, 'dims%ibl'
    )
    assert var is not None
    assert var == 'dims%ibl'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_get_block_index_missing(tmp_path, frontend):
    """get_block_index returns None for unknown index."""
    fcode = """
subroutine kernel(field)
  implicit none
  real, intent(inout) :: field(10)
  field(1) = 1.0
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    var = CreateLocalCopiesTransformation.get_block_index(
        routine, routine.variable_map, 'ibl'
    )
    assert var is None


# ---------------------------------------------------------------------------
# _create_local_copies tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_create_local_copies_block_index(tmp_path, frontend, horizontal, block_dim):
    """Local copy is created for block-dim index and horizontal bounds."""
    fcode = """
subroutine kernel(ibl, start, end, field)
  implicit none
  integer, intent(in) :: ibl, start, end
  real, intent(inout) :: field(10)
  integer :: jl

  do jl = start, end
    field(jl) = field(jl) + real(ibl)
  end do
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal)
    trafo._create_local_copies(routine)

    # Check local copies exist
    var_names = {v.name.lower() for v in routine.variables}
    assert 'local_ibl' in var_names
    assert 'local_start' in var_names
    assert 'local_end' in var_names

    # Check body references are substituted
    body_vars = {v.name.lower() for v in FindVariables().visit(routine.body)}
    assert 'local_ibl' in body_vars
    assert 'local_start' in body_vars
    assert 'local_end' in body_vars


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_create_local_copies_excludes_sizes(tmp_path, frontend, block_dim):
    """horizontal.sizes vars are excluded from local-copy creation."""
    horizontal = Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'nlon'), aliases=('klon',)
    )
    fcode = """
subroutine kernel(ibl, start, nlon, field)
  implicit none
  integer, intent(in) :: ibl, start, nlon
  real, intent(inout) :: field(nlon)
  integer :: jl

  do jl = start, nlon
    field(jl) = field(jl) + real(ibl)
  end do
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal)
    trafo._create_local_copies(routine)

    var_names = {v.name.lower() for v in routine.variables}
    # nlon appears in both horizontal.upper and horizontal.sizes — should NOT be localized
    assert 'local_nlon' not in var_names
    # start and ibl should be localized
    assert 'local_start' in var_names
    assert 'local_ibl' in var_names


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_create_local_copies_dedup(tmp_path, frontend, horizontal, block_dim):
    """If a local_X already exists, _create_local_copies skips it."""
    fcode = """
subroutine kernel(ibl, start, end, field)
  implicit none
  integer, intent(in) :: ibl, start, end
  real, intent(inout) :: field(10)
  integer :: local_ibl
  integer :: jl

  local_ibl = ibl
  do jl = start, end
    field(jl) = field(jl) + real(ibl)
  end do
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal)
    trafo._create_local_copies(routine)

    # local_ibl already existed so should not be duplicated
    local_ibl_vars = [v for v in routine.variables if v.name.lower() == 'local_ibl']
    assert len(local_ibl_vars) == 1

    # But local copies for bounds should still be created
    var_names = {v.name.lower() for v in routine.variables}
    assert 'local_start' in var_names
    assert 'local_end' in var_names


# ---------------------------------------------------------------------------
# _filter_device_present_pragma tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_filter_device_present_removes_vars(tmp_path, frontend):
    """Variables are removed from device-present pragma."""
    fcode = """
subroutine kernel(ibl, bnds, field)
  implicit none
  type :: bounds_t
    integer :: start
  end type
  integer, intent(in) :: ibl
  type(bounds_t), intent(in) :: bnds
  real, intent(inout) :: field(10)

  !$loki device-present vars(field, ibl, bnds)
  field(1) = real(ibl)
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    CreateLocalCopiesTransformation._filter_device_present_pragma(
        routine, {'ibl', 'bnds'}
    )

    pragmas = [p for p in FindNodes(Pragma).visit(routine.body)
               if p.keyword.lower() == 'loki' and 'device-present' in p.content]
    assert len(pragmas) == 1
    content = pragmas[0].content
    assert 'field' in content
    assert 'ibl' not in content
    assert 'bnds' not in content


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_filter_device_present_removes_all(tmp_path, frontend):
    """If all vars are removed, pragma content becomes just 'device-present'."""
    fcode = """
subroutine kernel(ibl, field)
  implicit none
  integer, intent(in) :: ibl
  real, intent(inout) :: field(10)

  !$loki device-present vars(ibl)
  field(1) = real(ibl)
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    CreateLocalCopiesTransformation._filter_device_present_pragma(
        routine, {'ibl'}
    )

    pragmas = [p for p in FindNodes(Pragma).visit(routine.body)
               if p.keyword.lower() == 'loki' and 'device-present' in p.content]
    assert len(pragmas) == 1
    assert pragmas[0].content.strip() == 'device-present'


# ---------------------------------------------------------------------------
# _get_bounds_parent_names tests
# ---------------------------------------------------------------------------


def test_get_bounds_parent_names(horizontal_derived, block_dim):
    """Parent names from %-separated dimension entries are extracted."""
    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal_derived)
    parents = trafo._get_bounds_parent_names()
    # horizontal has bounds=('bnds%start', 'bnds%end'), so 'bnds' should be present
    assert 'bnds' in parents


def test_get_bounds_parent_names_no_derived():
    """When no %-separated entries exist, result is empty."""
    block_dim = Dimension(name='block_dim', size='nb', index='ibl')
    horizontal = Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end')
    )
    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal)
    parents = trafo._get_bounds_parent_names()
    assert len(parents) == 0


# ---------------------------------------------------------------------------
# _remove_bounds_parents_from_device_present tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_remove_bounds_parents(tmp_path, frontend, horizontal_derived, block_dim):
    """Bounds-parent vars are removed from device-present pragmas."""
    fcode = """
subroutine kernel(bnds, field)
  implicit none
  type :: bounds_t
    integer :: start
    integer :: end
  end type
  type(bounds_t), intent(in) :: bnds
  real, intent(inout) :: field(10)

  !$loki device-present vars(field, bnds)
  field(1) = 1.0
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal_derived)
    trafo._remove_bounds_parents_from_device_present(routine)

    pragmas = [p for p in FindNodes(Pragma).visit(routine.body)
               if p.keyword.lower() == 'loki' and 'device-present' in p.content]
    assert len(pragmas) == 1
    assert 'bnds' not in pragmas[0].content
    assert 'field' in pragmas[0].content


# ---------------------------------------------------------------------------
# transform_subroutine integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_transform_subroutine_kernel_with_lower_block_index(
    tmp_path, frontend, horizontal, block_dim
):
    """Full transform_subroutine for a kernel with LowerBlockIndex data."""
    fcode = """
subroutine kernel(ibl, start, end, field)
  implicit none
  integer, intent(in) :: ibl, start, end
  real, intent(inout) :: field(10)
  integer :: jl

  !$loki device-present vars(field, ibl)
  do jl = start, end
    field(jl) = field(jl) + real(ibl)
  end do
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    item = _make_item({'LowerBlockIndex': True})

    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal)
    trafo.transform_subroutine(routine, role='kernel', item=item)

    var_names = {v.name.lower() for v in routine.variables}
    assert 'local_ibl' in var_names
    assert 'local_start' in var_names
    assert 'local_end' in var_names

    # device-present should have ibl removed
    pragmas = [p for p in FindNodes(Pragma).visit(routine.body)
               if p.keyword.lower() == 'loki' and 'device-present' in p.content]
    assert len(pragmas) == 1
    assert 'ibl' not in pragmas[0].content
    assert 'field' in pragmas[0].content


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_transform_subroutine_kernel_without_lower_block_index(
    tmp_path, frontend, horizontal_derived, block_dim
):
    """Without LowerBlockIndex data, only bounds-parents are cleaned."""
    fcode = """
subroutine kernel(bnds, field)
  implicit none
  type :: bounds_t
    integer :: start
    integer :: end
  end type
  type(bounds_t), intent(in) :: bnds
  real, intent(inout) :: field(10)

  !$loki device-present vars(field, bnds)
  field(1) = 1.0
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    item = _make_item({})  # No LowerBlockIndex

    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal_derived)
    trafo.transform_subroutine(routine, role='kernel', item=item)

    # No local copies should be created
    var_names = {v.name.lower() for v in routine.variables}
    assert 'local_start' not in var_names

    # But bnds should still be removed from device-present (it's a bounds parent)
    pragmas = [p for p in FindNodes(Pragma).visit(routine.body)
               if p.keyword.lower() == 'loki' and 'device-present' in p.content]
    assert len(pragmas) == 1
    assert 'bnds' not in pragmas[0].content
    assert 'field' in pragmas[0].content


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI parser not reliably available')]
))
def test_transform_subroutine_driver_skipped(
    tmp_path, frontend, horizontal, block_dim
):
    """Driver role is not processed."""
    fcode = """
subroutine driver(ibl, field)
  implicit none
  integer, intent(in) :: ibl
  real, intent(inout) :: field(10)
  field(1) = real(ibl)
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    item = _make_item({'LowerBlockIndex': True})

    trafo = CreateLocalCopiesTransformation(block_dim=block_dim, horizontal=horizontal)
    trafo.transform_subroutine(routine, role='driver', item=item)

    var_names = {v.name.lower() for v in routine.variables}
    assert 'local_ibl' not in var_names
