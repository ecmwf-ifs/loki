# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Module, Dimension
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes

from loki.transformations.parallel import (
    remove_field_api_view_updates, add_field_api_view_updates
)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full type definitions for derived types')]
))
def test_field_api_remove_view_updates(frontend):
    """
    A simple test for :any:`remove_field_api_view_updates`
    """

    fcode = """
subroutine test_remove_block_loop(ngptot, nproma, nflux, dims, state, aux_fields, fluxes)
  implicit none
  integer(kind=4), intent(in) :: ngptot, nproma, nflux
  type(dimension_type), intent(inout) :: dims
  type(state_type), intent(inout) :: state
  type(aux_type), intent(inout) :: aux_fields
  type(flux_type), intent(inout) :: fluxes(nflux)

  integer :: JKGLO, IBL, ICEND, JK, JL, JF

  DO jkglo=1, ngptot, nproma
    icend = min(nproma, ngptot - JKGLO + 1)
    ibl = (jkglo - 1) / nproma + 1

    CALL DIMS%UPDATE(IBL, ICEND, JKGLO)
    CALL STATE%UPDATE_VIEW(IBL)
    CALL AUX_FIELDS%UPDATE_VIEW(block_index=IBL)
    DO jf=1, nflux
      CALL FLUXES(JF)%UPDATE_VIEW(IBL)
    END DO

    CALL MY_KERNEL(STATE%U, STATE%V, AUX_FIELDS%STUFF, FLUXES(1)%FOO, FLUXES(2)%BAR)
  END DO
end subroutine test_remove_block_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.CallStatement).visit(routine.body)) == 5
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 2

    field_group_types = ['state_type', 'aux_type', 'flux_type']
    remove_field_api_view_updates(
        routine, field_group_types=field_group_types, dim_object='DIMS'
    )

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].name == 'MY_KERNEL'

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].variable == 'jkglo'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full type definitions for derived types')]
))
def test_field_api_add_view_updates(frontend):
    """
    A simple test for :any:`add_field_api_view_updates`.
    """

    fcode = """
subroutine test_remove_block_loop(ngptot, nproma, nflux, dims, state, aux_fields, fluxes)
  implicit none
  integer(kind=4), intent(in) :: ngptot, nproma, nflux
  type(dimension_type), intent(inout) :: dims
  type(state_type), intent(inout) :: state
  type(aux_type), intent(inout) :: aux_fields
  type(flux_type), intent(inout) :: fluxes

  integer :: JKGLO, IBL, ICEND, JK, JL, JF

  DO jkglo=1, ngptot, nproma
    icend = min(nproma, ngptot - jkglo + 1)
    ibl = (jkglo - 1) / nproma + 1

    CALL MY_KERNEL(STATE%U, STATE%V, AUX_FIELDS%STUFF, FLUXES%FOO, FLUXES%BAR)
  END DO
end subroutine test_remove_block_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.CallStatement).visit(routine.body)) == 1
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1

    block = Dimension(
        index=('jkglo', 'ibl'), step='NPROMA',
        lower=('1', 'ICST'), upper=('NGPTOT', 'ICEND')
    )
    field_group_types = ['state_type', 'aux_type', 'flux_type']
    add_field_api_view_updates(
        routine, dimension=block, field_group_types=field_group_types,
        dim_object='DIMS'
    )

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 5
    assert calls[0].name == 'DIMS%UPDATE' and calls[0].arguments == ('IBL', 'ICEND', 'JKGLO')
    assert calls[1].name == 'AUX_FIELDS%UPDATE_VIEW' and calls[1].arguments == ('IBL',)
    assert calls[2].name == 'FLUXES%UPDATE_VIEW' and calls[2].arguments == ('IBL',)
    assert calls[3].name == 'STATE%UPDATE_VIEW' and calls[3].arguments == ('IBL',)

    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1
