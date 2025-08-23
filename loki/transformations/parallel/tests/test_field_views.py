# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Dimension
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes
from loki.expression import symbols as sym
from loki.types import BasicType, SymbolAttributes, Scope
from loki.logging import WARNING

from loki.transformations.field_api import (
    get_field_type, field_get_device_data, field_get_host_data, field_sync_host, field_sync_device,
    FieldAPITransferType, field_create_device_data, field_delete_device_data
)
from loki.transformations.parallel import (
    remove_field_api_view_updates, add_field_api_view_updates
)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI needs full type definitions for derived types')]
))
def test_field_api_remove_view_updates(caplog, frontend):
    """
    A simple test for :any:`remove_field_api_view_updates`
    """

    fcode = """
subroutine test_remove_block_loop(ngptot, nproma, nflux, dims, state, aux_fields, fluxes, ricks_fields)
  use type_module, only: dimension_type, state_type, aux_type, flux_type, ricks_type
  implicit none
  integer(kind=4), intent(in) :: ngptot, nproma, nflux
  type(dimension_type), intent(inout) :: dims
  type(STATE_TYPE), intent(inout) :: state
  type(aux_type), intent(inout) :: aux_fields
  type(FLUX_type), intent(inout) :: fluxes(nflux)
  type(ricks_type), intent(inout) :: ricks_fields

  integer :: JKGLO, IBL, ICEND, JK, JL, JF

  DO jkglo=1, ngptot, nproma
    icend = min(nproma, ngptot - JKGLO + 1)
    ibl = (jkglo - 1) / nproma + 1

    STATE = STATE%CLONE()

    CALL DIMS%UPDATE(IBL, ICEND, JKGLO)
    CALL STATE%update_VIEW(IBL)
    CALL AUX_FIELDS%UPDATE_VIEW(block_index=IBL)
    IF (NFLUX > 0) THEN
      DO jf=1, nflux
        CALL FLUXES(JF)%UPDATE_VIEW(IBL)
      END DO
    END IF
    CALL RICKS_FIELDS%UPDATE_VIEW(IBL)

    CALL MY_KERNEL(STATE%U, STATE%V, AUX_FIELDS%STUFF, FLUXES(1)%FOO, FLUXES(2)%BAR)
  END DO
end subroutine test_remove_block_loop
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.CallStatement).visit(routine.body)) == 6
    assert len(FindNodes(ir.Conditional).visit(routine.body)) == 1
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 2

    with caplog.at_level(WARNING):
        field_group_types = ['state_type', 'aux_type', 'flux_type']
        remove_field_api_view_updates(
            routine, field_group_types=field_group_types, dim_object='DIMS'
        )

        assert len(caplog.records) == 2
        assert '[Loki::ControlFlow] Found LHS field group assign: Assignment:: STATE = STATE%CLONE()'\
            in caplog.records[0].message
        assert '[Loki::ControlFlow] Removing RICKS_FIELDS%UPDATE_VIEW call, but not in field group types!'\
            in caplog.records[1].message

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].name == 'MY_KERNEL'

    assert len(FindNodes(ir.Conditional).visit(routine.body)) == 0
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


def test_get_field_type():
    type_map = ["jprb",
                "jpit",
                "jpis",
                "jpim",
                "jpib",
                "jpia",
                "jprt",
                "jprs",
                "jprm",
                "jprd",
                "jplm"]
    field_types = [
                    "field_1rb", "field_2rb", "field_3rb",
                    "field_1it", "field_2it", "field_3it",
                    "field_1is", "field_2is", "field_3is",
                    "field_1im", "field_2im", "field_3im",
                    "field_1ib", "field_2ib", "field_3ib",
                    "field_1ia", "field_2ia", "field_3ia",
                    "field_1rt", "field_2rt", "field_3rt",
                    "field_1rs", "field_2rs", "field_3rs",
                    "field_1rm", "field_2rm", "field_3rm",
                    "field_1rd", "field_2rd", "field_3rd",
                    "field_1lm", "field_2lm", "field_3lm",
                  ]

    def generate_fields(types):
        generated = []
        for type_name in types:
            for dim in range(1, 4):
                shape = tuple(None for _ in range(dim))
                a = sym.Variable(name='test_array',
                                 type=SymbolAttributes(BasicType.REAL,
                                                       shape=shape,
                                                       kind=sym.Variable(name=type_name)))
                generated.append(get_field_type(a))
        return generated

    generated = generate_fields(type_map)
    for field, field_name in zip(generated, field_types):
        assert isinstance(field, sym.DerivedType) and field.name == field_name

    generated = generate_fields([t.upper() for t in type_map])
    for field, field_name in zip(generated, field_types):
        assert isinstance(field, sym.DerivedType) and field.name == field_name


@pytest.mark.parametrize('transfer_type', list(FieldAPITransferType))
def test_field_api_call_namegen(transfer_type):
    """
    Test the correct generation of FIELD_API calls.
    """

    mode_str = {
        'READ_ONLY': 'rdonly',
        'READ_WRITE': 'rdwr',
        'WRITE_ONLY': 'wronly',
        'FORCE': 'force'
    }

    def _check_get_call(call, target, mode):
        assert call.name.name.lower() == f'field%get_{target}_data_{mode_str[mode.name]}'

    def _check_sync_call(call, target, mode):
        assert call.name.name.lower() == f'field%sync_{target}_{mode_str[mode.name]}'

    routine = Subroutine('routine')
    field_object = sym.Variable(name='field')
    access_ptr = sym.Variable(name='field_access_ptr')

    # verify that get_device_data calls are generated corectly
    get_device_data = field_get_device_data(field_object, access_ptr, transfer_type, scope=routine)
    _check_get_call(get_device_data, 'device', transfer_type)
    # verify that get_host_data calls are generated correctly
    if transfer_type != FieldAPITransferType.WRITE_ONLY:
        get_host_data = field_get_host_data(field_object, access_ptr, transfer_type, scope=routine)
        _check_get_call(get_host_data, 'host', transfer_type)

    # verify that sync_device calls are generated corectly
    sync_device = field_sync_device(field_object, transfer_type, scope=routine)
    _check_sync_call(sync_device, 'device', transfer_type)
    # verify that sync_host calls are generated correctly
    if transfer_type != FieldAPITransferType.WRITE_ONLY:
        sync_host = field_sync_host(field_object, transfer_type, scope=routine)
        _check_sync_call(sync_host, 'host', transfer_type)


    # verify that create_device_data calls are generated corectly
    create_device_data = field_create_device_data(field_object, scope=routine)
    assert create_device_data.name.name.lower() == 'field%create_device_data'

    # verify that delete_device_data calls are generated corectly
    delete_device_data = field_delete_device_data(field_object, scope=routine)
    assert delete_device_data.name.name.lower() == 'field%delete_device_data'


@pytest.mark.parametrize("field_get_fn", [field_get_device_data, field_get_host_data])
def test_field_get_data(field_get_fn):
    scope= Scope()
    fptr = sym.Variable(name='fptr_var')
    dev_ptr = sym.Variable(name='data_var')
    queue = sym.IntLiteral(1)
    blk_bounds = sym.Array(name='blk_bounds', scope=scope, dimensions=(None, None))

    for fttype in FieldAPITransferType:
        if fttype == FieldAPITransferType.FORCE:
            sync_call = field_get_fn(fptr, dev_ptr, fttype, scope, queue, blk_bounds)
            assert isinstance(sync_call, ir.CallStatement)
            assert sync_call.name.parent == fptr, dev_ptr
        else:
            # queue and blk bounds can only be used for force methods
            with pytest.raises(ValueError):
                _ = field_get_fn(fptr, dev_ptr, fttype, scope, queue, blk_bounds)
            # field_sync_host has not write-only option
            if fttype == FieldAPITransferType.WRITE_ONLY and field_get_fn.__name__ == 'field_get_host_data':
                with pytest.raises(TypeError):
                    _ = field_get_fn(fptr, dev_ptr, fttype, scope)
            else:
                sync_call = field_get_fn(fptr, dev_ptr, fttype, scope)
                assert isinstance(sync_call, ir.CallStatement)
                assert sync_call.name.parent == fptr

    with pytest.raises(TypeError):
        _ = field_get_fn(fptr, dev_ptr, "none_transfer_type", scope)


@pytest.mark.parametrize("field_sync_fn", [field_sync_device, field_sync_host])
def test_field_sync(field_sync_fn):
    scope= Scope()
    fptr = sym.Variable(name='fptr_var')
    queue = sym.IntLiteral(1)
    blk_bounds = sym.Array(name='blk_bounds', scope=scope, dimensions=(None, None))

    for fttype in FieldAPITransferType:
        if fttype == FieldAPITransferType.FORCE:
            sync_call = field_sync_fn(fptr, fttype, scope, queue, blk_bounds)
            assert isinstance(sync_call, ir.CallStatement)
            assert sync_call.name.parent == fptr
        else:
            # queue and blk bounds can only be used for force methods
            with pytest.raises(ValueError):
                _ = field_sync_fn(fptr, fttype, scope, queue, blk_bounds)
            # field_sync_host has not write-only option
            if fttype == FieldAPITransferType.WRITE_ONLY and field_sync_fn.__name__ == 'field_sync_host':
                with pytest.raises(TypeError):
                    _ = field_sync_fn(fptr, fttype, scope)
            else:
                sync_call = field_sync_fn(fptr, fttype, scope)
                assert isinstance(sync_call, ir.CallStatement)
                assert sync_call.name.parent == fptr

    with pytest.raises(TypeError):
        _ = field_sync_fn(fptr, "none_transfer_type", scope)
