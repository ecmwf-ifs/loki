# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for richer interface constructs.
"""

import pytest

from loki import Interface, Module, Subroutine, fgen, ProcedureSymbol, ProcedureType
from loki.frontend import available_frontends, OMNI


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_subroutine_integration(frontend):
    """
    Test correct integration of interfaces into subroutines.
    """
    fcode = """
subroutine interface_subroutine_integration(x, y, n, proc)
    integer, intent(in) :: x(:,:), n
    integer, intent(out) :: y(:)
    interface
        subroutine proc(a, b)
            integer, intent(in) :: a(:)
            integer, intent(out) :: b
        end subroutine proc
    end interface
    integer :: i

    do i=1,n
        call proc(x(:, i), y(i))
    end do
end subroutine interface_subroutine_integration
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(routine.interfaces) == 1
    interface = routine.interfaces[0]
    assert isinstance(interface, Interface)

    assert interface.symbols == ('proc',)
    assert routine.interface_symbols == ('proc',)
    assert routine.interface_map['proc'] is interface
    assert routine.interface_symbol_map == {'proc': interface.symbols[0]}
    assert 'proc' in routine.symbols
    assert routine.symbol_map['proc'] == interface.symbols[0]
    assert 'proc' in routine.arguments
    assert 'proc' in [arg.lower() for arg in routine.argnames]
    assert routine.symbol_map['proc'].type.dtype.procedure is interface.body[0]

    code = routine.to_fortran().lower()
    assert 'interface' in code
    assert 'subroutine proc' in code


@pytest.mark.parametrize(
    'frontend', available_frontends(skip=[(OMNI, 'OMNI separates generic interfaces differently')])
)
def test_interface_generic_spec(frontend, tmp_path):
    """
    Test interfaces with a generic identifier.
    """
    fcode = """
module interface_generic_spec_mod
    implicit none
    interface switch
        subroutine int_switch(x, y)
            integer, intent(inout) :: x, y
        end subroutine int_switch
        subroutine real_switch(x, y)
            real, intent(inout) :: x, y
        end subroutine real_switch
        subroutine complex_switch(x, y)
            complex, intent(inout) :: x, y
        end subroutine complex_switch
    end interface switch
end module interface_generic_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert len(module.interfaces) == 1

    intf = module.interfaces[0]
    assert set(intf.symbols) == {'switch', 'int_switch', 'real_switch', 'complex_switch'}
    assert isinstance(intf.spec, ProcedureSymbol)
    assert intf.spec.scope is module
    assert intf.spec == 'switch'
    assert intf.spec.type.dtype.is_generic is True
    assert 'INTERFACE SWITCH' in fgen(intf).upper()
    assert all(s in module.symbols for s in ('switch', 'int_switch', 'real_switch', 'complex_switch'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_operator_module_procedure(frontend, tmp_path):
    """
    Test interfaces that declare generic operators and refer to module procedures.
    """
    fcode = """
module spectral_fields_mod
implicit none
private
public spectral_field, assignment(=), operator(.eqv.)

type spectral_field
    real, allocatable :: sp2d(:,:)
    integer :: ns2d
    integer :: nspec2
end type spectral_field

interface assignment (=)
    module procedure assign_scalar_sp, assign_sp_ar
end interface

interface operator (.eqv.)
    procedure equiv_spec
end interface

contains

subroutine assign_scalar_sp(ydsp,pval)
    type (spectral_field), intent(inout) :: ydsp
    real, intent(in) :: pval
    ydsp%sp2d(:,:)  =pval
end subroutine assign_scalar_sp

subroutine assign_sp_ar(pflat,ydsp)
    real, intent(out) :: pflat(:)
    type (spectral_field), intent(in) :: ydsp
    integer :: i2d,ishape2d(1)

    i2d=ydsp%ns2d*ydsp%nspec2
    ishape2d(1)=i2d
    pflat(1:i2d)=reshape(ydsp%sp2d(:,:), ishape2d)
end subroutine assign_sp_ar

logical function equiv_spec(ydsp1,ydsp2)
    type(spectral_field), intent(in) :: ydsp1
    type(spectral_field), intent(in) :: ydsp2
    equiv_spec = ydsp1%ns2d == ydsp2%ns2d
end function equiv_spec
end module spectral_fields_mod
    """.strip()

    mod = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert len(mod.interfaces) == 2

    assign_intf = mod.interface_map['assignment(=)']
    assert assign_intf.spec == 'assignment(=)'
    assert set(assign_intf.symbols) == {'assignment(=)', 'assign_scalar_sp', 'assign_sp_ar'}
    assign_map = {symbol.name.lower(): symbol for symbol in assign_intf.symbols}
    assert assign_map['assignment(=)'].type.dtype.is_generic is True
    assert assign_map['assign_scalar_sp'].type.dtype.procedure is mod['assign_scalar_sp']
    assert assign_map['assign_sp_ar'].type.dtype.procedure is mod['assign_sp_ar']
    assign_code = fgen(assign_intf).lower().strip()
    assert assign_code.startswith('interface assignment(=)')
    assert 'module procedure' in assign_code

    op_intf = mod.interface_map['operator(.eqv.)']
    assert op_intf.spec == 'operator(.eqv.)'
    assert set(op_intf.symbols) == {'operator(.eqv.)', 'equiv_spec'}
    op_map = {symbol.name.lower(): symbol for symbol in op_intf.symbols}
    assert op_map['operator(.eqv.)'].type.dtype.is_generic is True
    assert op_map['equiv_spec'].type.dtype.procedure is mod['equiv_spec']
    op_code = fgen(op_intf).lower().strip()
    assert op_code.startswith('interface operator(.eqv.)')
    assert op_code.endswith('end interface operator(.eqv.)')


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_procedure_pointer(frontend, tmp_path):
    """
    Test interface bodies with procedure-typed dummy arguments.
    """
    fcode = """
module my_interface_mod
implicit none
abstract interface
  function sim_func(x)
    real, intent(in) :: x
    real :: sim_func
  end function sim_func
end interface

interface
  subroutine sub2(x, p)
    real, intent(inout) :: x
    procedure(sim_func) :: p
  end subroutine sub2
end interface
end module my_interface_mod
    """.strip()

    mod = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    intf_sim_func = mod.interface_map['sim_func']
    assert intf_sim_func.abstract
    assert intf_sim_func.symbols[0].type.dtype.procedure is intf_sim_func.body[0]

    intf_sub2 = mod.interface_map['sub2']
    sub2 = intf_sub2.body[0]
    arg_p = sub2.arguments[1]
    assert isinstance(arg_p.type.dtype, ProcedureType)
    assert arg_p.type.dtype.procedure is intf_sim_func.body[0]
