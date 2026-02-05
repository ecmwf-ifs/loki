# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Function, Module, Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes
from loki.types import ProcedureType


@pytest.mark.parametrize('frontend', available_frontends())
def test_procedure_type(tmp_path, frontend):
    """ Test `ProcedureType` links to the procedure when it is defined. """

    fcode_mod = """
module my_mod
implicit none

contains

  subroutine test_routine(n, a)
    integer, intent(in) :: n
    real(kind=4), intent(inout) :: a(3)
    real(kind=4) :: smoke
    real(kind=4) :: pants, on, fire
    pants(on, fire) = on + fire

    call me_maybe(n, a)

    smoke = on_the_water(a(3))
  end subroutine test_routine

  subroutine me_maybe(n, a)
    integer, intent(in) :: n
    real(kind=4), intent(inout) :: a(3)

    a(1) = on_the_water(a(2))
  end subroutine me_maybe

  function on_the_water(b) result(rick)
    real(kind=4), intent(in) :: b
    real(kind=4) :: rick

    rick = 2*b
  end function on_the_water
end module my_mod
"""
    module = Module.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])
    routine = module['test_routine']
    assert isinstance(module['me_maybe'], Subroutine)
    assert isinstance(module.symbol_attrs['me_maybe'].dtype, ProcedureType)
    assert isinstance(module['on_the_water'], Function)
    assert isinstance(module.symbol_attrs['on_the_water'].dtype, ProcedureType)

    # Procedure type linked to Subroutine
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    ptype = calls[0].name.type.dtype
    assert isinstance(ptype, ProcedureType)
    assert str(ptype) == 'me_maybe' and repr(ptype) == '<ProcedureType me_maybe>'
    assert ptype.procedure == module['me_maybe']

    # Procedure type linked to Function
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 1
    assert isinstance(assigns[0].rhs, sym.InlineCall)
    ftype = assigns[0].rhs.function.type.dtype
    assert str(ftype) == 'on_the_water' and repr(ftype) == '<ProcedureType on_the_water>'
    assert ftype.procedure == module['on_the_water']

    # Procedure type linked to StatementFunction (not supported in OMNI)
    stmtfuncs = FindNodes(ir.StatementFunction).visit(routine.spec)
    if frontend != OMNI:
        assert len(stmtfuncs) == 1
        sftype = stmtfuncs[0].variable.type.dtype
        assert isinstance(sftype, ProcedureType)
        assert str(sftype) == 'pants' and repr(sftype) == '<ProcedureType pants>'
        assert sftype.procedure == stmtfuncs[0]
