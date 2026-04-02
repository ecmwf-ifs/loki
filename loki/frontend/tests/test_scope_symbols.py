# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for symbol scoping.
"""

import pytest

from loki import Function, Module, BasicType
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_import_of_private_symbols(tmp_path, frontend):
    """
    Verify that only public symbols are imported from other modules.
    """
    code_mod_private = """
module mod_private
    private
    integer :: var
end module mod_private
    """
    code_mod_public = """
module mod_public
    public
    integer:: var
end module mod_public
    """
    code_mod_main = """
module mod_main
    use mod_public
    use mod_private
contains

    subroutine test_routine()
        integer :: result
        result = var
    end subroutine test_routine

end module mod_main
    """

    mod_private = Module.from_source(code_mod_private, frontend=frontend, xmods=[tmp_path])
    mod_public = Module.from_source(code_mod_public, frontend=frontend, xmods=[tmp_path])
    mod_main = Module.from_source(
        code_mod_main, frontend=frontend, definitions=[mod_private, mod_public], xmods=[tmp_path]
    )
    var = mod_main.subroutines[0].body.body[0].rhs
    # Check if this is really our symbol
    assert var.name == 'var'
    assert var.scope is mod_main
    # Check if the symbol is imported
    assert var.type.imported is True
    # Check if the symbol comes from the mod_public module
    assert var.type.module is mod_public


@pytest.mark.parametrize('frontend', available_frontends())
def test_access_spec(tmp_path, frontend):
    """
    Check that access-spec statements are dealt with correctly.
    """
    code_mod_private_var_public = """
module mod_private_var_public
    private
    integer :: var
    public :: var
end module mod_private_var_public
    """
    code_mod_public_var_private = """
module mod_public_var_private
    public
    integer :: var
    private :: var
end module mod_public_var_private
    """
    code_mod_main = """
module mod_main
    use mod_private_var_public
    use mod_public_var_private
contains

    subroutine test_routine()
        integer :: result
        result = var
    end subroutine test_routine

end module mod_main
    """

    mod_private_var_public = Module.from_source(code_mod_private_var_public, frontend=frontend, xmods=[tmp_path])
    mod_public_var_private = Module.from_source(code_mod_public_var_private, frontend=frontend, xmods=[tmp_path])
    mod_main = Module.from_source(
        code_mod_main, frontend=frontend, definitions=[mod_private_var_public, mod_public_var_private], xmods=[tmp_path]
    )
    var = mod_main.subroutines[0].body.body[0].rhs
    # Check if this is really our symbol
    assert var.name == 'var'
    assert var.scope is mod_main
    # Check if the symbol is imported
    assert var.type.imported is True
    # Check if the symbol comes from the mod_private_var_public module
    assert var.type.module is mod_private_var_public


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_symbol_scoping(frontend):
    """ Check that the return symbol of a function has the right scope """
    fcode = """
real(kind=8) function double_real(i)
  implicit none
  integer, intent(in) :: i

  double_real =  dble(i*2)
end function double_real
"""
    routine = Function.from_source(fcode, frontend=frontend)

    rtyp = routine.symbol_attrs['double_real']
    assert rtyp.dtype == BasicType.REAL
    assert rtyp.kind == 8

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 1
    assert assigns[0].lhs == 'double_real'
    assert isinstance(assigns[0].lhs, sym.Scalar)
    assert assigns[0].lhs.type.dtype == BasicType.REAL
    assert assigns[0].lhs.type.kind == 8
    assert assigns[0].lhs.scope == routine
