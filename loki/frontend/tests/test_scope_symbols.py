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

from loki import Function, Module, Subroutine, BasicType
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


@pytest.mark.parametrize('frontend', available_frontends())
def test_contained_routine_parent_scope(frontend):
    """
    Test that contained routines retain host association correctly.
    """
    fcode = """
subroutine outer_routine(a, b)
  implicit none
  integer, intent(in) :: a
  integer, intent(out) :: b
  integer :: host_value

  host_value = a
  call inner_routine(b)
contains
  subroutine inner_routine(out)
    implicit none
    integer, intent(out) :: out
    out = host_value + a
  end subroutine inner_routine
end subroutine outer_routine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    inner = routine['inner_routine']
    assert inner.parent is routine
    assert inner.symbol_attrs.lookup('host_value', recursive=False) is None
    assert inner.symbol_attrs.lookup('host_value') is not None
    assert inner.get_symbol_scope('host_value') is routine
    assert inner.get_symbol_scope('a') is routine
    assert inner.get_symbol_scope('out') is inner


@pytest.mark.parametrize('frontend', available_frontends())
def test_local_symbol_shadows_import(frontend, tmp_path):
    """
    Test that local declarations shadow imported names in nested scope.
    """
    code_mod = """
module imported_mod
  implicit none
  integer :: val
end module imported_mod
    """
    code = """
module host_mod
  use imported_mod, only: val
contains
  subroutine driver(out)
    integer, intent(out) :: out
    integer :: val
    val = 3
    out = val
  end subroutine driver
end module host_mod
    """

    imported_mod = Module.from_source(code_mod, frontend=frontend, xmods=[tmp_path])
    host_mod = Module.from_source(code, frontend=frontend, definitions=[imported_mod], xmods=[tmp_path])
    routine = host_mod['driver']
    local_val = routine.variable_map['val']
    assert local_val.scope is routine
    assert local_val.type.imported is None


@pytest.mark.parametrize('frontend', available_frontends())
def test_interface_block_symbol_scope(frontend):
    """
    Test that symbols declared inside interface bodies have interface-local scope.
    """
    fcode = """
subroutine interface_scope(proc)
  implicit none
  interface
    subroutine proc(a, b)
      integer, intent(in) :: a
      integer, intent(out) :: b
    end subroutine proc
  end interface
end subroutine interface_scope
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    interface_routine = routine.interface_map['proc'].body[0]
    assert interface_routine.parent is routine
    assert interface_routine.arguments[0].scope is interface_routine
    assert interface_routine.arguments[1].scope is interface_routine
