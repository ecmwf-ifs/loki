# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, fgen
from loki.jit_build import jit_compile, clean_test
from loki.frontend import available_frontends
from loki.ir import FindVariables, FindInlineCalls


@pytest.mark.parametrize('frontend', available_frontends())
def test_member_procedures(tmp_path, frontend):
    """
    Test member subroutine and function
    """
    fcode = """
subroutine routine_member_procedures(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call member_procedure(in1, out1)
  out2 = member_function(out1)
contains
  subroutine member_procedure(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar + member_function(1)
  end subroutine member_procedure

  ! Below is disabled because f90wrap (wrongly) exhibits that
  ! symbol to the public, which causes double defined symbols
  ! upon compilation.

  function member_function(in2)
    ! This function is just included to test that functions
    ! are also possible
    implicit none
    integer, intent(in) :: in2
    integer :: member_function

    member_function = 3 * in2 + 2
  end function member_function
end subroutine routine_member_procedures
"""
    # Check that member procedures are parsed correctly
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(routine.members) == 2

    assert routine.members[0].name == 'member_procedure'
    assert routine.members[0].symbol_attrs.lookup('localvar', recursive=False) is None
    assert routine.members[0].symbol_attrs.lookup('localvar') is not None
    assert routine.members[0].get_symbol_scope('localvar') is routine
    assert routine.members[0].symbol_attrs.lookup('in1') is not None
    assert routine.symbol_attrs.lookup('in1') is not None
    assert routine.members[0].get_symbol_scope('in1') is routine.members[0]

    # Check that inline function is correctly identified
    inline_calls = list(FindInlineCalls().visit(routine.members[0].body))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.name == 'member_function'
    assert inline_calls[0].function.type.dtype.procedure == routine.members[1]

    assert routine.members[1].name == 'member_function'
    assert routine.members[1].symbol_attrs.lookup('in2') is not None
    assert routine.members[1].get_symbol_scope('in2') is routine.members[1]
    assert routine.symbol_attrs.lookup('in2') is not None
    assert routine.get_symbol_scope('in2') is routine

    # Generate code, compile and load
    filepath = tmp_path/(f'routine_member_procedures_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='routine_member_procedures')

    # Test results of the generated and compiled code
    out1, out2 = function(1, 2)
    assert out1 == 12
    assert out2 == 38
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_member_routine_clone(frontend):
    """
    Test that member subroutine scopes get cloned correctly.
    """
    fcode = """
subroutine member_routine_clone(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call member_procedure(in1, out1)
  out2 = 3 * out1 + 2

contains
  subroutine member_procedure(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar
  end subroutine member_procedure
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    new_routine = routine.clone()

    # Ensure we have cloned routine and member
    assert routine is not new_routine
    assert routine.members[0] is not new_routine.members[0]
    assert fgen(routine) == fgen(new_routine)
    assert fgen(routine.members[0]) == fgen(new_routine.members[0])

    # Check that the scopes are linked correctly
    assert routine.members[0].parent is routine
    assert new_routine.members[0].parent is new_routine

    # Check that variables are in the right scope everywhere
    assert all(v.scope is routine for v in FindVariables().visit(routine.ir))
    assert all(v.scope in (routine, routine.members[0]) for v in FindVariables().visit(routine.members[0].ir))
    assert all(v.scope is new_routine for v in FindVariables().visit(new_routine.ir))
    assert all(
        v.scope in (new_routine, new_routine.members[0])
        for v in FindVariables().visit(new_routine.members[0].ir)
    )


@pytest.mark.parametrize('frontend', available_frontends())
def test_member_routine_clone_inplace(frontend):
    """
    Test that member subroutine scopes get cloned correctly.
    """
    fcode = """
subroutine member_routine_clone(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call member_procedure(in1, out1)
  out2 = 3 * out1 + 2

contains
  subroutine member_procedure(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar
  end subroutine member_procedure

  subroutine other_member(inout1)
    ! Another member that uses a parent symbol
    implicit none
    integer, intent(inout) :: inout1

    inout1 = 2 * inout1 + localvar
  end subroutine other_member
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Make sure the initial state is as expected
    member = routine['member_procedure']
    assert member.parent is routine
    assert member.symbol_attrs.parent is routine.symbol_attrs
    other_member = routine['other_member']
    assert other_member.parent is routine
    assert other_member.symbol_attrs.parent is routine.symbol_attrs

    # Put the inherited symbol in the local scope, first with a clean clone...
    member.variables += (routine.variable_map['localvar'].clone(scope=member),)
    member = member.clone(parent=None)
    # ...and then with a clone that preserves the symbol table
    other_member.variables += (routine.variable_map['localvar'].clone(scope=other_member),)
    other_member = other_member.clone(parent=None, symbol_attrs=other_member.symbol_attrs)
    # Ultimately, remove the member routines
    routine = routine.clone(contains=None)

    # Check that variables are in the right scope everywhere
    assert all(v.scope is routine for v in FindVariables().visit(routine.ir))
    assert all(v.scope is member for v in FindVariables().visit(member.ir))

    # Check that we aren't looking somewhere above anymore
    assert member.parent is None
    assert member.symbol_attrs.parent is None
    assert member.parent is None
    assert member.symbol_attrs._parent is None
    assert other_member.parent is None
    assert other_member.symbol_attrs.parent is None
    assert other_member.parent is None
    assert other_member.symbol_attrs.parent is None
