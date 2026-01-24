# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, fgen
from loki.frontend import available_frontends
from loki.ir import FindVariables, FindInlineCalls
from loki.jit_build import jit_compile, clean_test
from loki.types import INTEGER


@pytest.mark.parametrize('frontend', available_frontends())
def test_internal_procedures(tmp_path, frontend):
    """
    Test internal subroutine and function
    """
    fcode = """
subroutine routine_internal_procedures(in1, in2, out1, out2)
  ! Test internal subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call internal_procedure(in1, out1)
  out2 = internal_function(out1)
contains
  subroutine internal_procedure(in1, out1)
    ! This internal procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar + internal_function(1)
  end subroutine internal_procedure

  ! Below is disabled because f90wrap (wrongly) exhibits that
  ! symbol to the public, which causes double defined symbols
  ! upon compilation.

  function internal_function(in2)
    ! This function is just included to test that functions
    ! are also possible
    implicit none
    integer, intent(in) :: in2
    integer :: internal_function

    internal_function = 3 * in2 + 2
  end function internal_function
end subroutine routine_internal_procedures
"""
    # Check that internal procedures are parsed correctly
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(routine.procedures) == 2

    assert routine.procedures[0].name == 'internal_procedure'
    assert routine.procedures[0].symbol_attrs.lookup('localvar', recursive=False) is None
    assert routine.procedures[0].symbol_attrs.lookup('localvar') is not None
    assert routine.procedures[0].get_symbol_scope('localvar') is routine
    assert routine.procedures[0].symbol_attrs.lookup('in1') is not None
    assert routine.symbol_attrs.lookup('in1') is not None
    assert routine.procedures[0].get_symbol_scope('in1') is routine.procedures[0]

    # Check that inline function is correctly identified
    inline_calls = list(FindInlineCalls().visit(routine.procedures[0].body))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.name == 'internal_function'
    assert inline_calls[0].function.type.dtype.procedure == routine.procedures[1]

    assert routine.procedures[1].name == 'internal_function'
    assert routine.procedures[1].symbol_attrs.lookup('in2') is not None
    assert routine.procedures[1].get_symbol_scope('in2') is routine.procedures[1]
    assert routine.symbol_attrs.lookup('in2') is not None
    assert routine.get_symbol_scope('in2') is routine

    # Generate code, compile and load
    filepath = tmp_path/(f'routine_internal_procedures_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='routine_internal_procedures')

    # Test results of the generated and compiled code
    out1, out2 = function(1, 2)
    assert out1 == 12
    assert out2 == 38
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_internal_routine_clone(frontend):
    """
    Test that internal subroutine scopes get cloned correctly.
    """
    fcode = """
subroutine internal_routine_clone(in1, in2, out1, out2)
  ! Test internal subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call internal_procedure(in1, out1)
  out2 = 3 * out1 + 2

contains
  subroutine internal_procedure(in1, out1)
    ! This internal procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar
  end subroutine internal_procedure
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    new_routine = routine.clone()

    # Ensure we have cloned parent and internal routine
    assert routine is not new_routine
    assert routine.procedures[0] is not new_routine.procedures[0]
    assert fgen(routine) == fgen(new_routine)
    assert fgen(routine.procedures[0]) == fgen(new_routine.procedures[0])

    # Check that the scopes are linked correctly
    assert routine.procedures[0].parent is routine
    assert new_routine.procedures[0].parent is new_routine

    # Check that variables are in the right scope everywhere
    assert all(v.scope is routine for v in FindVariables().visit(routine.ir))
    assert all(v.scope in (routine, routine.procedures[0]) for v in FindVariables().visit(routine.procedures[0].ir))
    assert all(v.scope is new_routine for v in FindVariables().visit(new_routine.ir))
    assert all(
        v.scope in (new_routine, new_routine.procedures[0])
        for v in FindVariables().visit(new_routine.procedures[0].ir)
    )


@pytest.mark.parametrize('frontend', available_frontends())
def test_internal_routine_clone_inplace(frontend):
    """
    Test that internal subroutine scopes get cloned correctly.
    """
    fcode = """
subroutine internal_routine_clone(in1, in2, out1, out2)
  ! Test internal subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call internal_procedure(in1, out1)
  out2 = 3 * out1 + 2

contains
  subroutine internal_procedure(in1, out1)
    ! This internal procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar
  end subroutine internal_procedure

  subroutine other_internal(inout1)
    ! Another internal routine that uses a parent symbol
    implicit none
    integer, intent(inout) :: inout1

    inout1 = 2 * inout1 + localvar
  end subroutine other_internal
end subroutine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Make sure the initial state is as expected
    internal = routine['internal_procedure']
    assert internal.parent is routine
    assert internal.symbol_attrs.parent is routine.symbol_attrs
    other_internal = routine['other_internal']
    assert other_internal.parent is routine
    assert other_internal.symbol_attrs.parent is routine.symbol_attrs

    # Put the inherited symbol in the local scope, first with a clean clone...
    internal.variables += (routine.variable_map['localvar'].clone(scope=internal),)
    internal = internal.clone(parent=None)
    # ...and then with a clone that preserves the symbol table
    other_internal.variables += (routine.variable_map['localvar'].clone(scope=other_internal),)
    other_internal = other_internal.clone(parent=None, symbol_attrs=other_internal.symbol_attrs)
    # Ultimately, remove the internal routines
    routine = routine.clone(contains=None)

    # Check that variables are in the right scope everywhere
    assert all(v.scope is routine for v in FindVariables().visit(routine.ir))
    assert all(v.scope is internal for v in FindVariables().visit(internal.ir))

    # Check that we aren't looking somewhere above anymore
    assert internal.parent is None
    assert internal.symbol_attrs.parent is None
    assert internal.parent is None
    assert internal.symbol_attrs._parent is None
    assert other_internal.parent is None
    assert other_internal.symbol_attrs.parent is None
    assert other_internal.parent is None
    assert other_internal.symbol_attrs.parent is None


@pytest.mark.parametrize('frontend', available_frontends())
def test_internal_procedures_alias(frontend):
    """ Test local variable aliases in internal subroutine and function """
    fcode = """
subroutine outer_routine(in, out)
  implicit none
  integer, intent(in) :: in
  integer, intent(out) :: out
  integer :: a, b(2, 2)

  b(1, 1) = in

  call internal_routine(in, out)
contains

  subroutine internal_routine(in, out)
    integer, intent(in) :: in
    integer, intent(out) :: out
    integer :: a(3, 4)

    a(1, 2) = 3
    out = a(1, 2) + b(1, 1) + in
  end subroutine internal_routine
end subroutine outer_routine
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    internal = routine['internal_routine']

    a_outer = routine.get_type('a')
    assert a_outer.dtype == INTEGER
    assert not a_outer.shape

    b_outer = routine.get_type('b')
    assert b_outer.dtype == INTEGER
    assert b_outer.dtype == INTEGER
    assert b_outer.shape == (2, 2)

    a_inner = internal.get_type('a')
    assert a_inner.dtype == INTEGER
    assert a_inner.shape == (3, 4)
