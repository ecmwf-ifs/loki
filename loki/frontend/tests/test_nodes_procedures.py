# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for procedure declarations and interfaces.
"""

from pathlib import Path

import pytest

from loki import Sourcefile, Subroutine, fgen
from loki.jit_build import jit_compile_lib, clean_test
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes
from loki.types import BasicType, ProcedureType, SymbolAttributes


@pytest.fixture(scope='module', name='header_path')
def fixture_header_path():
    return Path(__file__).parents[2] / 'tests' / 'sources' / 'header.f90'


@pytest.mark.parametrize('frontend', available_frontends())
def test_external_stmt(tmp_path, frontend):
    """
    Tests procedures passed as dummy arguments and declared as EXTERNAL.
    """
    fcode_external = """
! This should be tested as well with interface statements in the caller
! routine, and the subprogram definitions outside (to have "truly external"
! procedures, however, we need to make the INTERFACE support more robust first

subroutine other_external_subroutine(outvar)
  implicit none
  integer, intent(out) :: outvar
  outvar = 4
end subroutine other_external_subroutine

function other_external_function() result(outvar)
  implicit none
  integer :: outvar
  outvar = 6
end function other_external_function
    """.strip()

    fcode = """
subroutine routine_external_stmt(invar, sub1, sub2, sub3, outvar, func1, func2, func3)
  implicit none
  integer, intent(in) :: invar
  external sub1
  external :: sub2, sub3
  integer, intent(out) :: outvar
  integer, external :: func1, func2
  integer, external :: func3
  integer tmp

  call sub1(tmp)
  outvar = invar + tmp  ! invar + 1
  call sub2(tmp)
  outvar = outvar + tmp + func1()  ! (invar + 1) + 1 + 6
  call sub3(tmp)
  outvar = outvar + tmp + func2()  ! (invar + 8) + 4 + 2
  tmp = func3()
  outvar = outvar + tmp  ! (invar + 14) + 2
end subroutine routine_external_stmt

subroutine routine_call_external_stmt(invar, outvar)
  implicit none
  integer, intent(in) :: invar
  integer, intent(out) :: outvar

  interface
    subroutine other_external_subroutine(outvar)
      integer, intent(out) :: outvar
    end subroutine other_external_subroutine
  end interface

  interface
    function other_external_function()
      integer :: other_external_function
    end function other_external_function
  end interface

  call routine_external_stmt(invar, external_subroutine, external_subroutine, other_external_subroutine, &
                            &outvar, other_external_function, external_function, external_function)

contains

  subroutine external_subroutine(outvar)
    implicit none
    integer, intent(out) :: outvar
    outvar = 1
  end subroutine external_subroutine

  function external_function()
    implicit none
    integer :: external_function
    external_function = 2
  end function external_function

end subroutine routine_call_external_stmt
    """.strip()

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['routine_external_stmt']
    assert len(routine.arguments) == 8

    for decl in FindNodes(ir.ProcedureDeclaration).visit(routine.spec):
        # Is the EXTERNAL attribute set?
        assert decl.external
        for var in decl.symbols:
            assert isinstance(var, sym.ProcedureSymbol)
            assert isinstance(var.type.dtype, ProcedureType)
            assert var.type.external is True
            assert var.type.dtype.procedure == BasicType.DEFERRED
            if 'sub' in var.name:
                assert not var.type.dtype.is_function
                assert var.type.dtype.return_type is None
            else:
                assert var.type.dtype.is_function
                assert var.type.dtype.return_type.compare(SymbolAttributes(BasicType.INTEGER))

    # Generate code, compile and load
    extpath = tmp_path/(f'subroutine_routine_external_{frontend}.f90')
    extpath.write_text(fcode_external)
    filepath = tmp_path/(f'subroutine_routine_external_stmt_{frontend}.f90')
    source.path = filepath
    lib = jit_compile_lib([source, extpath], path=tmp_path, name='subroutine_external')
    function = lib.routine_call_external_stmt

    outvar = function(7)
    assert outvar == 23
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_subroutine_interface(tmp_path, frontend, header_path):
    """
    Test auto-generation of an interface block for a given subroutine.
    """
    fcode = """
subroutine test_subroutine_interface (in1, in2, in3, out1, out2)
  use header, only: jprb
  IMPLICIT NONE
  integer, intent(in) :: in1, in2
  real(kind=jprb), intent(in) :: in3(in1, in2)
  real(kind=jprb), intent(out) :: out1, out2
  integer :: localvar
  localvar = in1 + in2
  out1 = real(localvar, kind=jprb)
  out2 = out1 + 2.
end subroutine
"""
    if frontend == OMNI:
        # Generate xmod
        Sourcefile.from_file(header_path, frontend=frontend, xmods=[tmp_path])

    routine = Subroutine.from_source(fcode, xmods=[tmp_path], frontend=frontend)

    if frontend == OMNI:
        assert fgen(routine.interface).strip() == """
INTERFACE
  SUBROUTINE test_subroutine_interface (in1, in2, in3, out1, out2)
    USE header, ONLY: jprb
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: in1
    INTEGER, INTENT(IN) :: in2
    REAL(KIND=selected_real_kind(13, 300)), INTENT(IN) :: in3(in1, in2)
    REAL(KIND=selected_real_kind(13, 300)), INTENT(OUT) :: out1
    REAL(KIND=selected_real_kind(13, 300)), INTENT(OUT) :: out2
  END SUBROUTINE test_subroutine_interface
END INTERFACE
""".strip()
    else:
        assert fgen(routine.interface).strip() == """
INTERFACE
  SUBROUTINE test_subroutine_interface (in1, in2, in3, out1, out2)
    USE header, ONLY: jprb
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: in1, in2
    REAL(KIND=jprb), INTENT(IN) :: in3(in1, in2)
    REAL(KIND=jprb), INTENT(OUT) :: out1, out2
  END SUBROUTINE test_subroutine_interface
END INTERFACE
""".strip()


@pytest.mark.parametrize('frontend', available_frontends())
def test_mixed_declaration_interface(frontend):
    """
    A simple test to catch and shame mixed declarations.
    """
    fcode = """
subroutine valid_fortran(i, m)
   integer :: i, j, m
   integer :: k,l
end subroutine valid_fortran
"""

    with pytest.raises(AssertionError) as error:
        routine = Subroutine.from_source(fcode, frontend=frontend)
        assert isinstance(routine.body, ir.Section)
        assert isinstance(routine.spec, ir.Section)
        _ = routine.interface

    assert 'Declarations must have intents' in str(error.value)
