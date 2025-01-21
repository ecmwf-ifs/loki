# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Module, Subroutine
from loki.build import jit_compile_lib, Builder, Obj
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, FindInlineCalls
)
from loki.types import ProcedureType

from loki.transformations.inline import (
    inline_elemental_functions, inline_statement_functions
)


@pytest.fixture(name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_inline_elemental_functions(tmp_path, builder, frontend):
    """
    Test correct inlining of elemental functions.
    """
    fcode_module = """
module multiply_mod
  use iso_fortran_env, only: real64
  implicit none
contains

  elemental function multiply(a, b)
    real(kind=real64) :: multiply
    real(kind=real64), intent(in) :: a, b
    real(kind=real64) :: temp
    !$loki routine seq

    ! simulate multi-line function
    temp = a * b
    multiply = temp
  end function multiply
end module multiply_mod
"""

    fcode = """
subroutine transform_inline_elemental_functions(v1, v2, v3)
  use iso_fortran_env, only: real64
  use multiply_mod, only: multiply
  real(kind=real64), intent(in) :: v1
  real(kind=real64), intent(out) :: v2, v3

  v2 = multiply(v1, 6._real64)
  v3 = 600. + multiply(6._real64, 11._real64)
end subroutine transform_inline_elemental_functions
"""

    # Generate reference code, compile run and verify
    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    refname = f'ref_{routine.name}_{frontend}'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)

    v2, v3 = reference.transform_inline_elemental_functions(11.)
    assert v2 == 66.
    assert v3 == 666.

    (tmp_path/f'{module.name}.f90').unlink()
    (tmp_path/f'{routine.name}.f90').unlink()

    # Now inline elemental functions
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend, xmods=[tmp_path])
    inline_elemental_functions(routine)

    # Make sure there are no more inline calls in the routine body
    assert not FindInlineCalls().visit(routine.body)

    # Verify correct scope of inlined elements
    assert all(v.scope is routine for v in FindVariables().visit(routine.body))

    # Ensure the !$loki routine pragma has been removed
    assert not FindNodes(ir.Pragma).visit(routine.body)

    # Hack: rename routine to use a different filename in the build
    routine.name = f'{routine.name}_'
    kernel = jit_compile_lib([routine], path=tmp_path, name=routine.name, builder=builder)

    v2, v3 = kernel.transform_inline_elemental_functions_(11.)
    assert v2 == 66.
    assert v3 == 666.

    builder.clean()
    (tmp_path/f'{routine.name}.f90').unlink()

@pytest.fixture(name='multiply_extended_mod', params=available_frontends())
def fixture_multiply_extended_mod(request, tmp_path):
    fcode_module = """
module multiply_extended_mod
  use iso_fortran_env, only: real64
  implicit none
contains

  elemental function multiply(a, b) ! result (ret_mult)
    ! real(kind=real64) :: ret_mult
    real(kind=real64) :: multiply
    real(kind=real64), intent(in) :: a, b
    real(kind=real64) :: temp

    ! simulate multi-line function
    temp = a * b
    multiply = temp
    ! ret_mult = temp
  end function multiply

  elemental function multiply_single_line(a, b)
    real(kind=real64) :: multiply_single_line
    real(kind=real64), intent(in) :: a, b
    real(kind=real64) :: temp

    multiply_single_line = a * b
  end function multiply_single_line

  elemental function add(a, b)
    real(kind=real64) :: add
    real(kind=real64), intent(in) :: a, b
    real(kind=real64) :: temp

    ! simulate multi-line function
    temp = a + b
    add = temp
  end function add
end module multiply_extended_mod
"""

    frontend = request.param
    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    return module, frontend

def test_transform_inline_elemental_functions_extended_scalar(multiply_extended_mod, builder, tmp_path):
    module, frontend = multiply_extended_mod

    fcode = """
subroutine transform_inline_elemental_functions_extended_scalar(v1, v2, v3)
  use iso_fortran_env, only: real64
  use multiply_extended_mod, only: multiply, multiply_single_line, add
  real(kind=real64), intent(in) :: v1
  real(kind=real64), intent(out) :: v2, v3
  real(kind=real64), parameter :: param1 = 100.

  v2 = multiply(v1, 6._real64) + multiply_single_line(v1, 3._real64)
  v3 = add(param1, 200._real64) + add(150._real64, 150._real64) + multiply(6._real64, 11._real64)
end subroutine transform_inline_elemental_functions_extended_scalar
"""

    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=[module], xmods=[tmp_path])
    refname = f'ref_{routine.name}_{frontend}'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)
    v2, v3 = reference.transform_inline_elemental_functions_extended_scalar(11.)
    assert v2 == 99.
    assert v3 == 666.

    (tmp_path/f'{routine.name}.f90').unlink()

    # Now inline elemental functions
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend, xmods=[tmp_path])
    inline_elemental_functions(routine)
    # Make sure there are no more inline calls in the routine body
    assert not FindInlineCalls().visit(routine.body)
    # Verify correct scope of inlined elements
    assert all(v.scope is routine for v in FindVariables().visit(routine.body))
    # Hack: rename routine to use a different filename in the build
    routine.name = f'{routine.name}_'
    kernel = jit_compile_lib([routine, module], path=tmp_path, name=routine.name, builder=builder)
    v2, v3 = kernel.transform_inline_elemental_functions_extended_scalar_(11.)
    assert v2 == 99.
    assert v3 == 666.

    builder.clean()
    (tmp_path/f'{routine.name}.f90').unlink()
    (tmp_path/f'{module.name}.f90').unlink()

def test_transform_inline_elemental_functions_extended_arr(multiply_extended_mod, builder, tmp_path):
    module, frontend = multiply_extended_mod

    fcode_arr = """
subroutine transform_inline_elemental_functions_extended_array(v1, v2, v3, len)
  use iso_fortran_env, only: real64
  use multiply_extended_mod, only: multiply, multiply_single_line, add
  integer, intent(in) :: len  
  real(kind=real64), intent(in) :: v1(len)
  real(kind=real64), intent(inout) :: v2(len), v3(len)
  real(kind=real64), parameter :: param1 = 100.
  integer, parameter :: arr_index = 1

  v2 = multiply(v1(:), 6._real64) + multiply_single_line(v1(:), 3._real64)
  v3 = add(param1, 200._real64) + add(v1, 150._real64) + multiply(v1(arr_index), v2(1))
end subroutine transform_inline_elemental_functions_extended_array
"""

    routine = Subroutine.from_source(fcode_arr, frontend=frontend, definitions=[module], xmods=[tmp_path])
    refname = f'ref_{routine.name}_frontend'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)
    arr_len = 5
    v1 = np.array([1.0, 2.0, 3.0, 5.0, 3.0], dtype=np.float64, order='F')
    v2 = np.zeros((arr_len,), dtype=np.float64, order='F')
    v3 = np.zeros((arr_len,), dtype=np.float64, order='F')
    reference.transform_inline_elemental_functions_extended_array(v1, v2, v3, arr_len)
    assert (v2 == np.array([9., 18., 27., 45., 27.], dtype=np.float64, order='F')).all()
    assert (v3 == np.array([460., 461., 462., 464., 462.], dtype=np.float64, order='F')).all()

    (tmp_path/f'{routine.name}.f90').unlink()

    routine = Subroutine.from_source(fcode_arr, definitions=module, frontend=frontend, xmods=[tmp_path])
    inline_elemental_functions(routine)
    # TODO: Make sure there are no more inline calls in the routine body
    #  assert not FindInlineCalls().visit(routine.body)
    #  this is currently not achievable as calls to elemental functions with array arguments
    #  can't be properly inlined and therefore are skipped
    # Verify correct scope of inlined elements
    assert all(v.scope is routine for v in FindVariables().visit(routine.body))
    # Hack: rename routine to use a different filename in the build
    routine.name = f'{routine.name}_'
    kernel = jit_compile_lib([routine, module], path=tmp_path, name=routine.name, builder=builder)
    v1 = np.array([1.0, 2.0, 3.0, 5.0, 3.0], dtype=np.float64, order='F')
    v2 = np.zeros((arr_len,), dtype=np.float64, order='F')
    v3 = np.zeros((arr_len,), dtype=np.float64, order='F')
    kernel.transform_inline_elemental_functions_extended_array_(v1, v2, v3, arr_len)
    assert (v2 == np.array([9., 18., 27., 45., 27.], dtype=np.float64, order='F')).all()
    assert (v3 == np.array([460., 461., 462., 464., 462.], dtype=np.float64, order='F')).all()

    builder.clean()
    (tmp_path/f'{routine.name}.f90').unlink()
    (tmp_path/f'{module.name}.f90').unlink()


@pytest.mark.parametrize('frontend', available_frontends(
    skip={OMNI: "OMNI automatically inlines Statement Functions"}
))
@pytest.mark.parametrize('stmt_decls', (True, False))
def test_inline_statement_functions(frontend, stmt_decls):
    stmt_decls_code = """
    real :: PTARE
    real :: FOEDELTA
    FOEDELTA ( PTARE ) = PTARE + 1.0
    real :: FOEEW
    FOEEW ( PTARE ) = PTARE + FOEDELTA(PTARE) + EXP(PTARE)
    """.strip()

    fcode = f"""
subroutine stmt_func(arr, ret)
    implicit none
    real, intent(in) :: arr(:)
    real, intent(inout) :: ret(:)
    real :: ret2
    real, parameter :: rtt = 1.0
    {stmt_decls_code if stmt_decls else '#include "fcttre.func.h"'}

    ret = foeew(arr)
    ret2 = foedelta(3.0)
end subroutine stmt_func
     """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    if stmt_decls:
        assert FindNodes(ir.StatementFunction).visit(routine.spec)
    else:
        assert not FindNodes(ir.StatementFunction).visit(routine.spec)
    assert FindInlineCalls().visit(routine.body)
    inline_statement_functions(routine)

    assert not FindNodes(ir.StatementFunction).visit(routine.spec)
    if stmt_decls:
        assert len(FindInlineCalls().visit(routine.body)) == 1
        assignments = FindNodes(ir.Assignment).visit(routine.body)
        assert assignments[0].lhs  == 'ret'
        assert assignments[0].rhs  ==  "arr + arr + 1.0 + exp(arr)"
        assert assignments[1].lhs  == 'ret2'
        assert assignments[1].rhs  ==  "3.0 + 1.0"
    else:
        assert FindInlineCalls().visit(routine.body)

@pytest.mark.parametrize('frontend', available_frontends(
    skip={OMNI: "OMNI automatically inlines Statement Functions"}
))
@pytest.mark.parametrize('provide_myfunc', ('import', 'module', 'interface', 'intfb', 'routine'))
def test_inline_statement_functions_inline_call(frontend, provide_myfunc, tmp_path):
    fcode_myfunc = """
elemental function myfunc(a)
    real, intent(in) :: a
    real :: myfunc
    myfunc = a * 2.0
end function myfunc
    """.strip()

    if provide_myfunc == 'module':
        fcode_myfunc = f"""
module my_mod
implicit none
contains
{fcode_myfunc}
end module my_mod
        """.strip()

    if provide_myfunc in ('import', 'module'):
        module_import = 'use my_mod, only: myfunc'
    else:
        module_import = ''

    if provide_myfunc == 'interface':
        intf = """
            interface
            elemental function myfunc(a)
                implicit none
                real a
                real myfunc
            end function myfunc
            end interface
        """
    elif provide_myfunc in ('intfb', 'routine'):
        intf = '#include "myfunc.intfb.h"'
    else:
        intf = ''

    fcode = f"""
subroutine stmt_func(arr, val, ret)
    {module_import}
    implicit none
    real, intent(in) :: arr(:)
    real, intent(in) :: val
    real, intent(inout) :: ret(:)
    real :: ret2
    real, parameter :: rtt = 1.0
    real :: PTARE
    real :: FOEDELTA
    FOEDELTA ( PTARE ) = PTARE + 1.0 + MYFUNC(PTARE)
    real :: FOEEW
    FOEEW ( PTARE ) = PTARE + FOEDELTA(PTARE) + MYFUNC(PTARE)
    {intf}

    ret = foeew(arr)
    ret2 = foedelta(3.0) + foedelta(val)
end subroutine stmt_func
    """.strip()

    if provide_myfunc == 'module':
        definitions = (Module.from_source(fcode_myfunc, xmods=[tmp_path]),)
    elif provide_myfunc == 'routine':
        definitions = (Subroutine.from_source(fcode_myfunc, xmods=[tmp_path]),)
    else:
        definitions = None
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=definitions, xmods=[tmp_path])

    # Check the spec
    statement_funcs = FindNodes(ir.StatementFunction).visit(routine.spec)
    assert len(statement_funcs) == 2

    inline_calls = FindInlineCalls(unique=False).visit(routine.spec)
    if provide_myfunc in ('module', 'interface', 'routine'):
        # Enough information available that MYFUNC is recognized as a procedure call
        assert len(inline_calls) == 3
        assert all(isinstance(call.function.type.dtype, ProcedureType) for call in inline_calls)
    else:
        # No information available about MYFUNC, so fparser treats it as an ArraySubscript
        assert len(inline_calls) == 1
        assert inline_calls[0].function == 'foedelta'
        assert isinstance(inline_calls[0].function.type.dtype, ProcedureType)

    # Check the body
    inline_calls = FindInlineCalls().visit(routine.body)
    assert len(inline_calls) == 3

    # Apply the transformation
    inline_statement_functions(routine)

    # Check the outcome
    assert not FindNodes(ir.StatementFunction).visit(routine.spec)
    inline_calls = FindInlineCalls(unique=False).visit(routine.body)
    assignments = FindNodes(ir.Assignment).visit(routine.body)

    if provide_myfunc in ('import', 'intfb'):
          # MYFUNC(arr) is misclassified as array subscript
        assert len(inline_calls) == 0
    elif provide_myfunc in ('module', 'routine'):
          # MYFUNC(arr) is eliminated due to inlining
        assert len(inline_calls) == 0
    else:
        assert len(inline_calls) == 4

    assert assignments[0].lhs  == 'ret'
    assert assignments[1].lhs  == 'ret2'
    if provide_myfunc in ('module', 'routine'):
        # Fully inlined due to definition of myfunc available
        assert assignments[0].rhs  ==  "arr + arr + 1.0 + arr*2.0 + arr*2.0"
        assert assignments[1].rhs  ==  "3.0 + 1.0 + 3.0*2.0 + val + 1.0 + val*2.0"
    else:
        # myfunc not inlined
        assert assignments[0].rhs  ==  "arr + arr + 1.0 + myfunc(arr) + myfunc(arr)"
        assert assignments[1].rhs  ==  "3.0 + 1.0 + myfunc(3.0) + val + 1.0 + myfunc(val)"


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_elemental_functions_intrinsic_procs(frontend):
    fcode = """
subroutine test_inline_elementals(a)
implicit none
  integer, parameter :: jprb = 8
  real(kind=jprb), intent(inout) :: a

  a = fminj(0.5, a)
contains
  pure elemental function fminj(x,y) result(m)
    real(kind=jprb), intent(in) :: x, y
    real(kind=jprb) :: m

    m = y - 0.5_jprb*(abs(x-y)-(x-y))
  end function fminj
end subroutine test_inline_elementals
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 1
    assert isinstance(assigns[0].rhs.function, sym.ProcedureSymbol)
    assert assigns[0].rhs.function.type.dtype.procedure == routine.members[0]

    # Ensure we have an intrinsic in the internal elemental function
    inline_calls = tuple(FindInlineCalls().visit(routine.members[0].body))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.type.is_intrinsic
    assert inline_calls[0].function.scope == routine.members[0]

    inline_elemental_functions(routine)

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2

    # Ensure that the intrinsic function has been rescoped
    inline_calls = tuple(FindInlineCalls().visit(assigns[0]))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.type.is_intrinsic
    assert inline_calls[0].function.scope == routine
