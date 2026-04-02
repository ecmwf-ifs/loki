# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for declaration-like IR nodes.
"""

import pytest

from loki import Module, Subroutine, Function, fgen, BasicType
from loki.jit_build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI, REGEX, FP, HAVE_FP
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_enum(tmp_path, frontend):
    """Verify that enums are represented correctly"""
    # F2008, Note 4.67
    fcode = """
subroutine test_enum (out)
    implicit none

    ! Comment 1
    ENUM, BIND(C)
        ENUMERATOR :: RED = 4, BLUE = 9
        ! Comment 2
        ENUMERATOR YELLOW
    END ENUM
    ! Comment 3

    integer, intent(out) :: out

    out = RED + BLUE + YELLOW
end subroutine test_enum
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Check Enum exists
    enums = FindNodes(ir.Enumeration).visit(routine.spec)
    assert len(enums) == 1

    # Check symbols are available
    assert enums[0].symbols == ('red', 'blue', 'yellow')
    assert all(name in routine.symbols for name in ('red', 'blue', 'yellow'))
    assert all(s.scope is routine for s in enums[0].symbols)

    # Check assigned values
    assert routine.symbol_map['red'].type.initial == '4'
    assert routine.symbol_map['blue'].type.initial == '9'
    assert routine.symbol_map['yellow'].type.initial is None

    # Verify comments are preserved (don't care about the actual place)
    code = routine.to_fortran()
    for i in range(1, 4):
        assert f'! Comment {i}' in code

    # Check fgen produces valid code and runs
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    out = function()
    assert out == 23


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_derived_type_imports(tmp_path, frontend):
    """ Checks that provided module and type info is attached during parse """
    fcode_module = """
module my_type_mod
  type my_type
    real(kind=8) :: a, b(:)
  end type my_type
end module my_type_mod
"""

    fcode = """
subroutine test_derived_type_parse
  use my_type_mod, only: my_type
  implicit none
  type(my_type) :: obj

  obj%a = 42.0
  obj%b = 66.6
end subroutine test_derived_type_parse
"""
    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(
        fcode, definitions=module, frontend=frontend, xmods=[tmp_path]
    )

    assert len(module.typedefs) == 1
    assert module.typedefs[0].name == 'my_type'

    # Ensure that the imported type is recognised as such
    assert len(routine.imports) == 1
    assert routine.imports[0].module == 'my_type_mod'
    assert len(routine.imports[0].symbols) == 1
    assert routine.imports[0].symbols[0] == 'my_type'
    assert isinstance(routine.imports[0].symbols[0], sym.DerivedTypeSymbol)

    # Ensure that the declared variable and its components are recognised
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert isinstance(assigns[0].lhs, sym.Scalar)
    assert assigns[0].lhs.type.dtype == BasicType.REAL
    assert isinstance(assigns[1].lhs, sym.Array)
    assert assigns[1].lhs.type.dtype == BasicType.REAL
    assert assigns[1].lhs.type.shape == (':',)


@pytest.mark.skipif(not HAVE_FP, reason="Assumed size declarations only supported for FP")
def test_assumed_size_declarations():
    """
    Test if assumed size declarations are correctly parsed.
    """

    fcode = """
subroutine kernel(a, b, c)
  implicit none
  real, intent(in) :: a(*)
  real, intent(in) :: b(8,*)
  real, intent(in) :: c(8,0:*)

end subroutine kernel
"""

    kernel = Subroutine.from_source(fcode, frontend=FP)

    variable_map = kernel.variable_map
    a = variable_map['a']
    b = variable_map['b']
    c = variable_map['c']

    assert len(a.shape) == 1

    assert len(b.shape) == 2
    assert b.shape[0] == 8

    assert len(c.shape) == 2
    assert c.shape[0] == 8
    assert c.shape[1].lower == 0

    assert all('*' in str(shape) for shape in [a.shape, b.shape, c.shape])


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_return_type(tmp_path, frontend):
    """
    Test various ways to define the return type of a function
    """
    fcode = """
module my_funcs
implicit none
contains

  real(kind=8) function funca(a)
    real, intent(in) :: a

    funca = a
  end function funca

  function funcb(a)
    real(kind=8), intent(in) :: a
    real(kind=8) :: funcb

    funcb = a
  end function funcb

  function funcky(a) result(fun)
    real, intent(in) :: a
    real(kind=8) :: fun

    fun = a
  end function funcky

  real function square(a) result(b)
    implicit none
    real, intent(in) :: a
    b = a * a
  end function square
end module my_funcs
    """
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Implicit return type definition
    assert isinstance(module['funca'], Function)
    assert module['funca'].result_name == 'funca'
    assert module['funca'].return_type.dtype == BasicType.REAL
    assert module['funca'].return_type.kind == 8
    assert len(FindNodes(ir.VariableDeclaration).visit(module['funca'].spec)) == 2 if frontend == OMNI else 1
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['funca'].spec)) == 0

    if frontend == OMNI:
        # Ensure return type is declared (OMNI alwas inserts declaration)
        fdecl = tuple(
            d for d in FindNodes(ir.VariableDeclaration).visit(module['funca'].spec)
            if 'funca' in d.symbols
        )
        assert len(fdecl) == 1 and fdecl[0].symbols[0] == 'funca'
        assert fdecl[0].symbols[0].type.dtype == BasicType.REAL
        assert fdecl[0].symbols[0].type.kind == 8
    else:
        # Check for implicit return value in `fgen`
        fstr_header = fgen(module['funca']).splitlines()[0]
        assert 'real(kind=8) function funca (a)' == fstr_header.lower()

    # Explicit return type declaration
    assert isinstance(module['funcb'], Function)
    assert module['funcb'].result_name == 'funcb'
    assert module['funcb'].return_type.dtype == BasicType.REAL
    assert module['funcb'].return_type.kind == 8
    assert len(FindNodes(ir.VariableDeclaration).visit(module['funcb'].spec)) == 2
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['funcb'].spec)) == 0

    # Re-named return type declaration
    assert isinstance(module['funcky'], Function)
    assert module['funcky'].result_name == 'fun'
    assert module['funcky'].return_type.dtype == BasicType.REAL
    assert module['funcky'].return_type.kind == 8
    assert len(FindNodes(ir.VariableDeclaration).visit(module['funcky'].spec)) == 2
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['funcky'].spec)) == 0

    # Implicit return type and renamed result name
    assert isinstance(module['square'], Function)
    assert module['square'].result_name == 'b'
    assert module['square'].return_type.dtype == BasicType.REAL
    assert len(FindNodes(ir.VariableDeclaration).visit(module['square'].spec)) == 2 if frontend == OMNI else 1
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['square'].spec)) == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_prefix(frontend):
    """
    Test various prefixes that can occur in function/subroutine definitions
    """
    fcode = """
pure elemental real function f_elem(a)
    real, intent(in) :: a
    f_elem = a
end function f_elem
    """.strip()

    routine = Function.from_source(fcode, frontend=frontend)
    assert 'PURE' in routine.prefix
    assert 'ELEMENTAL' in routine.prefix
    assert isinstance(routine, Function)
    assert routine.return_type.dtype is BasicType.REAL

    assert routine.procedure_type.is_function is True
    assert routine.procedure_type.return_type.dtype is BasicType.REAL
    assert routine.procedure_type.procedure is routine

    assert routine.procedure_symbol.type.dtype.is_function is True
    assert routine.procedure_symbol.type.dtype.return_type.dtype is BasicType.REAL
    assert routine.procedure_symbol.type.dtype.procedure is routine

    code = fgen(routine)
    assert 'PURE' in code
    assert 'ELEMENTAL' in code


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_suffix(frontend, tmp_path):
    """
    Test that subroutine suffixes are supported and correctly reproduced
    """
    fcode = """
module subroutine_suffix_mod
    implicit none

    interface
        function check_value(value) bind(C, name='check_value')
            use, intrinsic :: iso_c_binding
            real(c_float), value :: value
            integer(c_int) :: check_value
        end function check_value
    end interface

    interface
        function fix_value(value) result(fixed) bind(C, name='fix_value')
            use, intrinsic :: iso_c_binding
            real(c_float), value :: value
            real(c_float) :: fixed
        end function fix_value
    end interface
contains
    function out_of_physical_bounds(field, istartcol, iendcol, do_fix) result(is_bad)
        real, intent(inout) :: field(:)
        integer, intent(in) :: istartcol, iendcol
        logical, intent(in) :: do_fix
        logical :: is_bad

        integer :: jcol
        logical :: bad_value

        is_bad = .false.
        do jcol=istartcol,iendcol
            bad_value = check_value(field(jcol)) > 0
            is_bad = is_bad .or. bad_value
            if (do_fix .and. bad_value) field(jcol) = fix_value(field(jcol))
        end do
    end function out_of_physical_bounds
end module subroutine_suffix_mod
    """.strip()
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    check_value = module.interface_map['check_value'].body[0]
    assert check_value.is_function
    assert check_value.result_name == 'check_value'
    assert check_value.return_type.dtype is BasicType.INTEGER
    assert check_value.return_type.kind == 'c_int'
    if frontend != OMNI:
        assert check_value.bind == 'check_value'
        assert "bind(c, name='check_value')" in fgen(check_value).lower()

    fix_value = module.interface_map['fix_value'].body[0]
    assert fix_value.is_function
    assert fix_value.result_name == 'fixed'
    assert fix_value.return_type.dtype is BasicType.REAL
    assert fix_value.return_type.kind == 'c_float'
    if frontend == OMNI:
        assert 'result(fixed)' in fgen(fix_value).lower()
    else:
        assert fix_value.bind == 'fix_value'
        assert "result(fixed) bind(c, name='fix_value')" in fgen(fix_value).lower()

    routine = module['out_of_physical_bounds']
    assert routine.is_function
    assert routine.result_name == 'is_bad'
    assert routine.bind is None
    assert routine.return_type.dtype is BasicType.LOGICAL
    assert 'result(is_bad)' in fgen(routine).lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_lazy_prefix(frontend):
    """
    Test that prefixes for functions are correctly captured when the object is made
    complete.

    This test represents a case where the REGEX frontend fails to capture these attributes correctly.

    The rationale for this test is that we don't currently need these attributes
    in the incomplete REGEX-parsed IR and we accept that this information is incomplete initially.
    tmp_path, we make sure this information is captured correctly after completing the full frontend
    parse.
    """
    fcode = """
pure elemental real function f_elem(a)
    real, intent(in) :: a
    f_elem = a
end function f_elem
    """.strip()

    routine = Function.from_source(fcode, frontend=REGEX)
    assert routine._incomplete
    assert routine.prefix == ('pure elemental real',)
    assert routine.arguments == ()
    assert routine.is_function is True
    assert routine.return_type is None

    routine.make_complete(frontend=frontend)
    assert not routine._incomplete
    assert 'PURE' in routine.prefix
    assert 'ELEMENTAL' in routine.prefix
    assert routine.arguments == ('a',)
    assert routine.is_function is True
    assert routine.return_type.dtype is BasicType.REAL
