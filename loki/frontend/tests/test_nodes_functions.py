# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for function declarations and signatures.
"""

import pytest

from loki import Module, Function, fgen, BasicType
from loki.frontend import available_frontends, OMNI, REGEX
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_return_type(tmp_path, frontend):
    """
    Test various ways to define the return type of a function.
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

    assert isinstance(module['funca'], Function)
    assert module['funca'].result_name == 'funca'
    assert module['funca'].return_type.dtype == BasicType.REAL
    assert module['funca'].return_type.kind == 8
    assert len(FindNodes(ir.VariableDeclaration).visit(module['funca'].spec)) == 2 if frontend == OMNI else 1
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['funca'].spec)) == 0

    if frontend == OMNI:
        fdecl = tuple(
            decl for decl in FindNodes(ir.VariableDeclaration).visit(module['funca'].spec)
            if 'funca' in decl.symbols
        )
        assert len(fdecl) == 1 and fdecl[0].symbols[0] == 'funca'
        assert fdecl[0].symbols[0].type.dtype == BasicType.REAL
        assert fdecl[0].symbols[0].type.kind == 8
    else:
        fstr_header = fgen(module['funca']).splitlines()[0]
        assert 'real(kind=8) function funca (a)' == fstr_header.lower()

    assert isinstance(module['funcb'], Function)
    assert module['funcb'].result_name == 'funcb'
    assert module['funcb'].return_type.dtype == BasicType.REAL
    assert module['funcb'].return_type.kind == 8
    assert len(FindNodes(ir.VariableDeclaration).visit(module['funcb'].spec)) == 2
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['funcb'].spec)) == 0

    assert isinstance(module['funcky'], Function)
    assert module['funcky'].result_name == 'fun'
    assert module['funcky'].return_type.dtype == BasicType.REAL
    assert module['funcky'].return_type.kind == 8
    assert len(FindNodes(ir.VariableDeclaration).visit(module['funcky'].spec)) == 2
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['funcky'].spec)) == 0

    assert isinstance(module['square'], Function)
    assert module['square'].result_name == 'b'
    assert module['square'].return_type.dtype == BasicType.REAL
    assert len(FindNodes(ir.VariableDeclaration).visit(module['square'].spec)) == 2 if frontend == OMNI else 1
    assert len(FindNodes(ir.ProcedureDeclaration).visit(module['square'].spec)) == 0


@pytest.mark.parametrize('frontend', available_frontends())
def test_function_prefix(frontend):
    """
    Test various prefixes that can occur in function definitions.
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
    Test that function suffixes are supported and correctly reproduced.
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
    Test that prefixes for functions are correctly captured when the object is made complete.
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


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('dim_decl', [':: add_to_a(n)', ', DIMENSION(n) :: add_to_a'])
def test_function_array_return_type(frontend, dim_decl):
    """
    Verify array return types are correctly represented with all frontends.
    """
    fcode = f"""
subroutine member_functions
    implicit none
    integer :: i
    real(kind=8) :: a(3)
    contains
    function add_to_a(b, n)
      integer, intent(in) :: n
      real(kind=8), intent(in) :: b(n)
      real(kind=8) {dim_decl}

      do i = 1, n
        add_to_a(i) = a(i) + b(i)
      end do
    end function
end subroutine member_functions
    """.strip()
    routine = Function.from_source(fcode, frontend=frontend)
    add_to_a = routine['add_to_a']
    return_type = add_to_a.procedure_type.return_type
    assert return_type.dtype == BasicType.REAL
    assert return_type.shape == ('n',)
    ret_var = add_to_a.variable_map['add_to_a']
    assert ret_var.type.dtype == BasicType.REAL
    assert ret_var.type.shape == ('n',)
    assert ret_var.dimensions == ('n',)

    if frontend == OMNI:
        assert ':: add_to_a(n)' in routine.to_fortran()
    else:
        assert dim_decl in routine.to_fortran()
