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

from loki import Module, Subroutine, BasicType
from loki.jit_build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends, FP, HAVE_FP
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
