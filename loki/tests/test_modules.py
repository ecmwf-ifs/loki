# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import (
    Module, Subroutine, VariableDeclaration, TypeDef, fexprgen,
    BasicType, Assignment, FindNodes, FindInlineCalls, FindTypedSymbols,
    Transformer, fgen, SymbolAttributes, Variable, Import, Section, Intrinsic,
    Scalar, DeferredTypeSymbol, FindVariables, SubstituteExpressions, Literal
)
from loki.build import jit_compile, clean_test
from loki.frontend import available_frontends, OFP, OMNI
from loki.sourcefile import Sourcefile


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_from_source(frontend, tmp_path):
    """
    Test the creation of `Module` objects from raw source strings.
    """
    fcode = """
module a_module
  integer, parameter :: x = 2
  integer, parameter :: y = 3

  type derived_type
    real :: array(x, y)
  end type derived_type
contains

  subroutine my_routine(pt)
    type(derived_type) :: pt
    pt%array(:,:) = 42.0
  end subroutine my_routine
end module a_module
""".strip()
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert len([o for o in module.spec.body if isinstance(o, VariableDeclaration)]) == 2
    assert len([o for o in module.spec.body if isinstance(o, TypeDef)]) == 1
    assert 'derived_type' in module.typedef_map
    assert len(module.routines) == 1
    assert module.routines[0].name == 'my_routine'
    if frontend != OMNI:
        assert module.source.string == fcode
        assert module.source.lines == (1, fcode.count('\n') + 1)


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_external_typedefs_subroutine(frontend, tmp_path):
    """
    Test that externally provided type information is correctly
    attached to a `Module` subroutine when supplied via the `typedefs`
    parameter in the constructor.
    """
    fcode_external = """
module external_mod
  integer, parameter :: x = 2
  integer, parameter :: y = 3

  type ext_type
    real :: array(x, y)
  end type ext_type
end module external_mod
"""

    fcode_module = """
module a_module
contains

  subroutine my_routine(pt_ext)
    use external_mod, only: ext_type
    implicit none

    type(ext_type) :: pt_ext
    pt_ext%array(:,:) = 42.0
  end subroutine my_routine
end module a_module
"""

    external = Module.from_source(fcode_external, frontend=frontend, xmods=[tmp_path])
    assert 'ext_type' in external.typedef_map

    module = Module.from_source(fcode_module, frontend=frontend, definitions=external, xmods=[tmp_path])
    routine = module.subroutines[0]
    pt_ext = routine.variables[0]

    # OMNI resolves explicit shape parameters in the frontend parser
    exptected_array_shape = '(1:2, 1:3)' if frontend == OMNI else '(x, y)'

    # Check that the `array` variable in the `ext` type is found and
    # has correct type and shape info
    assert 'array' in pt_ext.variable_map
    a = pt_ext.variable_map['array']
    assert a.type.dtype == BasicType.REAL
    assert fexprgen(a.shape) == exptected_array_shape

    # Check the LHS of the assignment has correct meta-data
    stmt = FindNodes(Assignment).visit(routine.body)[0]
    pt_ext_arr = stmt.lhs
    assert pt_ext_arr.type.dtype == BasicType.REAL
    assert fexprgen(pt_ext_arr.shape) == exptected_array_shape


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_external_typedefs_type(frontend, tmp_path):
    """
    Test that externally provided type information is correctly
    attached to a `Module` type and used in a contained subroutine
    when supplied via the `typedefs` parameter in the constructor.
    """
    fcode_external = """
module external_mod
  integer, parameter :: x = 2
  integer, parameter :: y = 3

  type ext_type
    real :: array(x, y)
  end type ext_type
end module external_mod
"""

    fcode_other = """
module other_mod
  integer, parameter :: z = 4

  type other_type
    real :: vector(z)
  end type other_type
end module other_mod
    """.strip()

    fcode_module = """
module a_module
  use external_mod, only: ext_type
  use other_mod
  implicit none

  type nested_type
    type(ext_type) :: ext
  end type nested_type
contains

  subroutine my_routine(pt)
    type(nested_type) :: pt
    pt%ext%array(:,:) = 42.0
  end subroutine my_routine

  subroutine other_routine(pt)
    type(other_type) :: pt
    pt%vector(:) = 13.37
  end subroutine other_routine
end module a_module
"""

    external = Module.from_source(fcode_external, frontend=frontend, xmods=[tmp_path])
    assert 'ext_type' in external.typedef_map

    other = Module.from_source(fcode_other, frontend=frontend, xmods=[tmp_path])
    assert 'other_type' in other.typedef_map

    if frontend != OMNI:  # OMNI needs to know imported modules
        module = Module.from_source(fcode_module, frontend=frontend)
        assert 'ext_type' in module.symbol_attrs
        assert module.symbol_attrs['ext_type'].dtype is BasicType.DEFERRED
        assert 'other_type' not in module.symbol_attrs
        assert 'other_type' not in module['other_routine'].symbol_attrs
        assert module['other_routine'].symbol_attrs['pt'].dtype.typedef is BasicType.DEFERRED

    module = Module.from_source(fcode_module, frontend=frontend, definitions=[external, other], xmods=[tmp_path])
    nested = module.typedef_map['nested_type']
    ext = nested.variables[0]

    # Verify correct attachment of type information
    assert 'ext_type' in module.symbol_attrs
    assert isinstance(module.symbol_attrs['ext_type'].dtype.typedef, TypeDef)
    assert isinstance(nested.symbol_attrs['ext'].dtype.typedef, TypeDef)
    assert isinstance(module['my_routine'].symbol_attrs['pt'].dtype.typedef, TypeDef)
    assert isinstance(module['my_routine'].symbol_attrs['pt%ext'].dtype.typedef, TypeDef)
    assert 'other_type' in module.symbol_attrs
    assert 'other_type' not in module['other_routine'].symbol_attrs
    assert isinstance(module.symbol_attrs['other_type'].dtype.typedef, TypeDef)
    assert isinstance(module['other_routine'].symbol_attrs['pt'].dtype.typedef, TypeDef)

    # OMNI resolves explicit shape parameters in the frontend parser
    exptected_array_shape = '(1:2, 1:3)' if frontend == OMNI else '(x, y)'

    # Check that the `array` variable in the `ext` type is found and
    # has correct type and shape info
    assert 'array' in ext.variable_map
    a = ext.variable_map['array']
    assert a.type.dtype == BasicType.REAL
    assert fexprgen(a.shape) == exptected_array_shape

    # Check the routine has got type and shape info too
    routine = module['my_routine']
    pt = routine.variables[0]
    pt_ext = pt.variable_map['ext']
    assert 'array' in pt_ext.variable_map
    pt_ext_a = pt_ext.variable_map['array']
    assert pt_ext_a.type.dtype == BasicType.REAL
    assert fexprgen(pt_ext_a.shape) == exptected_array_shape

    # Check the LHS of the assignment has correct meta-data
    stmt = FindNodes(Assignment).visit(routine.body)[0]
    pt_ext_arr = stmt.lhs
    assert pt_ext_arr.type.dtype == BasicType.REAL
    assert fexprgen(pt_ext_arr.shape) == exptected_array_shape


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_nested_types(frontend, tmp_path):
    """
    Test that ensure that nested internal derived type definitions are
    detected and connected correctly.
    """

    fcode = """
module type_mod
  integer, parameter :: x = 2
  integer, parameter :: y = 3

  type sub_type
    real :: array(x, y)
  end type sub_type

  type parent_type
    type(sub_type) :: pt
  end type parent_type
end module type_mod
"""
    # OMNI resolves explicit shape parameters in the frontend parser
    exptected_array_shape = '(1:2, 1:3)' if frontend == OMNI else '(x, y)'

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    parent = module.typedef_map['parent_type']
    pt = parent.variables[0]
    assert 'array' in pt.variable_map
    arr = pt.variable_map['array']
    assert arr.type.dtype == BasicType.REAL
    assert fexprgen(arr.shape) == exptected_array_shape


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Loki annotation break parser')]))
def test_dimension_pragmas(frontend, tmp_path):
    """
    Test that loki-specific dimension annotations are detected and
    used to set shapes.
    """

    fcode = """
module type_mod
  implicit none
  type mytype
    !$loki dimension(size)
    integer, pointer :: x(:)
  end type mytype
end module type_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    mytype = module.typedef_map['mytype']
    assert fexprgen(mytype.variables[0].shape) == '(size,)'


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Loki annotation break parser')]))
def test_nested_types_dimension_pragmas(frontend, tmp_path):
    """
    Test that loki-specific dimension annotations are detected and
    propagated in nested type definitions.
    """

    fcode = """
module type_mod
  implicit none
  type sub_type
    !$loki dimension(size)
    integer, pointer :: x(:)
  end type sub_type

  type parent_type
    type(sub_type) :: pt
  end type parent_type
end module type_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    parent = module.typedef_map['parent_type']
    child = module.typedef_map['sub_type']
    assert fexprgen(child.variables[0].shape) == '(size,)'

    pt_x = parent.variables[0].variable_map['x']
    assert fexprgen(pt_x.shape) == '(size,)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_internal_function_call(frontend, tmp_path):
    """
    Test the use of `InlineCall` symbols linked to an module function.
    """
    fcode = """
module module_mod
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)

contains

  subroutine test_inline_call(v1, v2, v3)
    implicit none

    integer, intent(in) :: v1
    real(kind=jprb), intent(in) :: v2
    real(kind=jprb), intent(out) :: v3

    v3 = util_fct(v2, v1)
  end subroutine test_inline_call

  function util_fct(var, mode)
    real(kind=jprb) :: util_fct
    integer, intent(in) :: var
    real(kind=jprb), intent(in) :: mode

    if (mode == 1) then
      util_fct = var + 2_jprb
    else
      util_fct = var + 3_jprb
    end if
  end function util_fct

end module
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['test_inline_call']

    inline_calls = list(FindInlineCalls().visit(routine.body))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.name == 'util_fct'
    assert inline_calls[0].parameters[0] == 'v2'
    assert inline_calls[0].parameters[1] == 'v1'

    assert isinstance(module.symbol_attrs['util_fct'].dtype.procedure, Subroutine)
    assert module.symbol_attrs['util_fct'].dtype.is_function


@pytest.mark.parametrize('frontend', available_frontends())
def test_external_function_call(frontend, tmp_path):
    """
    Test the use of `InlineCall` symbols linked to an external function definition.
    """
    fcode = """
subroutine test_inline_call(v1, v2, v3)
  use util_mod, only: util_fct
  implicit none

  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: v1
  real(kind=jprb), intent(in) :: v2
  real(kind=jprb), intent(out) :: v3

  v3 = util_fct(v2, v1)
end subroutine test_inline_call
"""

    fcode_util = """
module util_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

contains
  function util_fct(var, mode)
    real(kind=jprb) :: util_fct
    integer, intent(in) :: var
    real(kind=jprb), intent(in) :: mode

    if (mode == 1) then
      util_fct = var + 2_jprb
    else
      util_fct = var + 3_jprb
    end if
  end function util_fct
end module
"""
    module = Module.from_source(fcode_util, frontend=frontend, xmods=[tmp_path])
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend, xmods=[tmp_path])

    inline_calls = list(FindInlineCalls().visit(routine.body))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.name == 'util_fct'
    assert inline_calls[0].parameters[0] == 'v2'
    assert inline_calls[0].parameters[1] == 'v1'


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_variables_add_remove(frontend, tmp_path):
    """
    Test local variable addition and removal.
    """
    fcode = """
module module_variables_add_remove
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer :: x, y
  real(kind=jprb), allocatable :: vector(:)
end module module_variables_add_remove
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    module_vars = [str(arg) for arg in module.variables]
    assert module_vars == ['jprb', 'x', 'y', 'vector(:)']

    # Create a new set of variables and add to local routine variables
    x = module.variable_map['x']  # That's the symbol for variable 'x'
    real_type = SymbolAttributes('real', kind=module.variable_map['jprb'])
    int_type = SymbolAttributes('integer')
    a = Variable(name='a', type=real_type, scope=module)
    b = Variable(name='b', dimensions=(x, ), type=real_type, scope=module)
    c = Variable(name='c', type=int_type, scope=module)

    # Add new variables and check that they are all in the module spec
    module.variables += (a, b, c)
    if frontend == OMNI:
        # OMNI frontend inserts a few peculiarities
        assert fgen(module.spec).lower() == """
integer, parameter :: jprb = selected_real_kind(13, 300)
integer :: x
integer :: y
real(kind=selected_real_kind(13, 300)), allocatable :: vector(:)
real(kind=jprb) :: a
real(kind=jprb) :: b(x)
integer :: c
""".strip().lower()

    else:
        assert fgen(module.spec).lower() == """
implicit none
integer, parameter :: jprb = selected_real_kind(13, 300)
integer :: x, y
real(kind=jprb), allocatable :: vector(:)
real(kind=jprb) :: a
real(kind=jprb) :: b(x)
integer :: c
""".strip().lower()

    # Now remove the `vector` variable and make sure it's gone
    module.variables = [v for v in module.variables if v.name != 'vector']
    assert 'vector' not in fgen(module.spec).lower()
    module_vars = [str(arg) for arg in module.variables]
    assert module_vars == ['jprb', 'x', 'y', 'a', 'b(x)', 'c']


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parsing fails without dummy module provided')]))
def test_module_rescope_symbols(frontend, tmp_path):
    """
    Test the rescoping of variables.
    """
    fcode = """
module test_module_rescope
  use some_mod, only: ext1
  implicit none
  integer :: a, b, c
end module test_module_rescope
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    ref_fgen = fgen(module)

    # Create a copy of the module with rescoping and make sure all symbols are in the right scope
    spec = Transformer().visit(module.spec)
    module_copy = Module(name=module.name, spec=spec, rescope_symbols=True)

    for var in FindTypedSymbols().visit(module_copy.spec):
        assert var.scope is module_copy

    # Create another copy of the nested subroutine without rescoping
    spec = Transformer().visit(module.spec)
    other_module_copy = Module(name=module.name, spec=spec)

    # Explicitly throw away type information from original module
    module.symbol_attrs.clear()
    assert all(var.type is None for var in other_module_copy.variables)
    assert all(var.scope is not None for var in other_module_copy.variables)

    # fgen of the rescoped copy should work
    assert fgen(module_copy) == ref_fgen

    # fgen of the not rescoped copy should fail because the scope of the variables went away
    with pytest.raises(AttributeError):
        fgen(other_module_copy)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parsing fails without dummy module provided')]))
def test_module_rescope_clone(frontend, tmp_path):
    """
    Test the rescoping of variables in clone.
    """
    fcode = """
module test_module_rescope_clone
  use some_mod, only: ext1
  implicit none
  integer :: a, b, c
end module test_module_rescope_clone
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    ref_fgen = fgen(module)

    # Create a copy of the module with rescoping and make sure all symbols are in the right scope
    module_copy = module.clone()

    for var in FindTypedSymbols().visit(module_copy.spec):
        assert var.scope is module_copy

    # Create another copy of the nested subroutine without rescoping
    other_module_copy = module.clone(rescope_symbols=False, symbol_attrs=None)

    # Explicitly throw away type information from original module
    module.symbol_attrs.clear()
    assert all(var.type is None for var in other_module_copy.variables)
    assert all(var.scope is not None for var in other_module_copy.variables)

    # fgen of the rescoped copy should work
    assert fgen(module_copy) == ref_fgen

    # fgen of the not rescoped copy should fail because the scope of the variables went away
    with pytest.raises(AttributeError):
        fgen(other_module_copy)

@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'Parsing fails without dummy module provided')]
))
def test_module_deep_clone(frontend, tmp_path):
    """
    Test the rescoping of variables in clone with nested scopes.
    """
    fcode = """
module test_module_rescope_clone
  use parkind1, only : jpim, jprb
  implicit none

  integer :: n

  real :: array(n)

  type my_type
    real :: vector(n)
    real :: matrix(n, n)
  end type

end module test_module_rescope_clone
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Deep-copy/clone the module
    new_module = module.clone()

    n = [v for v in FindVariables().visit(new_module.spec) if v.name == 'n'][0]
    n_decl = FindNodes(VariableDeclaration).visit(new_module.spec)[0]

    # Remove the declaration of `n` and replace it with `3`
    new_module.spec = Transformer({n_decl: None}).visit(new_module.spec)
    new_module.spec = SubstituteExpressions({n: Literal(3)}).visit(new_module.spec)

    # Check the new module has been changed
    assert len(FindNodes(VariableDeclaration).visit(new_module.spec)) == 1
    new_type_decls = FindNodes(VariableDeclaration).visit(new_module['my_type'].body)
    assert len(new_type_decls) == 2
    assert new_type_decls[0].symbols[0] == 'vector(3)'
    assert new_type_decls[1].symbols[0] == 'matrix(3, 3)'

    # Check the old one has not changed
    assert len(FindNodes(VariableDeclaration).visit(module.spec)) == 2
    type_decls = FindNodes(VariableDeclaration).visit(module['my_type'].body)
    assert len(type_decls) == 2
    assert type_decls[0].symbols[0] == 'vector(n)'
    assert type_decls[1].symbols[0] == 'matrix(n, n)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_access_spec_none(frontend, tmp_path):
    """
    Test correct parsing without access-spec statements
    """
    fcode = """
module test_access_spec_mod
    implicit none

    integer pub_var = 1
contains
    subroutine routine
        integer i
        i = pub_var
    end subroutine routine
end module test_access_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Check module properties
    assert module.default_access_spec is None
    assert module.public_access_spec is ()
    assert module.private_access_spec is ()

    # Check backend output
    code = module.to_fortran().upper()
    assert 'PUBLIC' not in code
    assert 'PRIVATE' not in code

    # Check that property has not propagated to symbol type
    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is None
    assert pub_var.type.private is None

    # Check properties after clone
    new_module = module.clone(
        default_access_spec='PUBLIC', public_access_spec='PUB_VAR',
        private_access_spec='ROUTINE'
    )
    assert new_module.default_access_spec == 'public'
    assert new_module.public_access_spec == ('pub_var',)
    assert new_module.private_access_spec == ('routine',)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Inlines access-spec as declaration attr')]))
def test_module_access_spec_private(frontend, tmp_path):
    """
    Test correct parsing of access-spec statements with default private
    """
    fcode = """
module test_access_spec_mod
    implicit none
    private
    public :: pub_var, routine
    PRIVATE OTHER_PRIVATE_VAR

    integer pub_var = 1
    integer private_var = 2
    integer other_private_var = 3
contains
    subroutine routine
        integer i
        i = pub_var
    end subroutine routine
end module test_access_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Check module properties
    assert module.default_access_spec == 'private'
    assert module.public_access_spec == ('pub_var', 'routine')
    assert module.private_access_spec == ('other_private_var',)

    # Check backend output
    code = module.to_fortran().upper()
    assert 'PUBLIC\n' not in code
    assert 'PUBLIC :: PUB_VAR, ROUTINE' in code
    assert 'PRIVATE\n' in code
    assert 'PRIVATE :: OTHER_PRIVATE_VAR' in code

    # Check that property has not propagated to symbol type
    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is None
    assert pub_var.type.private is None

    # Check properties after clone
    new_module = module.clone(private_access_spec=None)
    assert new_module.default_access_spec == 'private'
    assert new_module.public_access_spec == ('pub_var', 'routine')
    assert new_module.private_access_spec == ()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Inlines access-spec as declaration attr')]))
def test_module_access_spec_public(frontend, tmp_path):
    """
    Test correct parsing of access-spec statements with default public
    """
    fcode = """
module test_access_spec_mod
    implicit none
    PUBLIC
    PUBLIC ROUTINE
    private :: private_var, other_private_var

    integer pub_var = 1
    integer private_var = 2
    integer other_private_var = 3
contains
    subroutine routine
        integer i
        i = pub_var
    end subroutine routine
end module test_access_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Check module properties
    assert module.default_access_spec == 'public'
    assert module.public_access_spec == ('routine', )
    assert module.private_access_spec == ('private_var', 'other_private_var')

    # Check backend output
    code = module.to_fortran().upper()
    assert 'PUBLIC\n' in code
    assert 'PUBLIC :: ROUTINE' in code
    assert 'PRIVATE\n' not in code
    assert 'PRIVATE :: PRIVATE_VAR, OTHER_PRIVATE_VAR' in code

    # Check that property has not propagated to symbol type
    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is None
    assert pub_var.type.private is None

    # Check properties after clone
    new_module = module.clone(
        default_access_spec='PRivate', public_access_spec=('ROUTINE', 'pub_var')
    )
    assert new_module.default_access_spec == 'private'
    assert new_module.public_access_spec == ('routine', 'pub_var')
    assert new_module.private_access_spec == ('private_var', 'other_private_var')


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_access_attr(frontend, tmp_path):
    """
    Test correct parsing of access-spec attributes
    """
    fcode = """
module test_access_attr_mod
    implicit none
    private
    integer, public :: pub_var
    integer :: unspecified_var
    integer, private :: priv_var
    integer :: other_var
end module test_access_attr_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    code = module.to_fortran().upper()

    priv_var = module.variable_map['priv_var']
    assert priv_var.type.private is True
    assert priv_var.type.public is None

    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is True
    assert pub_var.type.private is None

    unspecified_var = module.variable_map['unspecified_var']
    other_var = module.variable_map['other_var']

    assert unspecified_var.type.public is None
    assert other_var.type.public is None

    if frontend == OMNI:  # OMNI applies access spec to each variable
        assert code.count('PRIVATE') == 3
        assert unspecified_var.type.private is True
        assert other_var.type.private is True
    else:
        assert code.count('PRIVATE') == 2
        assert unspecified_var.type.private is None
        assert other_var.type.private is None
    assert code.count('PUBLIC') == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_rename_imports_with_definitions(frontend, tmp_path):
    """
    Test use statement with rename lists
    """
    fcode_mod1 = """
module test_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_rename_mod
    """.strip()

    fcode_mod2 = """
module test_other_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_other_rename_mod
    """.strip()

    fcode_mod3 = """
module some_mod
    use test_rename_mod, first_var1 => var1, first_var3 => var3
    use test_other_rename_mod, only: second_var1 => var1
    use test_other_rename_mod, only: other_var2 => var2, other_var3 => var3
    implicit none
end module some_mod
    """.strip()

    mod1 = Module.from_source(fcode_mod1, frontend=frontend, xmods=[tmp_path])
    mod2 = Module.from_source(fcode_mod2, frontend=frontend, xmods=[tmp_path])
    mod3 = Module.from_source(fcode_mod3, frontend=frontend, xmods=[tmp_path], definitions=[mod1, mod2])

    # Check all entries exist in the symbol table
    mod1_imports = {
        'first_var1': 'var1',
        'var2': None,
        'first_var3': 'var3'
    }
    mod2_imports = {
        'second_var1': 'var1',
        'other_var2': 'var2',
        'other_var3': 'var3'
    }
    expected_symbols = list(mod1_imports) + list(mod2_imports)
    for s in expected_symbols:
        assert s in mod3.symbol_attrs

    # Check that var1 has note been imported under that name
    assert 'var1' not in mod3.symbol_attrs

    # Verify correct symbol attributes
    for s, use_name in mod1_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is mod1
        assert mod3.symbol_attrs[s].use_name == use_name
        assert mod3.symbol_attrs[s].compare(mod1.symbol_attrs[use_name or s], ignore=('imported', 'module', 'use_name'))
    for s, use_name in mod2_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is mod2
        assert mod3.symbol_attrs[s].use_name == use_name
        assert mod3.symbol_attrs[s].compare(mod2.symbol_attrs[use_name or s], ignore=('imported', 'module', 'use_name'))

    # Verify Import IR node
    for imprt in FindNodes(Import).visit(mod3.spec):
        if imprt.module == 'test_rename_mod':
            assert imprt.rename_list
            assert not imprt.symbols
            assert 'var1' in dict(imprt.rename_list)
            assert 'var3' in dict(imprt.rename_list)
        else:
            assert not imprt.rename_list
            assert imprt.symbols

    # Verify fgen output
    fcode = fgen(mod3)
    for s, use_name in mod1_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode
    for s, use_name in mod2_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_rename_imports_no_definitions(frontend, tmp_path):
    """
    Test use statement with rename lists when definitions are not available
    """
    fcode_mod1 = """
module test_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_rename_mod
    """.strip()

    fcode_mod2 = """
module test_other_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_other_rename_mod
    """.strip()

    mod1 = Module.from_source(fcode_mod1, frontend=frontend, xmods=[tmp_path])
    mod2 = Module.from_source(fcode_mod2, frontend=frontend, xmods=[tmp_path])

    fcode_mod3 = """
module some_mod
    use test_rename_mod, first_var1 => var1, first_var3 => var3
    use test_other_rename_mod, only: second_var1 => var1
    use test_other_rename_mod, only: other_var2 => var2, other_var3 => var3
    implicit none
end module some_mod
    """.strip()

    mod3 = Module.from_source(fcode_mod3, frontend=frontend, xmods=[tmp_path])

    # Check all entries exist in the symbol table
    mod1_imports = {
        'first_var1': 'var1',
        'first_var3': 'var3'
    }
    mod2_imports = {
        'second_var1': 'var1',
        'other_var2': 'var2',
        'other_var3': 'var3'
    }
    expected_symbols = list(mod1_imports) + list(mod2_imports)
    for s in expected_symbols:
        assert s in mod3.symbol_attrs

    # Check that var1 has note been imported under that name
    assert 'var1' not in mod3.symbol_attrs
    assert 'var2' not in mod3.symbol_attrs

    # Verify correct symbol attributes
    for s, use_name in mod1_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is None
        assert mod3.symbol_attrs[s].use_name == use_name
    for s, use_name in mod2_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is None
        assert mod3.symbol_attrs[s].use_name == use_name

    # Verify Import IR node
    for imprt in FindNodes(Import).visit(mod3.spec):
        if imprt.module == 'test_rename_mod':
            assert imprt.rename_list
            assert not imprt.symbols
            assert 'var1' in dict(imprt.rename_list)
            assert 'var3' in dict(imprt.rename_list)
        else:
            assert not imprt.rename_list
            assert imprt.symbols

    # Verify fgen output
    fcode = fgen(mod3)
    for s, use_name in mod1_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode
    for s, use_name in mod2_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OFP, 'hasModuleNature on use-stmt but without conveying actual nature')]
))
def test_module_use_module_nature(frontend, tmp_path):
    """
    Test module natures attributes in ``USE`` statements
    """
    mcode = """
module iso_fortran_env
    use, intrinsic :: iso_c_binding, only: int16 => c_int16_t
    implicit none
    integer, parameter :: int8 = int16
end module iso_fortran_env
    """.strip()

    fcode = """
module module_nature_mod
    implicit none
contains
    subroutine inquire_my_kinds(i8, i16)
        use, non_intrinsic :: iso_fortran_env, only: int8, int16
        integer, intent(out) :: i8, i16
        i8 = int8
        i16 = int16
    end subroutine inquire_my_kinds
    subroutine inquire_kinds(i8, i16)
        use, intrinsic :: iso_fortran_env, only: int8, int16
        integer, intent(out) :: i8, i16
        i8 = int8
        i16 = int16
    end subroutine inquire_kinds
end module module_nature_mod
    """.strip()

    ext_mod = Module.from_source(mcode, frontend=frontend, xmods=[tmp_path])

    # Check properties on the Import IR node in the external module
    assert ext_mod.imported_symbols == ('int16',)
    imprt = FindNodes(Import).visit(ext_mod.spec)[0]
    assert imprt.nature.lower() == 'intrinsic'
    assert imprt.module.lower() == 'iso_c_binding'
    assert ext_mod.imported_symbol_map['int16'].type.imported is True
    assert ext_mod.imported_symbol_map['int16'].type.module is None

    if frontend == OMNI:
        # OMNI throws Syntax Error on NON_INTRINSIC...
        fcode = fcode.replace('use, non_intrinsic ::', 'use')

    mod = Module.from_source(fcode, frontend=frontend, definitions=[ext_mod], xmods=[tmp_path])

    # Check properties on the Import IR node in both routines
    my_kinds = mod['inquire_my_kinds']
    kinds = mod['inquire_kinds']

    assert set(my_kinds.imported_symbols) == {'int8', 'int16'}
    assert set(kinds.imported_symbols) == {'int8', 'int16'}

    my_import_map = {s.name: imprt for imprt in FindNodes(Import).visit(my_kinds.spec) for s in imprt.symbols}
    import_map = {s.name: imprt for imprt in FindNodes(Import).visit(kinds.spec) for s in imprt.symbols}

    assert my_import_map['int8'] is my_import_map['int16']
    assert import_map['int8'] is import_map['int16']

    if frontend == OMNI:
        assert my_import_map['int8'].nature is None
    else:
        assert my_import_map['int8'].nature.lower() == 'non_intrinsic'
    assert my_import_map['int8'].module.lower() == 'iso_fortran_env'
    assert import_map['int8'].nature.lower() == 'intrinsic'
    assert import_map['int8'].module.lower() == 'iso_fortran_env'

    # Check type annotations for imported symbols
    assert all(s.type.imported is True for s in my_kinds.imported_symbols)
    assert all(s.type.imported is True for s in kinds.imported_symbols)

    assert my_kinds.imported_symbol_map['int8'].type.module is ext_mod
    assert my_kinds.imported_symbol_map['int16'].type.module is ext_mod

    assert kinds.imported_symbol_map['int8'].type.module is None
    assert kinds.imported_symbol_map['int16'].type.module is None

    # Sanity check fgen
    assert 'use, intrinsic' in ext_mod.to_fortran().lower()
    if frontend != OMNI:
        assert 'use, non_intrinsic' in my_kinds.to_fortran().lower()
    assert 'use, intrinsic' in kinds.to_fortran().lower()

    # Verify JIT compile
    ext_filepath = tmp_path/f'{ext_mod.name}.f90'
    filepath = tmp_path/f'{mod.name}_{frontend}.f90'
    jit_ext = jit_compile(ext_mod, filepath=ext_filepath, objname=ext_mod.name)
    jit_mod = jit_compile(mod, filepath=filepath, objname=mod.name)
    my_kinds_func = jit_mod.inquire_my_kinds
    kinds_func = jit_mod.inquire_kinds

    my_i8, my_i16 = my_kinds_func()
    i8, i16 = kinds_func()

    assert my_i8 == my_i16
    assert i8 < i16
    assert my_i8 == i16
    assert my_i8 == jit_ext.int8

    clean_test(filepath)
    clean_test(ext_filepath)


@pytest.mark.parametrize('spec,part_lengths', [
    ('', (0, 0, 0)),
    ("""
implicit none
integer :: var1
integer :: var2
integer :: var3
    """.strip(), (0, 1, 3)),
    ("""
use header_mod
implicit none
integer :: var1
    """.strip(), (1, 1, 1)),
    ("""
use header_mod
integer :: var1
    """.strip(), (1, 0, 1)),
])
@pytest.mark.parametrize('frontend', available_frontends())
def test_module_spec_parts(frontend, spec, part_lengths, tmp_path):
    """Test the :attr:`spec_parts` property of :class:`Module`"""

    header_mod_fcode = """
module header_mod
    implicit none
    integer, parameter :: param1 = 1
end module header_mod
    """.strip()
    header_mod = Module.from_source(header_mod_fcode, frontend=frontend, xmods=[tmp_path])

    docstring = '! This should become the doc string\n'
    fcode = f"""
module spec_parts
{docstring if frontend != OMNI else ''}{spec}
end module spec_parts
    """.strip()

    module = Module.from_source(fcode, definitions=header_mod, frontend=frontend, xmods=[tmp_path])
    assert isinstance(module.spec_parts, tuple)
    assert all(isinstance(p, tuple) for p in module.spec_parts)

    if frontend == OMNI:
        # OMNI removes any 'IMPLICIT' statements so the middle part is always empty
        part_lengths = (part_lengths[0], 0, part_lengths[2])
    else:
        # OMNI _conveniently_ puts any use statements _before_ the docstring for
        # absolutely zero sensible reasons, so it would be purely based on good luck
        # and favourable circumstances to extract the right amount of comments for the
        # docstring with that _fantastic_ frontend...
        assert isinstance(module.docstring, tuple) and len(module.docstring) == 1

    assert part_lengths == tuple(len(p) for p in module.spec_parts)


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_comparison(frontend, tmp_path):
    """
    Test that string-equivalence works on relevant components.
    """

    fcode = """
module a_module
  integer, parameter :: x = 2
  integer, parameter :: y = 3

  type derived_type
    real :: array(x, y)
  end type derived_type
contains

  subroutine my_routine(pt)
    type(derived_type) :: pt
    pt%array(:,:) = 42.0
  end subroutine my_routine
end module a_module
"""

    # Two distinct string-equivalent subroutine objects
    m1 = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    m2 = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    assert m1.symbol_attrs == m2.symbol_attrs
    assert m1.spec == m2.spec
    assert m1.contains == m2.contains
    assert m1 == m2


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_comparison_case_sensitive(frontend, tmp_path):
    """
    Test that semantic, but no string-equivalence evaluates as not eqal
    """

    fcode = """
module a_module
  integer, parameter :: x = 2
  integer, parameter :: y = 3

  type derived_type
    real :: array(x, y)
  end type derived_type
contains

  subroutine my_routine(pt)
    type(derived_type) :: pt
    pt%array(:,:) = 42.0
  end subroutine my_routine
end module a_module
"""

    # Two distinct string-equivalent subroutine objects
    m1 = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    m2 = Module.from_source(fcode.replace('pt%array', 'pT%aRrAy'), frontend=frontend, xmods=[tmp_path])

    assert not 'pT%aRrAy' in fgen(m1)
    if frontend != OMNI:  # OMNI always downcases!
        assert 'pT%aRrAy' in fgen(m2)

    # Since the routine is different the procedure type will be!
    assert not m1.symbol_attrs == m2.symbol_attrs
    # OMNI source file paths are affected by the string change, which
    # are attached and check to each source node object
    if frontend != OMNI:
        assert m1.spec == m2.spec
    assert not m1.contains == m2.contains
    assert not m1 == m2


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_contains_auto_insert(frontend, tmp_path):
    """
    Test that `CONTAINS` keyword is automatically inserted into the `contains` section
    of a :any:`ProgramUnit` object.
    """
    fcode_mod = """
module empty_mod
    implicit none
end module empty_mod
    """.strip()
    fcode_routine1 = """
subroutine routine1
end subroutine routine1
    """.strip()
    fcode_routine2 = """
subroutine routine2
end subroutine routine2
    """.strip()

    module = Module.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])
    routine1 = Subroutine.from_source(fcode_routine1, frontend=frontend, xmods=[tmp_path])
    routine2 = Subroutine.from_source(fcode_routine2, frontend=frontend, xmods=[tmp_path])

    assert module.contains is None
    assert routine1.contains is None

    routine1 = routine1.clone(contains=routine2)
    assert isinstance(routine1.contains, Section)
    assert isinstance(routine1.contains.body[0], Intrinsic)
    assert routine1.contains.body[0].text == 'CONTAINS'

    module = module.clone(contains=routine1)
    assert isinstance(module.contains, Section)
    assert isinstance(module.contains.body[0], Intrinsic)
    assert module.contains.body[0].text == 'CONTAINS'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('only_list', [True, False])
@pytest.mark.parametrize('complete_tree', [True, False])
def test_module_missing_imported_symbol(frontend, only_list, complete_tree, tmp_path):
    fcode_mod1 = """
module mod1
    implicit none
    integer, parameter :: a = 1, b=2
end module mod1
    """.strip()

    fcode_mod2 = f"""
module mod2
    use mod1{', only: a, b' if only_list else ''}
    implicit none
end module mod2
    """.strip()

    fcode_driver = """
subroutine driver
    use mod2, only: a, b
    implicit none
    integer c
    c = a + b
end subroutine driver
    """.strip()

    mod1 = Module.from_source(fcode_mod1, frontend=frontend, xmods=[tmp_path])
    if complete_tree:
        modules = [mod1]
    else:
        modules = []
    modules += [Module.from_source(fcode_mod2, frontend=frontend, definitions=modules, xmods=[tmp_path])]
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=modules, xmods=[tmp_path])

    a = driver.symbol_map['a']
    b = driver.symbol_map['b']

    if complete_tree:
        assert isinstance(a, Scalar)
        assert a.type.dtype is BasicType.INTEGER
        assert isinstance(b, Scalar)
        assert b.type.dtype is BasicType.INTEGER
    else:
        assert isinstance(a, DeferredTypeSymbol)
        assert a.type.dtype is BasicType.DEFERRED
        assert isinstance(b, DeferredTypeSymbol)
        assert b.type.dtype is BasicType.DEFERRED

    assert a.type.imported
    assert b.type.imported
    assert a.type.module is modules[-1]
    assert b.type.module is modules[-1]


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_all_imports(frontend, tmp_path):
    fcode = {
        'header_a': (
        #--------
"""
module module_all_imports_header_a_mod
implicit none

integer, parameter :: a = 1
integer, parameter :: b = 2
end module module_all_imports_header_a_mod
"""
        ).strip(),
        'header_b': (
        #--------
"""
module module_all_imports_header_b_mod
implicit none

integer, parameter :: a = 2
integer, parameter :: b = 1
end module module_all_imports_header_b_mod
"""
        ).strip(),
        'routine': (
        #-------
"""
module module_all_imports_routine_mod
    use module_all_imports_header_a_mod, only: a
    use module_all_imports_header_b_mod, only: b_b => b
    implicit none
contains
    subroutine routine
        use module_all_imports_header_a_mod, only: b
        use module_all_imports_header_b_mod, only: a
        implicit none
        integer val
        val = a + b + b_b
    end subroutine routine
end module module_all_imports_routine_mod
"""
        ).strip()
    }

    header_a = Module.from_source(fcode['header_a'], frontend=frontend, xmods=[tmp_path])
    header_b = Module.from_source(fcode['header_b'], frontend=frontend, xmods=[tmp_path])
    routine_mod = Module.from_source(fcode['routine'], definitions=(header_a, header_b), frontend=frontend, xmods=[tmp_path])
    routine = routine_mod['routine']

    assert routine_mod.parents == ()
    assert routine.parents == (routine_mod,)

    assert routine_mod.all_imports == routine_mod.imports
    assert routine.all_imports == routine.imports + routine_mod.imports

    assert routine.symbol_map['a'].type.module is header_b
    assert routine_mod.symbol_map['a'].type.module is header_a
    assert routine.symbol_map['b'].type.module is header_a
    assert routine_mod.symbol_map['b_b'].type.module is header_b
    assert routine_mod.symbol_map['b_b'].type.use_name == 'b'


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_enrichment_within_file(frontend, tmp_path):
    fcode = """
module foo
  implicit none
  integer, parameter :: j = 16

  contains
    integer function SUM(v)
      implicit none
      integer, intent(in) :: v
      SUM = v + 1
    end function SUM
end module foo

module test
    use foo
    implicit none
    integer, parameter :: rk = selected_real_kind(12)
    integer, parameter :: ik = selected_int_kind(9)
contains
    subroutine calc (n, res)
        integer, intent(in) :: n
        real(kind=rk), intent(inout) :: res
        integer(kind=ik) :: i
        do i = 1, n
            res = res + SUM(j)
        end do
    end subroutine calc
end module test
"""

    source = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = source['calc']
    calls = list(FindInlineCalls().visit(routine.body))
    assert len(calls) == 1
    assert calls[0].function == 'sum'
    assert calls[0].function.type.imported
    assert calls[0].function.type.module is source['foo']
    assert calls[0].function.type.dtype.procedure is source['sum']
    if frontend != OMNI:
        # OMNI inlines parameters
        assert calls[0].arguments[0].type.dtype == BasicType.INTEGER
        assert calls[0].arguments[0].type.imported
        assert calls[0].arguments[0].type.parameter
        assert calls[0].arguments[0].type.initial == 16
        assert calls[0].arguments[0].type.module is source['foo']
