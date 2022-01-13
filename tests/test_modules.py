import pytest

from conftest import available_frontends
from loki import (
    OMNI, Module, Subroutine, Declaration, TypeDef, fexprgen,
    BasicType, Assignment, FindNodes, FindInlineCalls, FindTypedSymbols,
    Transformer, fgen, SymbolAttributes, Variable, Import
)


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_from_source(frontend):
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
    module = Module.from_source(fcode, frontend=frontend)
    assert len([o for o in module.spec.body if isinstance(o, Declaration)]) == 2
    assert len([o for o in module.spec.body if isinstance(o, TypeDef)]) == 1
    assert 'derived_type' in module.typedefs
    assert len(module.routines) == 1
    assert module.routines[0].name == 'my_routine'
    if frontend != OMNI:
        assert module.source.string == fcode
        assert module.source.lines == (1, fcode.count('\n') + 1)


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_external_typedefs_subroutine(frontend):
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

    external = Module.from_source(fcode_external, frontend=frontend)
    assert'ext_type' in external.typedefs

    module = Module.from_source(fcode_module, frontend=frontend, definitions=external)
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
def test_module_external_typedefs_type(frontend):
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

    external = Module.from_source(fcode_external, frontend=frontend)
    assert 'ext_type' in external.typedefs

    other = Module.from_source(fcode_other, frontend=frontend)
    assert 'other_type' in other.typedefs

    if frontend != OMNI:  # OMNI needs to know imported modules
        module = Module.from_source(fcode_module, frontend=frontend)
        assert 'ext_type' in module.symbols
        assert module.symbols['ext_type'].dtype is BasicType.DEFERRED
        assert 'other_type' not in module.symbols
        assert 'other_type' not in module['other_routine'].symbols
        assert module['other_routine'].symbols['pt'].dtype.typedef is BasicType.DEFERRED

    module = Module.from_source(fcode_module, frontend=frontend, definitions=[external, other])
    nested = module.typedefs['nested_type']
    ext = nested.variables[0]

    # Verify correct attachment of type information
    assert 'ext_type' in module.symbols
    assert isinstance(module.symbols['ext_type'].dtype.typedef, TypeDef)
    assert isinstance(nested.symbols['ext'].dtype.typedef, TypeDef)
    assert isinstance(module['my_routine'].symbols['pt'].dtype.typedef, TypeDef)
    assert isinstance(module['my_routine'].symbols['pt%ext'].dtype.typedef, TypeDef)
    assert 'other_type' in module.symbols
    assert 'other_type' not in module['other_routine'].symbols
    assert isinstance(module.symbols['other_type'].dtype.typedef, TypeDef)
    assert isinstance(module['other_routine'].symbols['pt'].dtype.typedef, TypeDef)

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
def test_module_nested_types(frontend):
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

    module = Module.from_source(fcode, frontend=frontend)
    parent = module.typedefs['parent_type']
    pt = parent.variables[0]
    assert 'array' in pt.variable_map
    arr = pt.variable_map['array']
    assert arr.type.dtype == BasicType.REAL
    assert fexprgen(arr.shape) == exptected_array_shape


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Loki annotation break parser')]))
def test_dimension_pragmas(frontend):
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
    module = Module.from_source(fcode, frontend=frontend)
    mytype = module.typedefs['mytype']
    assert fexprgen(mytype.variables[0].shape) == '(size,)'


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Loki annotation break parser')]))
def test_nested_types_dimension_pragmas(frontend):
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
    module = Module.from_source(fcode, frontend=frontend)
    parent = module.typedefs['parent_type']
    child = module.typedefs['sub_type']
    assert fexprgen(child.variables[0].shape) == '(size,)'

    pt_x = parent.variables[0].variable_map['x']
    assert fexprgen(pt_x.shape) == '(size,)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_internal_function_call(frontend):
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
    module = Module.from_source(fcode, frontend=frontend)
    routine = module['test_inline_call']

    inline_calls = list(FindInlineCalls().visit(routine.body))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.name == 'util_fct'
    assert inline_calls[0].parameters[0] == 'v2'
    assert inline_calls[0].parameters[1] == 'v1'

    assert isinstance(module.symbols['util_fct'].dtype.procedure, Subroutine)
    assert module.symbols['util_fct'].dtype.is_function


@pytest.mark.parametrize('frontend', available_frontends())
def test_external_function_call(frontend):
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
    module = Module.from_source(fcode_util, frontend=frontend)
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)

    inline_calls = list(FindInlineCalls().visit(routine.body))
    assert len(inline_calls) == 1
    assert inline_calls[0].function.name == 'util_fct'
    assert inline_calls[0].parameters[0] == 'v2'
    assert inline_calls[0].parameters[1] == 'v1'


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_variables_add_remove(frontend):
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
    module = Module.from_source(fcode, frontend=frontend)
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
def test_module_rescope_variables(frontend):
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

    module = Module.from_source(fcode, frontend=frontend)
    ref_fgen = fgen(module)

    # Create a copy of the module with rescoping and make sure all symbols are in the right scope
    spec = Transformer().visit(module.spec)
    module_copy = Module(name=module.name, spec=spec, rescope_variables=True)

    for var in FindTypedSymbols().visit(module_copy.spec):
        assert var.scope is module_copy

    # Create another copy of the nested subroutine without rescoping
    spec = Transformer().visit(module.spec)
    other_module_copy = Module(name=module.name, spec=spec)

    # Explicitly throw away type information from original module
    module.symbols.clear()
    assert all(var.type is None for var in other_module_copy.variables)
    assert all(var.scope is not None for var in other_module_copy.variables)

    # fgen of the rescoped copy should work
    assert fgen(module_copy) == ref_fgen

    # fgen of the not rescoped copy should fail because the scope of the variables went away
    with pytest.raises(AttributeError):
        fgen(other_module_copy)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Parsing fails without dummy module provided')]))
def test_module_rescope_clone(frontend):
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

    module = Module.from_source(fcode, frontend=frontend)
    ref_fgen = fgen(module)

    # Create a copy of the module with rescoping and make sure all symbols are in the right scope
    module_copy = module.clone()

    for var in FindTypedSymbols().visit(module_copy.spec):
        assert var.scope is module_copy

    # Create another copy of the nested subroutine without rescoping
    other_module_copy = module.clone(rescope_variables=False, symbols=None)

    # Explicitly throw away type information from original module
    module.symbols.clear()
    assert all(var.type is None for var in other_module_copy.variables)
    assert all(var.scope is not None for var in other_module_copy.variables)

    # fgen of the rescoped copy should work
    assert fgen(module_copy) == ref_fgen

    # fgen of the not rescoped copy should fail because the scope of the variables went away
    with pytest.raises(AttributeError):
        fgen(other_module_copy)


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_access_spec(frontend):
    """
    Test correct parsing of access-spec attributes
    """
    fcode = """
module test_access_spec_mod
    implicit none
    private
    integer, public :: pub_var
    integer :: unspecified_var
    integer, private :: priv_var
    integer :: other_var
end module test_access_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
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
def test_module_rename_imports_with_definitions(frontend):
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

    mod1 = Module.from_source(fcode_mod1, frontend=frontend)
    mod2 = Module.from_source(fcode_mod2, frontend=frontend)
    mod3 = Module.from_source(fcode_mod3, frontend=frontend, definitions=[mod1, mod2])

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
        assert s in mod3.symbols

    # Check that var1 has note been imported under that name
    assert 'var1' not in mod3.symbols

    # Verify correct symbol attributes
    for s, use_name in mod1_imports.items():
        assert mod3.symbols[s].imported
        assert mod3.symbols[s].module is mod1
        assert mod3.symbols[s].use_name == use_name
        assert mod3.symbols[s].compare(mod1.symbols[use_name or s], ignore=('imported', 'module', 'use_name'))
    for s, use_name in mod2_imports.items():
        assert mod3.symbols[s].imported
        assert mod3.symbols[s].module is mod2
        assert mod3.symbols[s].use_name == use_name
        assert mod3.symbols[s].compare(mod2.symbols[use_name or s], ignore=('imported', 'module', 'use_name'))

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
def test_module_rename_imports_no_definitions(frontend):
    """
    Test use statement with rename lists when definitions are not available
    (Note: for OMNI this relies on the fact that the twin test ran first to
    produce the xmod file)
    """
    fcode_mod3 = """
module some_mod
    use test_rename_mod, first_var1 => var1, first_var3 => var3
    use test_other_rename_mod, only: second_var1 => var1
    use test_other_rename_mod, only: other_var2 => var2, other_var3 => var3
    implicit none
end module some_mod
    """.strip()

    mod3 = Module.from_source(fcode_mod3, frontend=frontend)

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
        assert s in mod3.symbols

    # Check that var1 has note been imported under that name
    assert 'var1' not in mod3.symbols
    assert 'var2' not in mod3.symbols

    # Verify correct symbol attributes
    for s, use_name in mod1_imports.items():
        assert mod3.symbols[s].imported
        assert mod3.symbols[s].module is None
        assert mod3.symbols[s].use_name == use_name
    for s, use_name in mod2_imports.items():
        assert mod3.symbols[s].imported
        assert mod3.symbols[s].module is None
        assert mod3.symbols[s].use_name == use_name

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
