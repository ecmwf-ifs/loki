import pytest

from loki import (
    OFP, OMNI, FP, Module, Declaration, TypeDef, fexprgen, DataType, Statement, FindNodes)


@pytest.mark.parametrize('frontend', [FP, OFP, OMNI])
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


@pytest.mark.parametrize('frontend', [FP, OFP, OMNI])
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
    assert 'array' in pt_ext.type.dtype.variable_map
    a = pt_ext.type.dtype.variable_map['array']
    assert a.type.dtype == DataType.REAL
    assert fexprgen(a.shape) == exptected_array_shape

    # Check the LHS of the assignment has correct meta-data
    stmt = FindNodes(Statement).visit(routine.body)[0]
    pt_ext_arr = stmt.target
    assert pt_ext_arr.type.dtype == DataType.REAL
    assert fexprgen(pt_ext_arr.shape) == exptected_array_shape


@pytest.mark.parametrize('frontend', [
    FP,
    pytest.param(OFP, marks=pytest.mark.xfail(reason='Typedefs not yet supported in frontend')),
    OMNI
])
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

    fcode_module = """
module a_module
  use external_mod, only: ext_type
  implicit none

  type nested_type
    type(ext_type) :: ext
  end type nested_type
contains

  subroutine my_routine(pt)
    type(nested_type) :: pt
    pt%ext%array(:,:) = 42.0
  end subroutine my_routine
end module a_module
"""

    external = Module.from_source(fcode_external, frontend=frontend)
    assert'ext_type' in external.typedefs

    module = Module.from_source(fcode_module, frontend=frontend, definitions=external)
    nested = module.typedefs['nested_type']
    ext = nested.variables[0]

    # OMNI resolves explicit shape parameters in the frontend parser
    exptected_array_shape = '(1:2, 1:3)' if frontend == OMNI else '(x, y)'

    # Check that the `array` variable in the `ext` type is found and
    # has correct type and shape info
    assert 'array' in ext.type.dtype.variable_map
    a = ext.type.dtype.variable_map['array']
    assert a.type.dtype == DataType.REAL
    assert fexprgen(a.shape) == exptected_array_shape

    # Check the routine has got type and shape info too
    routine = module['my_routine']
    pt = routine.variables[0]
    pt_ext = pt.type.dtype.variable_map['ext']
    assert 'array' in pt_ext.type.dtype.variable_map
    pt_ext_a = pt_ext.type.dtype.variable_map['array']
    assert pt_ext_a.type.dtype == DataType.REAL
    assert fexprgen(pt_ext_a.shape) == exptected_array_shape

    # Check the LHS of the assignment has correct meta-data
    stmt = FindNodes(Statement).visit(routine.body)[0]
    pt_ext_arr = stmt.target
    assert pt_ext_arr.type.dtype == DataType.REAL
    assert fexprgen(pt_ext_arr.shape) == exptected_array_shape


@pytest.mark.parametrize('frontend', [FP, OFP, OMNI])
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
    assert 'array' in pt.type.dtype.variable_map
    arr = pt.type.dtype.variable_map['array']
    assert arr.type.dtype == DataType.REAL
    assert fexprgen(arr.shape) == exptected_array_shape


@pytest.mark.parametrize('frontend', [
    FP,
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Loki annotations break frontend parser'))
])
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


@pytest.mark.parametrize('frontend', [
    FP,
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Loki annotations break frontend parser'))
])
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

    pt_x = parent.variables[0].type.dtype.variable_map['x']
    assert fexprgen(pt_x.shape) == '(size,)'
