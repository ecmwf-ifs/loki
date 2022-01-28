from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, clean_test, available_frontends
from loki import (
    OMNI, Module, Subroutine, FindVariables, IntLiteral,
    RangeIndex, BasicType, DeferredTypeSymbol, Array, DerivedType, TypeDef,
    config, fgen, FindNodes, Associate
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(name='reset_frontend_mode')
def fixture_reset_frontend_mode():
    original_frontend_mode = config['frontend-strict-mode']
    yield
    config['frontend-strict-mode'] = original_frontend_mode


@pytest.mark.parametrize('frontend', available_frontends())
def test_simple_loops(here, frontend):
    """
    Test simple vector/matrix arithmetic with a derived type
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit
contains

  subroutine simple_loops(item)
    type(explicit), intent(inout) :: item
    integer :: i, j, n

    n = 3
    do i=1, n
       item%vector(i) = item%vector(i) + item%scalar
    end do

    do j=1, n
       do i=1, n
          item%matrix(i, j) = item%matrix(i, j) + item%scalar
       end do
    end do
  end subroutine simple_loops
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/(f'derived_types_simple_loops_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    mod.simple_loops(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_indexing_explicit(here, frontend):
    """
    Test simple vector/matrix arithmetic with a derived type
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit
contains

  subroutine array_indexing_explicit(item)
    type(explicit), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.
    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do
  end subroutine array_indexing_explicit
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/(f'derived_types_array_indexing_explicit_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    mod.array_indexing_explicit(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_indexing_deferred(here, frontend):
    """
    Test simple vector/matrix arithmetic with a derived type
    with dynamically allocated arrays.
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine array_indexing_deferred(item)
    type(deferred), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.

    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do
  end subroutine array_indexing_deferred
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/(f'derived_types_array_indexing_deferred_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.deferred()
    mod.alloc_deferred(item)
    mod.array_indexing_deferred(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    mod.free_deferred(item)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_indexing_nested(here, frontend):
    """
    Test simple vector/matrix arithmetic with a nested derived type
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit

  type nested
    real(kind=jprb) :: a_scalar, a_vector(3)
    type(explicit) :: another_item
  end type nested
contains

  subroutine array_indexing_nested(item)
    type(nested), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%a_vector(:) = 666.
    item%another_item%vector(:) = 999.

    do i=1, 3
       item%another_item%matrix(:, i) = vals(i)
    end do
  end subroutine array_indexing_nested
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/(f'derived_types_array_indexing_nested_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.nested()
    mod.array_indexing_nested(item)
    assert (item.a_vector == 666.).all()
    assert (item.another_item.vector == 999.).all()
    assert (item.another_item.matrix == np.array([[1., 2., 3.],
                                                  [1., 2., 3.],
                                                  [1., 2., 3.]])).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_deferred_array(here, frontend):
    """
    Test simple vector/matrix with an array of derived types
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine deferred_array(item)
    type(deferred), intent(inout) :: item
    type(deferred), allocatable :: item2(:)
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i, j

    allocate(item2(4))

    do j=1, 4
      call alloc_deferred(item2(j))

      item2(j)%vector(:) = 666.

      do i=1, 3
        item2(j)%matrix(:, i) = vals(i)
      end do
    end do

    item%vector(:) = 0.
    item%matrix(:,:) = 0.

    do j=1, 4
      item%vector(:) = item%vector(:) + item2(j)%vector(:)

      do i=1, 3
          item%matrix(:,i) = item%matrix(:,i) + item2(j)%matrix(:,i)
      end do

      call free_deferred(item2(j))
    end do

    deallocate(item2)
  end subroutine deferred_array
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/(f'derived_types_deferred_array_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.deferred()
    mod.alloc_deferred(item)
    mod.deferred_array(item)
    assert (item.vector == 4 * 666.).all()
    assert (item.matrix == 4 * np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    mod.free_deferred(item)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_caller(here, frontend):
    """
    Test a simple call to another routine specifying a derived type as argument
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit
contains

  subroutine simple_loops(item)
    type(explicit), intent(inout) :: item
    integer :: i, j, n

    n = 3
    do i=1, n
       item%vector(i) = item%vector(i) + item%scalar
    end do

    do j=1, n
       do i=1, n
          item%matrix(i, j) = item%matrix(i, j) + item%scalar
       end do
    end do
  end subroutine simple_loops

  subroutine derived_type_caller(item)
    ! simple call to another routine specifying a derived type as argument
    type(explicit), intent(inout) :: item

    item%red_herring = 42.
    call simple_loops(item)
  end subroutine derived_type_caller

end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/(f'derived_types_derived_type_caller_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    # Test the generated identity
    item = mod.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    item.red_herring = -1.
    mod.derived_type_caller(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all() and item.red_herring == 42.

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_associates(here, frontend):
    """
    Test the use of associate to access and modify other items
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine associates(item)
    type(explicit), intent(inout) :: item
    type(deferred) :: item2

    item%scalar = 17.0

    associate(vector2=>item%matrix(:,1))
        vector2(:) = 3.
        item%matrix(:,3) = vector2(:)
    end associate

    associate(vector=>item%vector)
        item%vector(2) = vector(1)
        vector(3) = item%vector(1) + vector(2)
        vector(1) = 1.
    end associate

    call alloc_deferred(item2)

    associate(vec=>item2%vector(2))
        vec = 1.
    end associate

    call free_deferred(item2)
  end subroutine associates
end module
"""
    # Test the internals
    module = Module.from_source(fcode, frontend=frontend)
    routine = module['associates']
    variables = FindVariables().visit(routine.body)
    if frontend == OMNI:
        assert all(v.shape == (RangeIndex((IntLiteral(1), IntLiteral(3))),)
                   for v in variables if v.name in ['vector', 'vector2'])
    else:
        assert all(v.shape == (IntLiteral(3),)
                   for v in variables if v.name in ['vector', 'vector2'])

    for assoc in FindNodes(Associate).visit(routine.body):
        for var in FindVariables().visit(assoc.body):
            if var.name in assoc.variables:
                assert var.scope is assoc
                assert var.type.parent is None
            else:
                assert var.scope is routine

    # Test the generated module
    filepath = here/(f'derived_types_associates_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    mod.associates(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI fails to read without full module')]))
def test_associates_deferred(frontend):
    """
    Verify that reading in subroutines with deferred external type definitions
    and associates working on that are supported.
    """

    fcode = """
SUBROUTINE ASSOCIATES_DEFERRED(ITEM, IDX)
USE SOME_MOD, ONLY: SOME_TYPE
IMPLICIT NONE
TYPE(SOME_TYPE), INTENT(IN) :: ITEM
INTEGER, INTENT(IN) :: IDX
ASSOCIATE(SOME_VAR=>ITEM%SOME_VAR(IDX))
SOME_VAR = 5
END ASSOCIATE
END SUBROUTINE
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(FindVariables(recurse_to_parent=False).visit(routine.body)) == 3
    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 4
    some_var = variables['SOME_VAR']
    assert isinstance(some_var, DeferredTypeSymbol)
    assert some_var.name.upper() == 'SOME_VAR'
    assert some_var.type.dtype == BasicType.DEFERRED
    assert some_var.scope is FindNodes(Associate).visit(routine.body)[0]


@pytest.mark.parametrize('frontend', available_frontends())
def test_associates_expr(here, frontend):
    """
    Verify that associates with expressions are supported
    """
    fcode = """
subroutine associates_expr(in, out)
  implicit none
  integer, intent(in) :: in(3)
  integer, intent(out) :: out(3)

  out(:) = 0

  associate(a=>1+3)
    out(:) = out(:) + a
  end associate

  associate(b=>2*in(:) + in(:))
    out(:) = out(:) + b(:)
  end associate
end subroutine associates_expr
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 4
    assert isinstance(variables['a'], DeferredTypeSymbol)
    assert variables['a'].type.dtype is BasicType.DEFERRED  # TODO: support type derivation for expressions
    assert isinstance(variables['b'], Array)  # Note: this is an array because we have a shape
    assert variables['b'].type.dtype is BasicType.DEFERRED  # TODO: support type derivation for expressions
    assert variables['b'].type.shape == (IntLiteral(3),)

    filepath = here/(f'associates_expr_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    a = np.array([1, 2, 3], dtype='i')
    b = np.zeros(3, dtype='i')
    function(a, b)
    assert np.all(b == [7, 10, 13])
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_case_sensitivity(here, frontend):
    """
    Some abuse of the case agnostic behaviour of Fortran
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type case_sensitive
    real(kind=jprb) :: u, v, T
    real(kind=jprb) :: q, A
  end type case_sensitive
contains

  subroutine check_case(item)
    type(case_sensitive), intent(inout) :: item

    item%u = 1.0
    item%v = 2.0
    item%t = 3.0
    item%q = -1.0
    item%A = -5.0
  end subroutine check_case
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/(f'derived_types_case_sensitivity_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.case_sensitive()
    item.u = 0.
    item.v = 0.
    item.t = 0.
    item.q = 0.
    item.a = 0.
    mod.check_case(item)
    assert item.u == 1.0 and item.v == 2.0 and item.t == 3.0
    assert item.q == -1.0 and item.a == -5.0

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_procedure_designator(frontend):
    mcode = """
module derived_type_procedure_designator_mod
  implicit none
  type some_type
    integer :: val
  contains
    procedure :: SOME_PROC => some_TYPE_some_proc
    PROCEDURE :: some_FUNC => some_type_SOME_func
  end type some_type

  TYPE other_type
    real :: val
  END TYPE other_type
contains
  subroutine some_type_some_proc(self, val)
    class(some_type) :: self
    integer, intent(in) :: val
    self%val = val
  end subroutine some_type_some_proc

  function some_type_some_func(self)
    integer :: some_type_some_func
    CLASS(SOME_TYPE) :: self
    some_type_some_func = self%val
  end function some_type_some_func
end module derived_type_procedure_designator_mod
    """.strip()

    fcode = """
subroutine derived_type_procedure_designator(val)
  use derived_type_procedure_designator_mod
  implicit none
  integer, intent(out) :: val
  type(some_type) :: tp

  call tp%some_proc(3)
  val = tp%some_func()
end subroutine derived_type_procedure_designator
    """.strip()

    module = Module.from_source(mcode, frontend=frontend)
    assert 'some_type' in module.typedefs
    assert 'other_type' in module.typedefs
    assert 'some_type' in module.symbol_attrs
    assert 'other_type' in module.symbol_attrs

    # First, without external definitions
    if frontend != OMNI:
        routine = Subroutine.from_source(fcode, frontend=frontend)
        assert 'some_type' not in routine.symbol_attrs
        assert 'other_type' not in routine.symbol_attrs
        assert isinstance(routine.symbol_attrs['tp'].dtype, DerivedType)
        assert routine.symbol_attrs['tp'].dtype.typedef == BasicType.DEFERRED

    # Now with external definitions
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=[module])

    for name in ('some_type', 'other_type'):
        assert name in routine.symbol_attrs
        assert routine.symbol_attrs[name].imported is True
        assert isinstance(routine.symbol_attrs[name].dtype, DerivedType)
        assert isinstance(routine.symbol_attrs[name].dtype.typedef, TypeDef)
    assert isinstance(routine.symbol_attrs['tp'].dtype, DerivedType)
    assert isinstance(routine.symbol_attrs['tp'].dtype.typedef, TypeDef)

    # TODO: actually verify representation of type-bound procedures


@pytest.mark.parametrize('frontend', available_frontends())
def test_frontend_strict_mode(frontend, reset_frontend_mode):  # pylint: disable=unused-argument
    """
    Verify that frontends fail on unsupported features if strict mode is enabled
    """
    fcode = """
module frontend_strict_mode
  implicit none
  type some_type
    integer :: val
  end type some_type
  type, extends(some_type) :: other_type
    integer :: foo
  end type other_type
end module frontend_strict_mode
    """
    config['frontend-strict-mode'] = True
    with pytest.raises(NotImplementedError):
        _ = Module.from_source(fcode, frontend=frontend)

    config['frontend-strict-mode'] = False
    module = Module.from_source(fcode, frontend=frontend)
    assert 'some_type' in module.symbol_attrs
    assert 'other_type' in module.symbol_attrs


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_clone(frontend):
    """
    Test cloning of derived types
    """
    fcode = """
module derived_types_clone_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit
end module
"""
    module = Module.from_source(fcode, frontend=frontend)

    explicit = module.typedefs['explicit']
    other = explicit.clone(name='other')

    assert explicit.name == 'explicit'
    assert other.name == 'other'
    assert all(v.scope is other for v in other.variables)
    assert all(v.scope is explicit for v in explicit.variables)

    fcode = fgen(other)
    assert fgen(explicit) == fcode.replace('other', 'explicit')


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_linked_list(frontend):
    """
    Test correct initialization of derived type members that create a circular
    dependency
    """
    fcode = """
module derived_type_linked_list
    implicit none

    type list_t
        integer :: payload
        type(list_t), pointer :: next => null()
    end type list_t

    type(list_t), pointer :: beg => null()
    type(list_t), pointer :: cur => null()

contains

    subroutine find(val, this)
        integer, intent(in) :: val
        type(list_t), pointer, intent(inout) :: this
        type(list_t), pointer :: x
        this => null()
        x => beg
        do while (associated(x))
            if (x%payload == val) then
                this => x
                return
            endif
            x => x%next
        end do
    end subroutine find
end module derived_type_linked_list
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    # Test correct instantiation and association of module-level variables
    for name in ('beg', 'cur'):
        assert name in module.variables
        assert isinstance(module.variable_map[name].type.dtype, DerivedType)
        assert module.variable_map[name].type.dtype.typedef is module.typedefs['list_t']

        variables = module.variable_map[name].type.dtype.typedef.variables
        assert all(v.scope is module.variable_map[name].type.dtype.typedef for v in variables)
        assert 'payload' in variables
        assert 'next' in variables

        variables = module.variable_map[name].variables
        assert all(v.scope is module for v in variables)
        assert f'{name}%payload' in variables
        assert f'{name}%next' in variables

    # Test correct instantiation and association of subroutine-level variables
    routine = module['find']
    for name in ('this', 'x'):
        var = routine.variable_map[name]
        assert var.type.dtype.typedef is module.typedefs['list_t']

        assert 'payload' in var.variable_map
        assert 'next' in var.variable_map
        assert all(v.scope is var.scope for v in var.variables)

    # Test on-the-fly creation of variable lists
    var = routine.variable_map['x']
    name = 'x'
    for _ in range(1000):  # Let's chase the next-chain 1000x
        var = var.variable_map['next']
        assert var
        assert var.type.dtype.typedef is module.typedefs['list_t']
        name = f'{name}%next'
        assert var.name == name
