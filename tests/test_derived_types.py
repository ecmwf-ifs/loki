from pathlib import Path
import re
import pytest
import numpy as np

from conftest import jit_compile, clean_test, available_frontends
from loki import (
    OMNI, Module, Subroutine, BasicType, DerivedType, TypeDef,
    fgen, FindNodes, Intrinsic, ProcedureDeclaration, ProcedureType,
    VariableDeclaration, Assignment
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


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
def test_derived_type_bind_c(frontend):
    # Example code from F2008, Note 15.13
    fcode = """
module derived_type_bind_c
    ! typedef struct {
    ! int m, n;
    ! float r;
    ! } myctype;

    USE, INTRINSIC :: ISO_C_BINDING
    TYPE, BIND(C) :: MYFTYPE
      INTEGER(C_INT) :: I, J
      REAL(C_FLOAT) :: S
    END TYPE MYFTYPE
end module derived_type_bind_c
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
    myftype = module.typedefs['myftype']
    assert myftype.bind_c is True
    assert ', BIND(C)' in fgen(myftype)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_inheritance(frontend):
    fcode = """
module derived_type_private_mod
    implicit none

    type, abstract :: base_type
        integer :: val
    end type base_type

    type, extends(base_type) :: some_type
        integer :: other_val
    end type some_type

contains

    function base_proc(self) result(result)
        class(base_type) :: self
        integer :: result
        result = self%val
    end function base_proc

    function some_proc(self) result(result)
        class(some_type) :: self
        integer :: result
        result = self%val + self%other_val
    end function some_proc
end module derived_type_private_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    base_type = module.typedefs['base_type']
    some_type = module.typedefs['some_type']

    # Verify correct properties on the `TypeDef` object
    assert base_type.abstract is True
    assert some_type.abstract is False

    assert base_type.extends is None
    assert some_type.extends.lower() == 'base_type'

    assert base_type.bind_c is False
    assert some_type.bind_c is False

    # Verify fgen
    assert 'type, abstract' in fgen(base_type).lower()
    assert 'extends(base_type)' in fgen(some_type).lower()


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_private(frontend):
    fcode = """
module derived_type_private_mod
    implicit none
    public
    TYPE, private :: PRIV_TYPE
      INTEGER :: I, J
    END TYPE PRIV_TYPE
end module derived_type_private_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    priv_type = module.typedefs['priv_type']
    assert priv_type.private is True
    assert priv_type.public is False
    assert ', PRIVATE' in fgen(priv_type)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_public(frontend):
    fcode = """
module derived_type_public_mod
    implicit none
    private
    TYPE, public :: PUB_TYPE
      INTEGER :: I, J
    END TYPE PUB_TYPE
end module derived_type_public_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    pub_type = module.typedefs['pub_type']
    assert pub_type.public is True
    assert pub_type.private is False
    assert ', PUBLIC' in fgen(pub_type)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_private_comp(frontend):
    fcode = """
module derived_type_private_comp_mod
    implicit none

    type, abstract :: base_type
        integer :: val
    end type base_type

    type, extends(base_type) :: some_private_comp_type
        private
        integer :: other_val
    contains
        procedure :: proc => other_proc
    end type some_private_comp_type

    type, extends(base_type) :: type_bound_proc_type
        integer :: other_val
    contains
        private
        procedure :: proc => other_proc
    end type type_bound_proc_type

contains

    function other_proc(self) result(result)
        class(type_bound_proc_type) :: self
        integer :: result
        result = self%val
    end function proc

end module derived_type_private_comp_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    some_private_comp_type = module.typedefs['some_private_comp_type']
    type_bound_proc_type = module.typedefs['type_bound_proc_type']

    intrinsic_nodes = FindNodes(Intrinsic).visit(type_bound_proc_type.body)
    assert len(intrinsic_nodes) == 2
    assert intrinsic_nodes[0].text.lower() == 'contains'
    assert intrinsic_nodes[1].text.lower() == 'private'

    assert re.search(
      r'^\s+contains$\s+private', fgen(type_bound_proc_type), re.I | re.MULTILINE
    ) is not None

    # OMNI gets the below wrong as it doesn't retain the private statement for components
    if frontend != OMNI:
        intrinsic_nodes = FindNodes(Intrinsic).visit(some_private_comp_type.body)
        assert len(intrinsic_nodes) == 2
        assert intrinsic_nodes[0].text.lower() == 'private'
        assert intrinsic_nodes[1].text.lower() == 'contains'

        assert re.search(
            r'^\s+private*$(\s.*?){2}\s+contains', fgen(some_private_comp_type), re.I | re.MULTILINE
        ) is not None


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_procedure_designator(frontend):
    mcode = """
module derived_type_procedure_designator_mod
  implicit none
  type :: some_type
    integer :: val
  contains
    procedure :: SOME_PROC => some_TYPE_some_proc
    PROCEDURE :: some_FUNC => SOME_TYPE_SOME_FUNC
    PROCEDURE :: OTHER_PROC
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

  subroutine other_proc(self)
    class(some_type) :: self
    self%val = self%val + 1
  end subroutine other_proc
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

    # First, with external definitions (generates xmod for OMNI)
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=[module])

    for name in ('some_type', 'other_type'):
        assert name in routine.symbol_attrs
        assert routine.symbol_attrs[name].imported is True
        assert isinstance(routine.symbol_attrs[name].dtype, DerivedType)
        assert isinstance(routine.symbol_attrs[name].dtype.typedef, TypeDef)

    # Make sure type-bound procedure declarations exist
    some_type = module.typedefs['some_type']
    proc_decls = FindNodes(ProcedureDeclaration).visit(some_type.body)
    assert len(proc_decls) == 3
    assert all(decl.interface is None for decl in proc_decls)

    proc_symbols = {s.name.lower(): s for d in proc_decls for s in d.symbols}
    assert set(proc_symbols.keys()) == {'some_proc', 'some_func', 'other_proc'}
    assert all(s.scope is some_type for s in proc_symbols.values())
    assert all(isinstance(s.type.dtype, ProcedureType) for s in proc_symbols.values())

    assert proc_symbols['some_proc'].type.bind_names == ('some_type_some_proc',)
    assert proc_symbols['some_proc'].type.bind_names[0].scope is module
    assert proc_symbols['some_func'].type.bind_names == ('some_type_some_func',)
    assert proc_symbols['some_proc'].type.bind_names[0].scope is module
    assert proc_symbols['other_proc'].type.bind_names is None
    assert all(proc.type.initial is None for proc in proc_symbols.values())

    # Verify type representation in bound routines
    some_type_some_proc = module['some_type_some_proc']
    self = some_type_some_proc.symbol_map['self']
    assert isinstance(self.type.dtype, DerivedType)
    assert self.type.dtype.typedef is some_type
    assert self.type.polymorphic is True
    decls = FindNodes(VariableDeclaration).visit(some_type_some_proc.spec)
    assert 'CLASS(SOME_TYPE)' in fgen(decls[0]).upper()

    # Verify type representation in using routine
    assert isinstance(routine.symbol_attrs['tp'].dtype, DerivedType)
    assert isinstance(routine.symbol_attrs['tp'].dtype.typedef, TypeDef)
    assert routine.symbol_attrs['tp'].polymorphic is None
    assert routine.symbol_attrs['tp'].dtype.typedef is some_type
    decls = FindNodes(VariableDeclaration).visit(routine.spec)
    assert 'TYPE(SOME_TYPE)' in fgen(decls[1]).upper()

    # TODO: verify correct type association of calls to type-bound procedures

    # Next, without external definitions
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert 'some_type' not in routine.symbol_attrs
    assert 'other_type' not in routine.symbol_attrs
    assert isinstance(routine.symbol_attrs['tp'].dtype, DerivedType)
    assert routine.symbol_attrs['tp'].dtype.typedef == BasicType.DEFERRED

    # TODO: verify correct type association of calls to type-bound procedures


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_bind_attrs(frontend):
    """
    Test attribute representation in type-bound procedures
    """
    fcode = """
module derived_types_bind_attrs_mod
    implicit none

    type some_type
        integer :: val
    contains
        PROCEDURE, PASS, NON_OVERRIDABLE :: pass_proc
        PROCEDURE, NOPASS, PUBLIC :: no_pass_proc
        PROCEDURE, PASS(this), private :: pass_arg_proc
    end type some_type

contains

    subroutine pass_proc(self)
        class(some_type) :: self
    end subroutine pass_proc

    subroutine no_pass_proc(val)
        integer, intent(inout) :: val
    end subroutine no_pass_proc

    subroutine pass_arg_proc(val, this)
        integer, intent(inout) :: val
        class(some_type) :: this
    end subroutine pass_arg_proc

end module derived_types_bind_attrs_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    some_type = module.typedefs['some_type']

    proc_decls = FindNodes(ProcedureDeclaration).visit(some_type.body)
    assert len(proc_decls) == 3
    assert all(decl.interface is None for decl in proc_decls)

    proc_symbols = {s.name.lower(): s for d in proc_decls for s in d.symbols}
    assert set(proc_symbols.keys()) == {'pass_proc', 'no_pass_proc', 'pass_arg_proc'}

    assert proc_symbols['pass_proc'].type.pass_attr is True
    assert proc_symbols['pass_proc'].type.non_overridable is True
    assert proc_symbols['pass_proc'].type.private is None
    assert proc_symbols['pass_proc'].type.public is None

    assert proc_symbols['no_pass_proc'].type.pass_attr is False
    assert proc_symbols['no_pass_proc'].type.non_overridable is None
    assert proc_symbols['no_pass_proc'].type.private is None
    assert proc_symbols['no_pass_proc'].type.public is True

    assert proc_symbols['pass_arg_proc'].type.pass_attr == 'this'
    assert proc_symbols['pass_arg_proc'].type.private is True
    assert proc_symbols['pass_arg_proc'].type.public is None

    proc_decls = {decl.symbols[0].name: decl for decl in proc_decls}
    assert ', PASS' in fgen(proc_decls['pass_proc'])
    assert ', NON_OVERRIDABLE' in fgen(proc_decls['pass_proc'])

    assert ', NOPASS' in fgen(proc_decls['no_pass_proc'])
    assert ', PUBLIC' in fgen(proc_decls['no_pass_proc'])

    assert ', PASS(this)' in fgen(proc_decls['pass_arg_proc'])
    assert ', PRIVATE' in fgen(proc_decls['pass_arg_proc'])


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_bind_deferred(frontend):
    # Example from https://www.ibm.com/docs/en/xffbg/121.141?topic=types-abstract-deferred-bindings-fortran-2003
    fcode = """
module derived_type_bind_deferred_mod
implicit none
TYPE, ABSTRACT :: FILE_HANDLE
   CONTAINS
   PROCEDURE(OPEN_FILE), DEFERRED, PASS(HANDLE) :: OPEN
END TYPE

INTERFACE
    SUBROUTINE OPEN_FILE(HANDLE)
        IMPORT FILE_HANDLE
        CLASS(FILE_HANDLE), INTENT(IN):: HANDLE
    END SUBROUTINE OPEN_FILE
END INTERFACE
end module derived_type_bind_deferred_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)

    file_handle = module.typedefs['file_handle']
    assert len(file_handle.body) == 2

    proc_decl = file_handle.body[1]
    assert proc_decl.interface == 'open_file'

    proc_sym = proc_decl.symbols[0]
    assert proc_sym.type.deferred is True
    assert proc_sym.type.pass_attr.lower() == 'handle'

    assert ', DEFERRED' in fgen(proc_decl)
    assert ', PASS(HANDLE)' in fgen(proc_decl).upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_final_generic(frontend):
    """
    Test derived types with generic and final bindings
    """
    fcode = """
module derived_type_final_generic_mod
    implicit none

    type hdf5_file
        logical :: is_open = .false.
        integer :: file_id
    contains
        procedure :: open_file => hdf5_file_open
        procedure, private :: hdf5_file_load_int
        procedure, private :: hdf5_file_load_real
        generic, public :: load => hdf5_file_load_int, hdf5_file_load_real
        final :: hdf5_file_close
    end type hdf5_file

contains

    subroutine hdf5_file_open (self, filepath)
        class(hdf5_file) :: self
        character(len=*), intent(in) :: filepath
        self%file_id = LEN(filepath)  ! dummy operation
        self%is_open = .true.
    end subroutine hdf5_file_open

    subroutine hdf5_file_load_int (self, val)
        class(hdf5_file) :: self
        integer, intent(out) :: val
        val = 0
        if (self%is_open) then
            val = self%file_id  ! dummy operation
        end if
    end subroutine hdf5_file_load_int

    subroutine hdf5_file_load_real (self, val)
        class(hdf5_file) :: self
        real, intent(out) :: val
        val = 0.
        if (self%is_open) then
            val = real(self%file_id)  ! dummy operation
        end if
    end subroutine hdf5_file_load_real

    subroutine hdf5_file_close (self)
        type(hdf5_file) :: self
        if (self%is_open) then
            self%file_id = 0
            self%is_open = .false.
        end if
    end subroutine hdf5_file_close
end module derived_type_final_generic_mod
    """.strip()

    mod = Module.from_source(fcode, frontend=frontend)
    hdf5_file = mod.typedefs['hdf5_file']
    proc_decls = FindNodes(ProcedureDeclaration).visit(hdf5_file.body)
    assert len(proc_decls) == 5

    assert all(decl.final is False for decl in proc_decls[:-1])
    assert all(decl.generic is False for decl in proc_decls[:-2])

    proc_map = {proc.name.lower(): proc for decl in proc_decls for proc in decl.symbols}

    assert proc_decls[-2].generic is True
    assert 'generic, public ::' in fgen(proc_decls[-2]).lower()
    assert 'load => ' in fgen(proc_decls[-2]).lower()
    assert proc_decls[-2].symbols == ('load',)
    assert proc_decls[-2].symbols[0].type.bind_names == ('hdf5_file_load_int', 'hdf5_file_load_real')
    assert proc_decls[-2].symbols[0].type.dtype.name == 'load'
    assert proc_decls[-2].symbols[0].type.dtype.is_generic is True
    assert all(proc.type.dtype.name == proc.name for proc in proc_decls[-2].symbols[0].type.bind_names)
    assert all(proc == proc_map[proc.name] for proc in proc_decls[-2].symbols[0].type.bind_names)

    assert proc_decls[-1].final is True
    assert proc_decls[-1].generic is False
    assert 'final ::' in fgen(proc_decls[-1]).lower()
    assert proc_decls[-1].symbols == ('hdf5_file_close',)
    assert proc_decls[-1].symbols[0].type.dtype.name == 'hdf5_file_close'


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


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_nested_procedure_call(frontend):
    """
    Test correct representation of inline calls and call statements for
    type-bound procedures in nested derived types.
    """
    fcode = """
module derived_type_nested_proc_call_mod
    implicit none

    type netcdf_file_raw
        private
    contains
        procedure, public :: exists => raw_exists
    end type

    type netcdf_file
        type(netcdf_file_raw) :: file
    contains
        procedure :: exists
    end type netcdf_file

contains

    function exists(this, var_name) result(is_present)
        class(netcdf_file)           :: this
        character(len=*), intent(in) :: var_name
        logical :: is_present

        is_present = this%file%exists(var_name)
    end function exists

    function raw_exists(this, var_name) result(is_present)
        class(netcdf_file_raw)      :: this
        character(len=*), intent(in) :: var_name
        logical :: is_present

        is_present = .true.
    end function raw_exists

end module derived_type_nested_proc_call_mod
    """.strip()

    mod = Module.from_source(fcode, frontend=frontend)

    assignment = FindNodes(Assignment).visit(mod['exists'].body)
    assert len(assignment) == 1
    assert fgen(assignment[0].rhs).lower() == 'this%file%exists(var_name)'

    # TODO: Verify type of function symbol etc


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_sequence(frontend):
    """
    Verify derived types with ``SEQUENCE`` stmt work as expected
    """
    # F2008, Note 4.18
    fcode = """
module derived_type_sequence
    implicit none
    TYPE NUMERIC_SEQ
        SEQUENCE
        INTEGER :: INT_VAL
        REAL :: REAL_VAL
        LOGICAL :: LOG_VAL
    END TYPE NUMERIC_SEQ
end module derived_type_sequence
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
    numeric_seq = module.typedefs['numeric_seq']
    assert 'SEQUENCE' in fgen(numeric_seq)
