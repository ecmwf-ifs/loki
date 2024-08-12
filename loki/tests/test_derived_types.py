# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines
from sys import getrecursionlimit
from inspect import stack

import re
import pytest
import numpy as np

from loki import (
    Module, Subroutine, BasicType, DerivedType, TypeDef, fgen,
    FindNodes, Intrinsic, ProcedureDeclaration, ProcedureType,
    VariableDeclaration, Assignment, InlineCall, Builder,
    StringSubscript, Conditional, CallStatement, ProcedureSymbol,
    FindVariables
)
from loki.build import jit_compile, jit_compile_lib, clean_test, Obj
from loki.frontend import available_frontends, OMNI, OFP


@pytest.fixture(name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_simple_loops(tmp_path, frontend):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['simple_loops']

    # Ensure type info is attached correctly
    item_vars = [v for v in FindVariables(unique=False).visit(routine.body) if v.parent]
    assert all(v.type.dtype == BasicType.REAL for v in item_vars)
    assert item_vars[0].name == 'item%vector' and item_vars[0].shape == (3,)
    assert item_vars[1].name == 'item%vector' and item_vars[1].shape == (3,)
    assert item_vars[2].name == 'item%scalar' and item_vars[2].type.shape is None
    assert item_vars[3].name == 'item%matrix' and item_vars[3].shape == (3, 3)
    assert item_vars[4].name == 'item%matrix' and item_vars[4].shape == (3, 3)
    assert item_vars[5].name == 'item%scalar' and item_vars[5].type.shape is None

    filepath = tmp_path/(f'derived_types_simple_loops_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    mod.simple_loops(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_indexing_explicit(tmp_path, frontend):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    filepath = tmp_path/(f'derived_types_array_indexing_explicit_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    mod.array_indexing_explicit(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_indexing_deferred(tmp_path, frontend):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    filepath = tmp_path/(f'derived_types_array_indexing_deferred_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.deferred()
    mod.alloc_deferred(item)
    mod.array_indexing_deferred(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    mod.free_deferred(item)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_array_indexing_nested(tmp_path, frontend):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    filepath = tmp_path/(f'derived_types_array_indexing_nested_{frontend}.f90')
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
def test_deferred_array(tmp_path, frontend):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    filepath = tmp_path/(f'derived_types_deferred_array_{frontend}.f90')
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.deferred()
    mod.alloc_deferred(item)
    mod.deferred_array(item)
    assert (item.vector == 4 * 666.).all()
    assert (item.matrix == 4 * np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    mod.free_deferred(item)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_caller(tmp_path, frontend):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    filepath = tmp_path/(f'derived_types_derived_type_caller_{frontend}.f90')
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
def test_case_sensitivity(tmp_path, frontend):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    filepath = tmp_path/(f'derived_types_case_sensitivity_{frontend}.f90')
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
def test_derived_type_bind_c(frontend, tmp_path):
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

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    myftype = module.typedef_map['myftype']
    assert myftype.bind_c is True
    assert ', BIND(C)' in fgen(myftype)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_inheritance(frontend, tmp_path):
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

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    base_type = module.typedef_map['base_type']
    some_type = module.typedef_map['some_type']

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
def test_derived_type_private(frontend, tmp_path):
    fcode = """
module derived_type_private_mod
    implicit none
    public
    TYPE, private :: PRIV_TYPE
      INTEGER :: I, J
    END TYPE PRIV_TYPE
end module derived_type_private_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    priv_type = module.typedef_map['priv_type']
    assert priv_type.private is True
    assert priv_type.public is False
    assert ', PRIVATE' in fgen(priv_type)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_public(frontend, tmp_path):
    fcode = """
module derived_type_public_mod
    implicit none
    private
    TYPE, public :: PUB_TYPE
      INTEGER :: I, J
    END TYPE PUB_TYPE
end module derived_type_public_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    pub_type = module.typedef_map['pub_type']
    assert pub_type.public is True
    assert pub_type.private is False
    assert ', PUBLIC' in fgen(pub_type)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_private_comp(frontend, tmp_path):
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
    end function other_proc

end module derived_type_private_comp_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    some_private_comp_type = module.typedef_map['some_private_comp_type']
    type_bound_proc_type = module.typedef_map['type_bound_proc_type']

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
def test_derived_type_procedure_designator(frontend, tmp_path):
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

    module = Module.from_source(mcode, frontend=frontend, xmods=[tmp_path])
    assert 'some_type' in module.typedef_map
    assert 'other_type' in module.typedef_map
    assert 'some_type' in module.symbol_attrs
    assert 'other_type' in module.symbol_attrs

    # First, with external definitions (generates xmod for OMNI)
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=[module], xmods=[tmp_path])

    for name in ('some_type', 'other_type'):
        assert name in routine.symbol_attrs
        assert routine.symbol_attrs[name].imported is True
        assert isinstance(routine.symbol_attrs[name].dtype, DerivedType)
        assert isinstance(routine.symbol_attrs[name].dtype.typedef, TypeDef)

    # Make sure type-bound procedure declarations exist
    some_type = module.typedef_map['some_type']
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
    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    assert 'some_type' not in routine.symbol_attrs
    assert 'other_type' not in routine.symbol_attrs
    assert isinstance(routine.symbol_attrs['tp'].dtype, DerivedType)
    assert routine.symbol_attrs['tp'].dtype.typedef == BasicType.DEFERRED

    # TODO: verify correct type association of calls to type-bound procedures


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_bind_attrs(frontend, tmp_path):
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

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    some_type = module.typedef_map['some_type']

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
def test_derived_type_bind_deferred(frontend, tmp_path):
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

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    file_handle = module.typedef_map['file_handle']
    assert len(file_handle.body) == 2

    proc_decl = file_handle.body[1]
    assert proc_decl.interface == 'open_file'

    proc_sym = proc_decl.symbols[0]
    assert proc_sym.type.deferred is True
    assert proc_sym.type.pass_attr.lower() == 'handle'

    assert ', DEFERRED' in fgen(proc_decl)
    assert ', PASS(HANDLE)' in fgen(proc_decl).upper()


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_final_generic(frontend, tmp_path):
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

    mod = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    hdf5_file = mod.typedef_map['hdf5_file']
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
def test_derived_type_clone(frontend, tmp_path):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    explicit = module.typedef_map['explicit']
    other = explicit.clone(name='other')

    assert explicit.name == 'explicit'
    assert other.name == 'other'
    assert all(v.scope is other for v in other.variables)
    assert all(v.scope is explicit for v in explicit.variables)

    fcode = fgen(other)
    assert fgen(explicit) == fcode.replace('other', 'explicit')


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_linked_list(frontend, tmp_path):
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

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Test correct instantiation and association of module-level variables
    for name in ('beg', 'cur'):
        assert name in module.variables
        assert isinstance(module.variable_map[name].type.dtype, DerivedType)
        assert module.variable_map[name].type.dtype.typedef is module.typedef_map['list_t']

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
        assert var.type.dtype.typedef is module.typedef_map['list_t']

        assert 'payload' in var.variable_map
        assert 'next' in var.variable_map
        assert all(v.scope is var.scope for v in var.variables)

    # Test on-the-fly creation of variable lists
    # Chase the next-chain to the limit with a buffer
    var = routine.variable_map['x']
    name = 'x'
    for _ in range(min(1000, getrecursionlimit()-len(stack())-50)):
        var = var.variable_map['next']
        assert var
        assert var.type.dtype.typedef is module.typedef_map['list_t']
        name = f'{name}%next'
        assert var.name == name


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_nested_procedure_call(frontend, tmp_path):
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

    mod = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    assignment = FindNodes(Assignment).visit(mod['exists'].body)
    assert len(assignment) == 1
    assignment = assignment[0]
    assert isinstance(assignment.rhs, InlineCall)
    assert fgen(assignment.rhs).lower() == 'this%file%exists(var_name)'

    assert isinstance(assignment.rhs.function, ProcedureSymbol)
    assert isinstance(assignment.rhs.function.type.dtype, ProcedureType)
    assert assignment.rhs.function.parent and isinstance(assignment.rhs.function.parent.type.dtype, DerivedType)
    assert assignment.rhs.function.parent.type.dtype.name == 'netcdf_file_raw'
    assert assignment.rhs.function.parent.type.dtype.typedef is mod['netcdf_file_raw']
    assert assignment.rhs.function.parent.parent
    assert isinstance(assignment.rhs.function.parent.parent.type.dtype, DerivedType)
    assert assignment.rhs.function.parent.parent.type.dtype.name == 'netcdf_file'
    assert assignment.rhs.function.parent.parent.type.dtype.typedef is mod['netcdf_file']


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_sequence(frontend, tmp_path):
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

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    numeric_seq = module.typedef_map['numeric_seq']
    assert 'SEQUENCE' in fgen(numeric_seq)


@pytest.fixture(name='shadowed_typedef_symbols_fcode')
def fixture_shadowed_typedef_symbols_fcode(tmp_path, builder):
    # Use a bespoke module name to avoid name clashes
    module_name = f'rad_rand_numb_{tmp_path.name[-4:]}'

    # Excerpt from ecrad's radiation_random_numbers.F90
    fcode = f"""
module {module_name}

  implicit none

  public :: rng_type, IRngNative

  enum, bind(c)
    enumerator IRngNative      ! Built-in Fortran-90 RNG
  end enum

  integer, parameter            :: jpim = selected_int_kind(9)
  integer, parameter            :: jprb = selected_real_kind(13,300)
  integer(kind=jpim), parameter :: NMaxStreams = 512

  type rng_type

    integer(kind=jpim) :: itype = IRngNative
    real(kind=jprb)    :: istate(NMaxStreams)
    integer(kind=jpim) :: nmaxstreams = NMaxStreams
    integer(kind=jpim) :: iseed = 0

  end type rng_type

contains

  subroutine rng_default(istate_dim, maxstreams)
    integer, intent(out) :: istate_dim, maxstreams
    type(rng_type) :: rng
    integer :: dim(1)
    rng = rng_type(istate=0._jprb)
    dim = shape(rng%istate)
    istate_dim = dim(1)
    maxstreams = rng%nmaxstreams
  end subroutine rng_default

  subroutine rng_init(istate_dim, maxstreams)
    integer, intent(out) :: istate_dim, maxstreams
    type(rng_type) :: rng
    integer :: dim(1)
    rng = rng_type(nmaxstreams=256, istate=0._jprb)
    dim = shape(rng%istate)
    istate_dim = dim(1)
    maxstreams = rng%nmaxstreams
  end subroutine rng_init

end module {module_name}
    """.strip()

    # Verify that this code behaves as expected
    ref_path = tmp_path/'radiation_random_numbers.F90'
    ref_path.write_text(fcode)

    ref_lib = jit_compile_lib([ref_path], path=tmp_path, name=module_name, builder=builder)
    ref_mod = getattr(ref_lib, module_name)
    ref_default_shape, ref_default_maxstreams = ref_mod.rng_default()
    ref_init_shape, ref_init_maxstreams = ref_mod.rng_init()

    assert ref_default_shape == 512
    assert ref_default_maxstreams == 512
    assert ref_init_shape == 512
    assert ref_init_maxstreams == 256

    yield fcode

    clean_test(ref_path)


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_rescope_symbols_shadowed(tmp_path, shadowed_typedef_symbols_fcode, frontend):
    """
    Test the rescoping of symbols with shadowed symbols in a typedef.
    """
    # Parse into Loki IR
    module = Module.from_source(shadowed_typedef_symbols_fcode, frontend=frontend, xmods=[tmp_path])
    mod_var = module.variable_map['nmaxstreams']
    assert mod_var.scope is module

    # Verify scope of variables in type def
    rng_type = module.typedef_map['rng_type']
    istate = rng_type.variable_map['istate']
    tdef_var = rng_type.variable_map['nmaxstreams']

    assert istate in ('istate(nmaxstreams)', 'istate(1:nmaxstreams)')
    assert istate.scope is rng_type

    if frontend == OMNI:
        assert istate.dimensions[0] == '1:nmaxstreams'
        assert istate.dimensions[0].stop.scope
    else:
        assert istate.dimensions[0] == 'nmaxstreams'
        assert istate.dimensions[0].scope

    # FIXME: Use of NMaxStreams from parent scope is in the wrong scope (LOKI-52)
    #assert istate.dimensions[0].scope is module

    assert tdef_var.scope is rng_type

    if frontend != OMNI:
        # FIXME: OMNI doesn't retain the initializer expressions in the typedef
        from loki.expression import Scalar  # pylint: disable=import-outside-toplevel
        assert tdef_var.type.initial == 'NMaxStreams'
        assert tdef_var.type.initial.scope is module
        assert tdef_var.type.initial == mod_var
        assert isinstance(tdef_var.type.initial, Scalar)

        # Test the outcome works as expected
        filepath = tmp_path/f'{module.name}_{frontend}.F90'
        mod = jit_compile(module, filepath=filepath, objname=module.name)

        default_shape, default_maxstreams = mod.rng_default()
        init_shape, init_maxstreams = mod.rng_init()

        assert default_shape == 512
        assert default_maxstreams == 512
        assert init_shape == 512
        assert init_maxstreams == 256

        clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[
    (OFP, 'OFP cannot parse the Fortran')
]))
def test_derived_types_character_array_subscript(frontend, tmp_path):
    fcode = """
module derived_type_char_arr_mod
    implicit none

    type char_arr_type
        character(len=511) :: some_name(3) = ["","",""]
    end type char_arr_type

contains

    subroutine some_routine(config)
        type(char_arr_type), intent(in) :: config
        integer :: i, strlen
        do i=1,3
            if (config%some_name(i)(1:1) == '/') then
                print *, 'absolute path'
            end if
            strlen = len_trim(config%some_name(i))
            if (config%some_name(i)(strlen-2:strlen) == '.nc') then
                print *, 'netcdf file'
            end if
        end do
    end subroutine some_routine
end module derived_type_char_arr_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    conditionals = FindNodes(Conditional).visit(module['some_routine'].body)
    assert all(isinstance(c.condition.left, StringSubscript) for c in conditionals)
    assert [fgen(c.condition.left) for c in conditionals] == [
      'config%some_name(i)(1:1)', 'config%some_name(i)(strlen - 2:strlen)'
    ]


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_types_nested_subscript(frontend, tmp_path):
    fcode = """
module derived_types_nested_subscript
    implicit none

    type inner_type
        integer :: val
    contains
        procedure :: some_routine
    end type inner_type

    type outer_type
        type(inner_type) :: inner(3)
    end type outer_type

contains

    subroutine some_routine(this, val)
        class(inner_type), intent(inout) :: this
        integer, intent(in) :: val
        this%val = val
    end subroutine some_routine

    subroutine driver(outers)
        type(outer_type), intent(inout) :: outers(5)
        integer :: i, j

        do i=1,5
            do j=1,3
                call outers(i)%inner(j)%some_routine(i*10 + j)
            end do
        end do
    end subroutine driver

end module derived_types_nested_subscript
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    calls = FindNodes(CallStatement).visit(module['driver'].body)
    assert len(calls) == 1
    assert str(calls[0].name) == 'outers(i)%inner(j)%some_routine'
    assert fgen(calls[0].name) == 'outers(i)%inner(j)%some_routine'


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_types_nested_type(frontend, tmp_path):
    fcode_module = """
module some_mod
    implicit none

    type some_type
        integer :: val
    contains
        procedure :: some_routine
    end type some_type

    type other_type
        type(some_type) :: data
    contains
        procedure :: other_routine
    end type other_type

contains

    subroutine some_routine(this)
        class(some_type), intent(inout) :: this
        this%val = 5
    end subroutine some_routine

    subroutine other_routine(this)
        class(other_type), intent(inout) :: this
        call this%data%some_routine
    end subroutine other_routine
end module some_mod
    """.strip()

    fcode_driver = """
subroutine driver
    use some_mod, only: other_type
    implicit none
    type(other_type) :: var
    integer :: val
    call var%other_routine
    call var%data%some_routine
    val = var%data%val
end subroutine driver
    """.strip()

    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=[module], xmods=[tmp_path])

    other_routine = module['other_routine']
    call = other_routine.body.body[0]
    assert isinstance(call, CallStatement)
    assert isinstance(call.name.type.dtype, ProcedureType)
    assert call.name.parent and isinstance(call.name.parent.type.dtype, DerivedType)
    assert call.name.parent.type.dtype.name == 'some_type'
    assert call.name.parent.type.dtype.typedef is module['some_type']
    assert call.name.parent.parent and isinstance(call.name.parent.parent.type.dtype, DerivedType)
    assert call.name.parent.parent.type.dtype.name == 'other_type'
    assert call.name.parent.parent.type.dtype.typedef is module['other_type']

    calls = FindNodes(CallStatement).visit(driver.body)
    assert len(calls) == 2
    for call in calls:
        assert isinstance(call.name.type.dtype, ProcedureType)
        assert call.name.parent and isinstance(call.name.parent.type.dtype, DerivedType)

    assert calls[0].name.parent.type.dtype.name == 'other_type'
    assert calls[0].name.parent.type.dtype.typedef is module['other_type']

    assert calls[1].name.parent.type.dtype.name == 'some_type'
    assert calls[1].name.parent.type.dtype.typedef is module['some_type']
    assert calls[1].name.parent.parent
    assert calls[1].name.parent.parent.type.dtype.name == 'other_type'
    assert calls[1].name.parent.parent.type.dtype.typedef is module['other_type']

    assignment = driver.body.body[-1]
    assert isinstance(assignment, Assignment)
    assert assignment.rhs.type.dtype is BasicType.INTEGER
    assert assignment.rhs.parent and isinstance(assignment.rhs.parent.type.dtype, DerivedType)
    assert assignment.rhs.parent.type.dtype.name == 'some_type'
    assert assignment.rhs.parent.type.dtype.typedef is module['some_type']
    assert assignment.rhs.parent.parent.type.dtype.name == 'other_type'
    assert assignment.rhs.parent.parent.type.dtype.typedef is module['other_type']


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_types_abstract_deferred_procedure(frontend, tmp_path):
    fcode = """
module some_mod
    implicit none
    type, abstract :: abstract_type
        contains
        procedure (some_proc), deferred :: some_proc
        procedure (other_proc), deferred :: other_proc
    end type abstract_type

    abstract interface
        subroutine some_proc(this)
            import abstract_type
            class(abstract_type), intent(in) :: this
        end subroutine some_proc
    end interface

    abstract interface
        subroutine other_proc(this)
            import abstract_type
            class(abstract_type), intent(inout) :: this
        end subroutine other_proc
    end interface
end module some_mod
    """.strip()
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    typedef = module['abstract_type']
    assert typedef.abstract is True
    assert typedef.variables == ('some_proc', 'other_proc')
    for symbol in typedef.variables:
        assert isinstance(symbol, ProcedureSymbol)
        assert isinstance(symbol.type.dtype, ProcedureType)
        assert symbol.type.dtype.name.lower() == symbol.name.lower()
        assert symbol.type.bind_names == (symbol,)
        assert symbol.scope is typedef
        assert symbol.type.bind_names[0].scope is module

    assert typedef.imported_symbols == ()
    assert not typedef.imported_symbol_map


@pytest.mark.parametrize('frontend', available_frontends())
def test_derived_type_symbol_inheritance(frontend, tmp_path):
    fcode = """
module some_mod
implicit none
type :: base_type
    integer :: memberA
    real :: memberB
    contains
    procedure :: init => init_base_type
    procedure :: final => final_base_type
    procedure :: copy
end type base_type

type, extends(base_type) :: extended_type
    integer :: memberC
    contains
    procedure :: init => init_extended_type
    procedure :: final => final_extended_type
    procedure :: do_something
end type extended_type

type, extends(extended_type) :: extended_extended_type
    integer :: memberD
    contains
    procedure :: init => init_extended_extended_type
    procedure :: final => final_extended_extended_type
    procedure :: do_something => do_something_else
end type extended_extended_type

contains

subroutine init_base_type(self)
  class(base_type) :: self
end subroutine init_base_type
subroutine final_base_type(self)
  class(base_type) :: self
end subroutine final_base_type
subroutine copy(self)
  class(base_type) :: self
end subroutine copy

subroutine init_extended_type(self)
  class(extended_type) :: self
end subroutine init_extended_type
subroutine final_extended_type(self)
  class(extended_type) :: self
end subroutine final_extended_type
subroutine do_something(self)
  class(extended_type) :: self
end subroutine do_something

subroutine init_extended_extended_type(self)
  class(extended_extended_type) :: self
end subroutine init_extended_extended_type
subroutine final_extended_extended_type(self)
  class(extended_extended_type) :: self
end subroutine final_extended_extended_type
subroutine do_something_else(self)
  class(extended_extended_type) :: self
end subroutine do_something_else
end module some_mod
""".strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    base_type = module['base_type']
    extended_type = module['extended_type']
    extended_extended_type = module['extended_extended_type']

    assert base_type.variables == ('memberA', 'memberB', 'init', 'final', 'copy')
    assert base_type.variables[2].type.bind_names[0] == 'init_base_type'
    assert base_type.variables[3].type.bind_names[0] == 'final_base_type'
    assert not base_type.variables[4].type.bind_names
    assert all(s.scope is base_type for d in base_type.declarations for s in d.symbols)
    assert base_type.imported_symbols == ()
    assert not base_type.imported_symbol_map

    assert extended_type.variables == ('memberC', 'init', 'final', 'do_something', 'memberA', 'memberB', 'copy')
    assert extended_type.variables[1].type.bind_names[0] == 'init_extended_type'
    assert extended_type.variables[2].type.bind_names[0] == 'final_extended_type'
    assert not extended_type.variables[3].type.bind_names
    assert not extended_type.variables[6].type.bind_names
    assert all(s.scope is extended_type for d in extended_type.declarations for s in d.symbols)
    assert extended_type.imported_symbols == ()
    assert not extended_type.imported_symbol_map
    #check for non-empty declarations
    assert all(decl.symbols for decl in extended_type.declarations)


    assert extended_extended_type.variables == ('memberD', 'init', 'final', 'do_something', 'memberC',
                                                'memberA', 'memberB', 'copy')
    assert extended_extended_type.variables[1].type.bind_names[0] == 'init_extended_extended_type'
    assert extended_extended_type.variables[2].type.bind_names[0] == 'final_extended_extended_type'
    assert extended_extended_type.variables[3].type.bind_names[0] == 'do_something_else'
    assert not extended_extended_type.variables[7].type.bind_names
    assert all(s.scope is extended_extended_type for d in extended_extended_type.declarations for s in d.symbols)
    assert extended_extended_type.imported_symbols == ()
    assert not extended_extended_type.imported_symbol_map
    #check for non-empty declarations
    assert all(decl.symbols for decl in extended_extended_type.declarations)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('qualified_import', (True, False))
def test_derived_type_inheritance_missing_parent(frontend, qualified_import, tmp_path):
    fcode_parent = """
module parent_mod
    implicit none
    type, abstract, public :: parent_type
        integer :: val
    end type parent_type
end module parent_mod
    """.strip()

    fcode_derived = f"""
module derived_mod
    use parent_mod{", only: parent_type" if qualified_import else ""}
    implicit none
    type, public, extends(parent_type) :: derived_type
        integer :: val2
    end type derived_type
contains
    subroutine do_something(this)
        class(derived_type), intent(inout) :: this
        this%val = 1
        this%val2 = 2
    end subroutine do_something
end module derived_mod
    """.strip()

    parent = Module.from_source(fcode_parent, frontend=frontend, xmods=[tmp_path])

    # Without enrichment we obtain only DEFERRED type information (but don't fail!)
    derived = Module.from_source(fcode_derived, frontend=frontend, xmods=[tmp_path])
    assert derived['derived_type'].parent_type == BasicType.DEFERRED

    # With enrichment we obtain the parent type from the parent module
    derived = Module.from_source(fcode_derived, frontend=frontend, xmods=[tmp_path], definitions=[parent])
    assert derived['derived_type'].parent_type is parent['parent_type']
