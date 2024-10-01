# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Module, Subroutine
from loki.build import jit_compile, jit_compile_lib, Builder, Obj
from loki.frontend import available_frontends
from loki.ir import FindNodes, Section, Assignment, CallStatement, Intrinsic
from loki.tools import as_tuple
from loki.types import BasicType

from loki.transformations.extract.outline import outline_pragma_regions


@pytest.fixture(scope='function', name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path/'build')
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_outline_pragma_regions(tmp_path, frontend):
    """
    A very simple :any:`outline_pragma_regions` test case
    """
    fcode = """
subroutine test_outline(a, b, c)
  integer, intent(out) :: a, b, c

  a = 5
  a = 1

!$loki outline in(a) out(b)
  b = a
!$loki end outline

  c = a + b
end subroutine test_outline
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 1 and b == 1 and c == 2

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = outline_pragma_regions(routine)
    assert len(routines) == 1 and routines[0].name == f'{routine.name}_outlined_0'

    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assert len(FindNodes(Assignment).visit(routines[0].body)) == 1
    assert len(FindNodes(CallStatement).visit(routine.body)) == 1

    # Test transformation
    contains = Section(body=as_tuple([Intrinsic('CONTAINS'), *routines, routine]))
    module = Module(name=f'{routine.name}_mod', spec=None, contains=contains)
    mod_filepath = tmp_path/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 1 and b == 1 and c == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_outline_pragma_regions_multiple(tmp_path, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    """
    fcode = """
subroutine test_outline_mult(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1
  a = a + 1
  a = a + 1
!$loki outline name(oiwjfklsf) inout(a)
  a = a + 1
!$loki end outline
  a = a + 1

!$loki outline in(a) out(b)
  b = a
!$loki end outline

!$loki outline in(a,b) out(c)
  c = a + b
!$loki end outline
end subroutine test_outline_mult
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 5 and b == 5 and c == 10

    assert len(FindNodes(Assignment).visit(routine.body)) == 7
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = outline_pragma_regions(routine)
    assert len(routines) == 3
    assert routines[0].name == 'oiwjfklsf'
    assert all(routines[i].name == f'{routine.name}_outlined_{i}' for i in (1,2))

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert all(len(FindNodes(Assignment).visit(r.body)) == 1 for r in routines)
    assert len(FindNodes(CallStatement).visit(routine.body)) == 3

    # Test transformation
    contains = Section(body=as_tuple([Intrinsic('CONTAINS'), *routines, routine]))
    module = Module(name=f'{routine.name}_mod', spec=None, contains=contains)
    mod_filepath = tmp_path/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 5 and b == 5 and c == 10


@pytest.mark.parametrize('frontend', available_frontends())
def test_outline_pragma_regions_arguments(tmp_path, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    and automatic derivation of arguments
    """
    fcode = """
subroutine test_outline_args(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1
  a = a + 1
  a = a + 1
!$loki outline name(func_a)
  a = a + 1
!$loki end outline
  a = a + 1

!$loki outline name(func_b)
  b = a
!$loki end outline

! partially override arguments
!$loki outline name(func_c) inout(b)
  c = a + b
!$loki end outline
end subroutine test_outline_args
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 5 and b == 5 and c == 10

    assert len(FindNodes(Assignment).visit(routine.body)) == 7
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = outline_pragma_regions(routine)
    assert len(routines) == 3
    assert [r.name for r in routines] == ['func_a', 'func_b', 'func_c']

    assert len(routines[0].arguments) == 1
    assert routines[0].arguments[0] == 'a' and routines[0].arguments[0].type.intent == 'inout'

    assert {str(a) for a in routines[1].arguments} == {'a', 'b'}
    assert routines[1].variable_map['a'].type.intent == 'in'
    assert routines[1].variable_map['b'].type.intent == 'out'

    assert {str(a) for a in routines[2].arguments} == {'a', 'b', 'c'}
    assert routines[2].variable_map['a'].type.intent == 'in'
    assert routines[2].variable_map['b'].type.intent == 'inout'
    assert routines[2].variable_map['c'].type.intent == 'out'

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert all(len(FindNodes(Assignment).visit(r.body)) == 1 for r in routines)
    assert len(FindNodes(CallStatement).visit(routine.body)) == 3

    # Test transformation
    contains = Section(body=as_tuple([Intrinsic('CONTAINS'), *routines, routine]))
    module = Module(name=f'{routine.name}_mod', spec=None, contains=contains)
    mod_filepath = tmp_path/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 5 and b == 5 and c == 10


@pytest.mark.parametrize('frontend', available_frontends())
def test_outline_pragma_regions_arrays(tmp_path, frontend):
    """
    Test hoisting with array variables
    """
    fcode = """
subroutine test_outline_arr(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: j

!$loki outline
  do j=1,n
    a(j) = j
  end do
!$loki end outline

!$loki outline
  do j=1,n
    b(j) = j
  end do
!$loki end outline

!$loki outline
  do j=1,n-1
    b(j) = b(j+1) - a(j)
  end do
  b(n) = 1
!$loki end outline
end subroutine test_outline_arr
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 10
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    function(a, b, n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == [1] * n)

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = outline_pragma_regions(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 0
    assert len(FindNodes(CallStatement).visit(routine.body)) == 3

    assert len(routines) == 3

    assert {(str(a), a.type.intent) for a in routines[0].arguments} == {('a(n)', 'out'), ('n', 'in')}
    assert {(str(a), a.type.intent) for a in routines[1].arguments} == {('b(n)', 'out'), ('n', 'in')}
    assert {(str(a), a.type.intent) for a in routines[2].arguments} == {('a(n)', 'in'), ('b(n)', 'inout'), ('n', 'in')}
    assert routines[0].variable_map['a'].dimensions[0].scope is routines[0]

    # Test transformation
    contains = Section(body=as_tuple([Intrinsic('CONTAINS'), *routines, routine]))
    module = Module(name=f'{routine.name}_mod', spec=None, contains=contains)
    mod_filepath = tmp_path/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    mod_function(a, b, n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == [1] * n)


@pytest.mark.parametrize('frontend', available_frontends())
def test_outline_pragma_regions_imports(tmp_path, builder, frontend):
    """
    Test hoisting with correct treatment of imports
    """
    fcode_module = """
module outline_mod
  implicit none
  integer, parameter :: param = 1
  integer :: arr1(10)
  integer :: arr2(10)
end module outline_mod
    """.strip()

    fcode = """
module test_outline_imps_mod
  implicit none
contains
  subroutine test_outline_imps(a, b)
    use outline_mod, only: param, arr1, arr2
    integer, intent(out) :: a(10), b(10)
    integer :: j

!$loki outline
    do j=1,10
      a(j) = param
    end do
!$loki end outline

!$loki outline
    do j=1,10
      arr1(j) = j+1
    end do
!$loki end outline

    arr2(:) = arr1(:)

!$loki outline
    do j=1,10
      b(j) = arr2(j) - a(j)
    end do
!$loki end outline
  end subroutine test_outline_imps
end module test_outline_imps_mod
"""
    ext_module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    module = Module.from_source(fcode, frontend=frontend, definitions=ext_module, xmods=[tmp_path])
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module, ext_module], path=tmp_path, name=refname, builder=builder)
    function = getattr(getattr(reference, module.name), module.subroutines[0].name)

    # Test the reference solution
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    function(a, b)
    assert np.all(a == [1] * 10)
    assert np.all(b == range(1,11))
    (tmp_path/f'{module.name}.f90').unlink()

    assert len(FindNodes(Assignment).visit(module.subroutines[0].body)) == 4
    assert len(FindNodes(CallStatement).visit(module.subroutines[0].body)) == 0

    # Apply transformation
    routines = outline_pragma_regions(module.subroutines[0])

    assert len(FindNodes(Assignment).visit(module.subroutines[0].body)) == 1
    assert len(FindNodes(CallStatement).visit(module.subroutines[0].body)) == 3

    assert len(routines) == 3

    assert {(str(a), a.type.intent) for a in routines[0].arguments} == {('a(10)', 'out')}
    assert {(str(a), a.type.intent) for a in routines[1].arguments} == set()
    assert {(str(a), a.type.intent) for a in routines[2].arguments} == {('a(10)', 'in'), ('b(10)', 'out')}

    # Insert created routines into module
    module.contains.append(routines)

    obj = jit_compile_lib([module, ext_module], path=tmp_path, name=f'{module.name}_{frontend}', builder=builder)
    mod_function = getattr(getattr(obj, module.name), module.subroutines[0].name)

    # Test transformation
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    mod_function(a, b)
    assert np.all(a == [1] * 10)
    assert np.all(b == range(1,11))


@pytest.mark.parametrize('frontend', available_frontends())
def test_outline_pragma_regions_derived_args(tmp_path, builder, frontend):
    """
    Test subroutine extraction with derived-type arguments.
    """

    fcode = """
module test_outline_dertype_mod
  implicit none

  type rick
    integer :: a(10), b(10)
  end type rick
contains

  subroutine test_outline_imps(a, b)
    integer, intent(out) :: a(10), b(10)
    type(rick) :: dave
    integer :: j

    dave%a(:) = a(:)
    dave%b(:) = b(:)

!$loki outline
    do j=1,10
      dave%a(j) = j + 1
    end do

    dave%b(:) = dave%b(:) + 42
!$loki end outline

    a(:) = dave%a(:)
    b(:) = dave%b(:)
  end subroutine test_outline_imps
end module test_outline_dertype_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module], path=tmp_path, name=refname, builder=builder)
    function = getattr(getattr(reference, module.name), module.subroutines[0].name)

    # Test the reference solution
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    function(a, b)
    assert np.all(a == range(2,12))
    assert np.all(b == 42)
    (tmp_path/f'{module.name}.f90').unlink()

    assert len(FindNodes(Assignment).visit(module.subroutines[0].body)) == 6
    assert len(FindNodes(CallStatement).visit(module.subroutines[0].body)) == 0

    # Apply transformation
    routines = outline_pragma_regions(module.subroutines[0])

    assert len(FindNodes(Assignment).visit(module.subroutines[0].body)) == 4
    assert len(FindNodes(CallStatement).visit(module.subroutines[0].body)) == 1

    # Check for a single derived-type argument
    assert len(routines) == 1
    assert len(routines[0].arguments) == 1
    assert routines[0].arguments[0] == 'dave'
    assert routines[0].arguments[0].type.dtype.name == 'rick'
    assert routines[0].arguments[0].type.intent == 'inout'

    # Insert created routines into module
    module.contains.append(routines)

    obj = jit_compile_lib([module], path=tmp_path, name=f'{module.name}_{frontend}', builder=builder)
    mod_function = getattr(getattr(obj, module.name), module.subroutines[0].name)

    # Test the transformed module solution
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    mod_function(a, b)
    assert np.all(a == range(2,12))
    assert np.all(b == 42)


@pytest.mark.parametrize('frontend', available_frontends())
def test_outline_pragma_regions_associates(tmp_path, builder, frontend):
    """
    Test subroutine extraction with derived-type arguments.
    """

    fcode = """
module test_outline_assoc_mod
  implicit none

  type rick
    integer :: a(10), b(10)
  end type rick
contains

  subroutine test_outline_imps(a, b)
    integer, intent(out) :: a(10), b(10)
    type(rick) :: dave
    integer :: j

    associate(c=>dave%a, d=>dave%b)

    c(:) = a(:)
    d(:) = b(:)

!$loki outline
    do j=1,10
      c(j) = j + 1
    end do

    d(:) = d(:) + 42
!$loki end outline

    a(:) = c(:)
    b(:) = d(:)
    end associate
  end subroutine test_outline_imps
end module test_outline_assoc_mod
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module.subroutines[0]
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module], path=tmp_path, name=refname, builder=builder)
    function = getattr(getattr(reference, module.name), routine.name)

    # Test the reference solution
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    function(a, b)
    assert np.all(a == range(2,12))
    assert np.all(b == 42)
    (tmp_path/f'{module.name}.f90').unlink()

    assert len(FindNodes(Assignment).visit(routine.body)) == 6
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    outlined = outline_pragma_regions(routine)

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments == ('d', 'c')

    # Check for a single derived-type argument
    assert len(outlined) == 1
    assert len(outlined[0].arguments) == 2
    assert outlined[0].arguments[0].name == 'd'
    assert outlined[0].arguments[0].type.shape == (10,)
    assert outlined[0].arguments[0].type.dtype == BasicType.INTEGER
    assert outlined[0].arguments[0].type.intent == 'inout'
    assert outlined[0].arguments[1].name == 'c'
    assert outlined[0].arguments[1].type.shape == (10,)
    assert outlined[0].arguments[1].type.dtype == BasicType.INTEGER
    assert outlined[0].arguments[1].type.intent == 'out'

    # Insert created routines into module
    module.contains.append(outlined)

    obj = jit_compile_lib(
        [module], path=tmp_path, name=f'{module.name}_{frontend}', builder=builder
    )
    mod_function = getattr(getattr(obj, module.name), routine.name)
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    mod_function(a, b)
    assert np.all(a == range(2,12))
    assert np.all(b == 42)
