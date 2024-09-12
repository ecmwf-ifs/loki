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

from loki.transformations.extract.marked import extract_marked_subroutines


@pytest.fixture(scope='function', name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path/'build')
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_extract_marked_subroutines(tmp_path, frontend):
    """
    A very simple :any:`extract_marked_subroutine` test case
    """
    fcode = """
subroutine test_extract(a, b, c)
  integer, intent(out) :: a, b, c

  a = 5
  a = 1

!$loki extract in(a) out(b)
  b = a
!$loki end extract

  c = a + b
end subroutine test_extract
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
    routines = extract_marked_subroutines(routine)
    assert len(routines) == 1 and routines[0].name == f'{routine.name}_extracted_0'

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
def test_extract_marked_subroutines_multiple(tmp_path, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    """
    fcode = """
subroutine test_extract_mult(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1
  a = a + 1
  a = a + 1
!$loki extract name(oiwjfklsf) inout(a)
  a = a + 1
!$loki end extract
  a = a + 1

!$loki extract in(a) out(b)
  b = a
!$loki end extract

!$loki extract in(a,b) out(c)
  c = a + b
!$loki end extract
end subroutine test_extract_mult
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
    routines = extract_marked_subroutines(routine)
    assert len(routines) == 3
    assert routines[0].name == 'oiwjfklsf'
    assert all(routines[i].name == f'{routine.name}_extracted_{i}' for i in (1,2))

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
def test_extract_marked_subroutines_arguments(tmp_path, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    and automatic derivation of arguments
    """
    fcode = """
subroutine test_extract_args(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1
  a = a + 1
  a = a + 1
!$loki extract name(func_a)
  a = a + 1
!$loki end extract
  a = a + 1

!$loki extract name(func_b)
  b = a
!$loki end extract

! partially override arguments
!$loki extract name(func_c) inout(b)
  c = a + b
!$loki end extract
end subroutine test_extract_args
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
    routines = extract_marked_subroutines(routine)
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
def test_extract_marked_subroutines_arrays(tmp_path, frontend):
    """
    Test hoisting with array variables
    """
    fcode = """
subroutine test_extract_arr(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: j

!$loki extract
  do j=1,n
    a(j) = j
  end do
!$loki end extract

!$loki extract
  do j=1,n
    b(j) = j
  end do
!$loki end extract

!$loki extract
  do j=1,n-1
    b(j) = b(j+1) - a(j)
  end do
  b(n) = 1
!$loki end extract
end subroutine test_extract_arr
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
    routines = extract_marked_subroutines(routine)

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
def test_extract_marked_subroutines_imports(tmp_path, builder, frontend):
    """
    Test hoisting with correct treatment of imports
    """
    fcode_module = """
module extract_mod
  implicit none
  integer, parameter :: param = 1
  integer :: arr1(10)
  integer :: arr2(10)
end module extract_mod
    """.strip()

    fcode = """
module test_extract_imps_mod
  implicit none
contains
  subroutine test_extract_imps(a, b)
    use extract_mod, only: param, arr1, arr2
    integer, intent(out) :: a(10), b(10)
    integer :: j

!$loki extract
    do j=1,10
      a(j) = param
    end do
!$loki end extract

!$loki extract
    do j=1,10
      arr1(j) = j+1
    end do
!$loki end extract

    arr2(:) = arr1(:)

!$loki extract
    do j=1,10
      b(j) = arr2(j) - a(j)
    end do
!$loki end extract
  end subroutine test_extract_imps
end module test_extract_imps_mod
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
    routines = extract_marked_subroutines(module.subroutines[0])

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
