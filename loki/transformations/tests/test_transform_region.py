# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from loki import (
    Module, Subroutine, Section, as_tuple, FindNodes, Loop,
    Assignment, CallStatement, Intrinsic
)
from loki.build import jit_compile, jit_compile_lib, clean_test, Builder
from loki.expression import symbols as sym
from loki.frontend import available_frontends

from loki.transformations.array_indexing import normalize_range_indexing
from loki.transformations.transform_region import (
    region_hoist, region_to_call
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(here):
    return Builder(source_dirs=here, build_dir=here/'build')


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist(here, frontend):
    """
    A very simple hoisting example
    """
    fcode = """
subroutine transform_region_hoist(a, b, c)
  integer, intent(out) :: a, b, c

  a = 5

!$loki region-hoist target

  a = 1

!$loki region-hoist
  b = a
!$loki end region-hoist

  c = a + b
end subroutine transform_region_hoist
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 1 and b == 1 and c == 2

    # Apply transformation
    region_hoist(routine)
    hoisted_filepath = here/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    a, b, c = hoisted_function()
    assert a == 1 and b == 5 and c == 6

    clean_test(filepath)
    clean_test(hoisted_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_inlined_pragma(here, frontend):
    """
    Hoisting when pragmas are potentially inlined into other nodes.
    """
    fcode = """
subroutine transform_region_hoist_inlined_pragma(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl

!$loki region-hoist target

  do jl=1,klon
    a(jl, 1) = jl
  end do

  do jk=2,klev
    do jl=1,klon
      a(jl, jk) = a(jl, jk-1)
    end do
  end do

!$loki region-hoist
  do jk=1,klev
    b(1, jk) = jk
  end do
!$loki end region-hoist

  do jk=1,klev
    do jl=2,klon
      b(jl, jk) = b(jl-1, jk)
    end do
  end do
end subroutine transform_region_hoist_inlined_pragma
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    klon, klev = 32, 100
    ref_a = np.array([[jl + 1] * klev for jl in range(klon)], order='F')
    ref_b = np.array([[jk + 1 for jk in range(klev)] for _ in range(klon)], order='F')

    # Test the reference solution
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['jl', 'jk', 'jl', 'jk', 'jk', 'jl']

    # Apply transformation
    region_hoist(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['jk', 'jl', 'jk', 'jl', 'jk', 'jl']

    hoisted_filepath = here/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    hoisted_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    clean_test(filepath)
    clean_test(hoisted_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_multiple(here, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    """
    fcode = """
subroutine transform_region_hoist_multiple(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1

!$loki region-hoist target
!$loki region-hoist target group(some-group)

  a = a + 1
  a = a + 1
!$loki region-hoist group(some-group)
  a = a + 1
!$loki end region-hoist
  a = a + 1

!$loki region-hoist
  b = a
!$loki end region-hoist

!$loki region-hoist group(some-group)
  c = a + b
!$loki end region-hoist
end subroutine transform_region_hoist_multiple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 5 and b == 5 and c == 10

    # Apply transformation
    region_hoist(routine)
    hoisted_filepath = here/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    a, b, c = hoisted_function()
    assert a == 5 and b == 1 and c == 3

    clean_test(filepath)
    clean_test(hoisted_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_collapse(here, frontend):
    """
    Use collapse with region-hoist.
    """
    fcode = """
subroutine transform_region_hoist_collapse(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl

!$loki region-hoist target

  do jl=1,klon
    a(jl, 1) = jl
  end do

  do jk=2,klev
    do jl=1,klon
      a(jl, jk) = a(jl, jk-1)
    end do
  end do

  do jk=1,klev
!$loki region-hoist collapse(1)
    b(1, jk) = jk
!$loki end region-hoist
  end do

  do jk=1,klev
    do jl=2,klon
      b(jl, jk) = b(jl-1, jk)
    end do
  end do
end subroutine transform_region_hoist_collapse
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    klon, klev = 32, 100
    ref_a = np.array([[jl + 1] * klev for jl in range(klon)], order='F')
    ref_b = np.array([[jk + 1 for jk in range(klev)] for _ in range(klon)], order='F')

    # Test the reference solution
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['jl', 'jk', 'jl', 'jk', 'jk', 'jl']

    # Apply transformation
    region_hoist(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 7
    assert [str(loop.variable) for loop in loops] == ['jk', 'jl', 'jk', 'jl', 'jk', 'jk', 'jl']

    hoisted_filepath = here/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    hoisted_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    clean_test(filepath)
    clean_test(hoisted_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_hoist_promote(here, frontend):
    """
    Use collapse with region-hoist.
    """
    fcode = """
subroutine transform_region_hoist_promote(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl, b_tmp

!$loki region-hoist target

  do jl=1,klon
    a(jl, 1) = jl
  end do

  do jk=2,klev
    do jl=1,klon
      a(jl, jk) = a(jl, jk-1)
    end do
  end do

  do jk=1,4
    b(1, jk) = jk
  end do

  do jk=5,klev
!$loki region-hoist collapse(1) promote(b_tmp)
    b_tmp = jk + 1
!$loki end region-hoist
    b(1, jk) = b_tmp - 1
  end do

  do jk=1,klev
    do jl=2,klon
      b(jl, jk) = b(jl-1, jk)
    end do
  end do
end subroutine transform_region_hoist_promote
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    klon, klev = 32, 100
    ref_a = np.array([[jl + 1] * klev for jl in range(klon)], order='F')
    ref_b = np.array([[jk + 1 for jk in range(klev)] for _ in range(klon)], order='F')

    # Test the reference solution
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 7
    assert [str(loop.variable) for loop in loops] == ['jl', 'jk', 'jl', 'jk', 'jk', 'jk', 'jl']

    assert isinstance(routine.variable_map['b_tmp'], sym.Scalar)

    # Apply transformation
    region_hoist(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 8
    assert [str(loop.variable) for loop in loops] == ['jk', 'jl', 'jk', 'jl', 'jk', 'jk', 'jk', 'jl']

    b_tmp = routine.variable_map['b_tmp']
    assert isinstance(b_tmp, sym.Array) and len(b_tmp.type.shape) == 1
    assert str(b_tmp.type.shape[0]) == 'klev'

    hoisted_filepath = here/(f'{routine.name}_hoisted_{frontend}.f90')
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    hoisted_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == ref_a)
    assert np.all(b == ref_b)

    clean_test(filepath)
    clean_test(hoisted_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_to_call(here, frontend):
    """
    A very simple region-to-call test case
    """
    fcode = """
subroutine reg_to_call(a, b, c)
  integer, intent(out) :: a, b, c

  a = 5
  a = 1

!$loki region-to-call in(a) out(b)
  b = a
!$loki end region-to-call

  c = a + b
end subroutine reg_to_call
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 1 and b == 1 and c == 2

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = region_to_call(routine)
    assert len(routines) == 1 and routines[0].name == f'{routine.name}_region_to_call_0'

    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assert len(FindNodes(Assignment).visit(routines[0].body)) == 1
    assert len(FindNodes(CallStatement).visit(routine.body)) == 1

    # Test transformation
    contains = Section(body=as_tuple([Intrinsic('CONTAINS'), *routines, routine]))
    module = Module(name=f'{routine.name}_mod', spec=None, contains=contains)
    mod_filepath = here/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 1 and b == 1 and c == 2

    clean_test(filepath)
    clean_test(mod_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_to_call_multiple(here, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    """
    fcode = """
subroutine reg_to_call_mult(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1
  a = a + 1
  a = a + 1
!$loki region-to-call name(oiwjfklsf) inout(a)
  a = a + 1
!$loki end region-to-call
  a = a + 1

!$loki region-to-call in(a) out(b)
  b = a
!$loki end region-to-call

!$loki region-to-call in(a,b) out(c)
  c = a + b
!$loki end region-to-call
end subroutine reg_to_call_mult
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 5 and b == 5 and c == 10

    assert len(FindNodes(Assignment).visit(routine.body)) == 7
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = region_to_call(routine)
    assert len(routines) == 3
    assert routines[0].name == 'oiwjfklsf'
    assert all(routines[i].name == f'{routine.name}_region_to_call_{i}' for i in (1,2))

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert all(len(FindNodes(Assignment).visit(r.body)) == 1 for r in routines)
    assert len(FindNodes(CallStatement).visit(routine.body)) == 3

    # Test transformation
    contains = Section(body=as_tuple([Intrinsic('CONTAINS'), *routines, routine]))
    module = Module(name=f'{routine.name}_mod', spec=None, contains=contains)
    mod_filepath = here/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 5 and b == 5 and c == 10

    clean_test(filepath)
    clean_test(mod_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_to_call_arguments(here, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    and automatic derivation of arguments
    """
    fcode = """
subroutine reg_to_call_args(a, b, c)
  integer, intent(out) :: a, b, c

  a = 1
  a = a + 1
  a = a + 1
!$loki region-to-call name(func_a)
  a = a + 1
!$loki end region-to-call
  a = a + 1

!$loki region-to-call name(func_b)
  b = a
!$loki end region-to-call

! partially override arguments
!$loki region-to-call name(func_c) inout(b)
  c = a + b
!$loki end region-to-call
end subroutine reg_to_call_args
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 5 and b == 5 and c == 10

    assert len(FindNodes(Assignment).visit(routine.body)) == 7
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = region_to_call(routine)
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
    mod_filepath = here/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 5 and b == 5 and c == 10

    clean_test(filepath)
    clean_test(mod_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_to_call_arrays(here, frontend):
    """
    Test hoisting with array variables
    """
    fcode = """
subroutine reg_to_call_arrays(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: j

!$loki region-to-call
  do j=1,n
    a(j) = j
  end do
!$loki end region-to-call

!$loki region-to-call
  do j=1,n
    b(j) = j
  end do
!$loki end region-to-call

!$loki region-to-call
  do j=1,n-1
    b(j) = b(j+1) - a(j)
  end do
  b(n) = 1
!$loki end region-to-call
end subroutine reg_to_call_arrays
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_range_indexing(routine)

    filepath = here/(f'{routine.name}_{frontend}.f90')
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
    routines = region_to_call(routine)

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
    mod_filepath = here/(f'{module.name}_converted_{frontend}.f90')
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    mod_function(a, b, n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == [1] * n)

    clean_test(filepath)
    clean_test(mod_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_region_to_call_imports(here, builder, frontend):
    """
    Test hoisting with correct treatment of imports
    """
    fcode_module = """
module region_to_call_mod
  implicit none
  integer, parameter :: param = 1
  integer :: arr1(10)
  integer :: arr2(10)
end module region_to_call_mod
    """.strip()

    fcode = """
module region_to_call_imports_mod
  implicit none
contains
  subroutine region_to_call_imports(a, b)
    use region_to_call_mod, only: param, arr1, arr2
    integer, intent(out) :: a(10), b(10)
    integer :: j

!$loki region-to-call
    do j=1,10
      a(j) = param
    end do
!$loki end region-to-call

!$loki region-to-call
    do j=1,10
      arr1(j) = j+1
    end do
!$loki end region-to-call

    arr2(:) = arr1(:)

!$loki region-to-call
    do j=1,10
      b(j) = arr2(j) - a(j)
    end do
!$loki end region-to-call
  end subroutine region_to_call_imports
end module region_to_call_imports_mod
"""
    ext_module = Module.from_source(fcode_module, frontend=frontend)
    module = Module.from_source(fcode, frontend=frontend, definitions=ext_module)
    normalize_range_indexing(module.subroutines[0])
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module, ext_module], path=here, name=refname, builder=builder)
    function = getattr(getattr(reference, module.name), module.subroutines[0].name)

    # Test the reference solution
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    function(a, b)
    assert np.all(a == [1] * 10)
    assert np.all(b == range(1,11))
    (here/f'{module.name}.f90').unlink()

    assert len(FindNodes(Assignment).visit(module.subroutines[0].body)) == 4
    assert len(FindNodes(CallStatement).visit(module.subroutines[0].body)) == 0

    # Apply transformation
    routines = region_to_call(module.subroutines[0])

    assert len(FindNodes(Assignment).visit(module.subroutines[0].body)) == 1
    assert len(FindNodes(CallStatement).visit(module.subroutines[0].body)) == 3

    assert len(routines) == 3

    assert {(str(a), a.type.intent) for a in routines[0].arguments} == {('a(10)', 'out')}
    assert {(str(a), a.type.intent) for a in routines[1].arguments} == set()
    assert {(str(a), a.type.intent) for a in routines[2].arguments} == {('a(10)', 'in'), ('b(10)', 'out')}

    # Insert created routines into module
    module.contains.append(routines)

    obj = jit_compile_lib([module, ext_module], path=here, name=f'{module.name}_{frontend}', builder=builder)
    mod_function = getattr(getattr(obj, module.name), module.subroutines[0].name)

    # Test transformation
    a = np.zeros(shape=(10,), dtype=np.int32)
    b = np.zeros(shape=(10,), dtype=np.int32)
    mod_function(a, b)
    assert np.all(a == [1] * 10)
    assert np.all(b == range(1,11))
    (here/f'{module.name}.f90').unlink()
    (here/f'{ext_module.name}.f90').unlink()
