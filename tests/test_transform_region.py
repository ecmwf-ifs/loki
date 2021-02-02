from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, clean_test
from loki import Module, Subroutine, OFP, OMNI, FP, FindNodes, Loop, Assignment, CallStatement
from loki.expression import symbols as sym
from loki.transform import region_hoist, region_to_call


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 1 and b == 1 and c == 2

    # Apply transformation
    region_hoist(routine)
    hoisted_filepath = here/('%s_hoisted_%s.f90' % (routine.name, frontend))
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    a, b, c = hoisted_function()
    assert a == 1 and b == 5 and c == 6

    clean_test(filepath)
    clean_test(hoisted_filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
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

    hoisted_filepath = here/('%s_hoisted_%s.f90' % (routine.name, frontend))
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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 5 and b == 5 and c == 10

    # Apply transformation
    region_hoist(routine)
    hoisted_filepath = here/('%s_hoisted_%s.f90' % (routine.name, frontend))
    hoisted_function = jit_compile(routine, filepath=hoisted_filepath, objname=routine.name)

    # Test transformation
    a, b, c = hoisted_function()
    assert a == 5 and b == 1 and c == 3

    clean_test(filepath)
    clean_test(hoisted_filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
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

    hoisted_filepath = here/('%s_hoisted_%s.f90' % (routine.name, frontend))
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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
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

    hoisted_filepath = here/('%s_hoisted_%s.f90' % (routine.name, frontend))
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


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_region_to_call(here, frontend):
    """
    A very simple region-to-call test case
    """
    fcode = """
subroutine transform_region_to_call(a, b, c)
  integer, intent(out) :: a, b, c

  a = 5
  a = 1

!$loki region-to-call in(a) out(b)
  b = a
!$loki end region-to-call

  c = a + b
end subroutine transform_region_to_call
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    a, b, c = function()
    assert a == 1 and b == 1 and c == 2

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert len(FindNodes(CallStatement).visit(routine.body)) == 0

    # Apply transformation
    routines = region_to_call(routine)
    assert len(routines) == 1 and routines[0].name == '{}_region_to_call_0'.format(routine.name)

    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assert len(FindNodes(Assignment).visit(routines[0].body)) == 1
    assert len(FindNodes(CallStatement).visit(routine.body)) == 1

    # Test transformation
    module = Module(name='{}_mod'.format(routine.name), spec=(), routines=[*routines, routine])
    mod_filepath = here/('%s_converted_%s.f90' % (module.name, frontend))
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 1 and b == 1 and c == 2

    clean_test(filepath)
    clean_test(mod_filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_region_to_call_multiple(here, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    """
    fcode = """
subroutine transform_region_to_call_multiple(a, b, c)
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
end subroutine transform_region_to_call_multiple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
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
    assert all(routines[i].name == '{}_region_to_call_{}'.format(routine.name, i) for i in (1,2))

    assert len(FindNodes(Assignment).visit(routine.body)) == 4
    assert all(len(FindNodes(Assignment).visit(r.body)) == 1 for r in routines)
    assert len(FindNodes(CallStatement).visit(routine.body)) == 3

    # Test transformation
    module = Module(name='{}_mod'.format(routine.name), spec=(), routines=[*routines, routine])
    mod_filepath = here/('%s_converted_%s.f90' % (module.name, frontend))
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 5 and b == 5 and c == 10

    clean_test(filepath)
    clean_test(mod_filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_region_to_call_arguments(here, frontend):
    """
    Test hoisting with multiple groups and multiple regions per group
    and automatic derivation of arguments
    """
    fcode = """
subroutine transform_region_to_call_arguments(a, b, c)
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
end subroutine transform_region_to_call_arguments
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
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
    module = Module(name='{}_mod'.format(routine.name), spec=(), routines=[*routines, routine])
    mod_filepath = here/('%s_converted_%s.f90' % (module.name, frontend))
    mod = jit_compile(module, filepath=mod_filepath, objname=module.name)
    mod_function = getattr(mod, routine.name)

    a, b, c = mod_function()
    assert a == 5 and b == 5 and c == 10

    clean_test(filepath)
    clean_test(mod_filepath)
