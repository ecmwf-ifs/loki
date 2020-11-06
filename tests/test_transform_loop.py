from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, clean_test, parse_expression
from loki import Subroutine, OFP, OMNI, FP, FindNodes, Loop
from loki.transform import loop_fusion, Polyhedron
from loki.expression import symbols as sym


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('variables_scopes, lbounds_scopes, ubounds_scopes, A, b, variable_names', [
    # do i=0,5: do j=i,7: ...
    ([parse_expression('i'), parse_expression('j')],
     [parse_expression('0'), parse_expression('i')], [parse_expression('5'), parse_expression('7')],
     [[-1, 0], [1, 0], [1, -1], [0, 1]], [0, 5, 0, 7], ['i', 'j']),
    # do i=1,n: do j=0,2*i+1: do k=a,b: ...
    ([parse_expression('i'), parse_expression('j'), parse_expression('k')],
     [parse_expression('1'), parse_expression('0'), parse_expression('a')],
     [parse_expression('n'), parse_expression('2*i+1'), parse_expression('b')],
     [[-1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, -1], [0, -1, 0, 0, 0, 0], [-2, 1, 0, 0, 0, 0],
      [0, 0, -1, 1, 0, 0], [0, 0, 1, 0, -1, 0]], [-1, 0, 0, 1, 0, 0],
      ['i', 'j', 'k', 'a', 'b', 'n']),
    # do jk=1,klev: ...
    ([parse_expression('jk')], [parse_expression('1')], [parse_expression('klev')],
     [[-1, 0], [1, -1]], [-1, 0], ['jk', 'klev']),
    # do JK=1,klev-1: ...
    ([parse_expression('JK')], [parse_expression('1')], [parse_expression('klev - 1')],
     [[-1, 0], [1, -1]], [-1, -1], ['jk', 'klev']),
    # do jk=ncldtop,klev: ...
    ([parse_expression('jk')], [parse_expression('ncldtop')], [parse_expression('klev')],
     [[-1, 0, 1], [1, -1, 0]], [0, 0], ['jk', 'klev', 'ncldtop']),
    # do jk=1,KLEV+1: ...
    ([parse_expression('jk')], [parse_expression('1')], [parse_expression('KLEV+1')],
     [[-1, 0], [1, -1]], [-1, 1], ['jk', 'klev']),
])
def test_polyhedron_from_loop_ranges(variables_scopes, lbounds_scopes, ubounds_scopes, A, b, variable_names):
    """
    Test converting loop ranges to polyedron representation of iteration space.
    """
    loop_variables, _ = zip(*variables_scopes)
    lbounds, _ = zip(*lbounds_scopes)
    ubounds, _ = zip(*ubounds_scopes)
    loop_ranges = [sym.LoopRange((l, u)) for l, u in zip(lbounds, ubounds)]
    p = Polyhedron.from_loop_ranges(loop_variables, loop_ranges)
    assert np.all(p.A == np.array(A, dtype=np.dtype(int)))
    assert np.all(p.b == np.array(b, dtype=np.dtype(int)))
    assert p.variables == variable_names


def test_polyhedron_from_loop_ranges_failures():
    """
    Test known limitation of the conversion from loop ranges to polyhedron.
    """
    # m*n is non-affine and thus can't be represented
    loop_variable, _ = parse_expression('i')
    lower_bound, _ = parse_expression('1')
    upper_bound, _ = parse_expression('m * n')
    loop_range = sym.LoopRange((lower_bound, upper_bound))
    with pytest.raises(ValueError):
        _ = Polyhedron.from_loop_ranges([loop_variable], [loop_range])

    # no functionality to flatten exponentials, yet
    upper_bound, _ = parse_expression('5**2')
    loop_range = sym.LoopRange((lower_bound, upper_bound))
    with pytest.raises(ValueError):
        _ = Polyhedron.from_loop_ranges([loop_variable], [loop_range])


@pytest.mark.parametrize('A, b, variable_names, lower_bounds, upper_bounds', [
    # do i=1,n: ...
    ([[-1, 0], [1, -1]], [-1, 0], ['i', 'n'], [['1'], ['i']], [['n'], []]),
    # do i=1,10: ...
    ([[-1], [1]], [-1, 10], ['i'], [['1']], [['10']]),
    # do i=0,5: do j=i,7: ...
    ([[-1, 0], [1, 0], [1, -1], [0, 1]], [0, 5, 0, 7], ['i', 'j'], [['0'], ['i']], [['5', 'j'], ['7']]),
    # do i=1,n: do j=0,2*i+1: do k=a,b: ...
    ([[-1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, -1], [0, -1, 0, 0, 0, 0], [-2, 1, 0, 0, 0, 0],
      [0, 0, -1, 1, 0, 0], [0, 0, 1, 0, -1, 0]], [-1, 0, 0, 1, 0, 0],
      ['i', 'j', 'k', 'a', 'b', 'n'],                               # variable names
      [['1', '-1 / 2 + j / 2'], ['0'], ['a'], [], ['k'], ['i']],    # lower bounds
      [['n'], ['1 + 2*i'], ['b'], ['k'], [], []]),                  # upper bounds
])
def test_polyhedron_bounds(A, b, variable_names, lower_bounds, upper_bounds):
    """
    Test the production of lower and upper bounds.
    """
    var, scope = parse_expression(variable_names[0])
    variables = [var] + [parse_expression(v, scope) for v in variable_names[1:]]
    p = Polyhedron(A, b, variables)
    for var, ref_bounds in zip(variables, lower_bounds):
        lbounds = p.lower_bounds(var)
        assert len(lbounds) == len(ref_bounds)
        assert all(str(b1) == b2 for b1, b2 in zip(lbounds, ref_bounds))
    for var, ref_bounds in zip(variables, upper_bounds):
        ubounds = p.upper_bounds(var)
        assert len(ubounds) == len(ref_bounds)
        assert all(str(b1) == b2 for b1, b2 in zip(ubounds, ref_bounds))


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_loop_fuse_matching(here, frontend):
    """
    Apply loop fusion for two loops with matching iteration spaces.
    """
    fcode = """
subroutine transform_loop_fuse_matching(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: i

  !$loki loop-fusion
  do i=1,n
    a(i) = i
  end do

  !$loki loop-fusion
  do i=1,n
    b(i) = n-i+1
  end do
end subroutine transform_loop_fuse_matching
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    fused_filepath = here/('%s_fused_%s.f90' % (routine.name, frontend))
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fused_function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_loop_fuse_subranges(here, frontend):
    """
    Apply loop fusion with annotated range for loops with
    non-matching iteration spaces.
    """
    fcode = """
subroutine transform_loop_fuse_subranges(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: i, j

  a(:) = 0
  b(:) = 0

  !$loki loop-fusion
  do i=1,n
    a(i) = a(i) + i
  end do

  !$loki loop-fusion range(1:n)
  do j=1,15
    b(j) = b(j) + n-j+1
  end do

  !$loki loop-fusion range(1:n)
  do i=16,n
    b(i) = b(i) + n-i+1
  end do
end subroutine transform_loop_fuse_subranges
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 3
    loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    fused_filepath = here/('%s_fused_%s.f90' % (routine.name, frontend))
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fused_function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_loop_fuse_groups(here, frontend):
    """
    Apply loop fusion for multiple loop fusion groups.
    """
    fcode = """
subroutine transform_loop_fuse_groups(a, b, c, n)
  integer, intent(out) :: a(n), b(n), c(n)
  integer, intent(in) :: n
  integer :: i

  c(1) = 1

  !$loki loop-fusion group(g1)
  do i=1,n
    a(i) = i
  end do

  !$loki loop-fusion group(g1)
  do i=1,n
    b(i) = n-i+1
  end do

  !$loki loop-fusion group(loop-group2)
  do i=1,n
    a(i) = a(i) + 1
  end do

  !$loki loop-fusion group(loop-group2)
  do i=1,n
    b(i) = b(i) + 1
  end do

  !$loki loop-fusion group(g1) range(1:n)
  do i=2,n
    c(i) = c(i-1) + 1
  end do
end subroutine transform_loop_fuse_groups
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('%s_%s.f90' % (routine.name, frontend))
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    c = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, c=c, n=n)
    assert np.all(a == range(2, n+2))
    assert np.all(b == range(n+1, 1, -1))
    assert np.all(c == range(1, n+1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 5
    loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 2

    fused_filepath = here/('%s_fused_%s.f90' % (routine.name, frontend))
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    c = np.zeros(shape=(n,), dtype=np.int32)
    fused_function(a=a, b=b, c=c, n=n)
    assert np.all(a == range(2, n+2))
    assert np.all(b == range(n+1, 1, -1))
    assert np.all(c == range(1, n+1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_loop_fuse_failures(here, frontend):
    """
    Test that loop-fusion fails for known mistakes.
    """
    fcode = """
subroutine transform_loop_fuse_non_matching(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: i

  !$loki loop-fusion group(1)
  do i=1,n
    a(i) = i
  end do

  !$loki loop-fusion group(1)
  do i=0,n-1
    b(i+1) = n-i
  end do
end subroutine transform_loop_fuse_non_matching
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    with pytest.raises(RuntimeError):
        loop_fusion(routine)
