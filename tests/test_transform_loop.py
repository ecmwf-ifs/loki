from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, clean_test
from loki import Subroutine, OFP, OMNI, FP, FindNodes, Loop
from loki.transform import loop_fusion


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


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
