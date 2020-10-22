from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, clean_test
from loki import Subroutine, OFP, OMNI, FP, FindNodes, Loop
from loki.transform import fuse_loops


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
  use iso_fortran_env, only: real64
  real(kind=real64), intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: i

  !$loki fuse
  do i=1,n
    a(i) = i
  end do

  !$loki fuse
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
    a = np.zeros(shape=(n,))
    b = np.zeros(shape=(n,))
    function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    fuse_loops(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    fused_filepath = here/('%s_fused_%s.f90' % (routine.name, frontend))
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    a = np.zeros(shape=(n,))
    b = np.zeros(shape=(n,))
    fused_function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    clean_test(filepath)
    clean_test(fused_filepath)
