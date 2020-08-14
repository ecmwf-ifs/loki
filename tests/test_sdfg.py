import importlib
from pathlib import Path
import numpy as np
import pytest

from conftest import jit_compile, clean_test
from loki import Subroutine, OFP, OMNI, FP, FortranSDFGTransformation


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def load_module(path):
    path = Path(path)
    return importlib.import_module(path.stem)
    # with open(path, 'r') as f:
    #     code = f.read()
    # spec = importlib.util.spec_from_loader(path.stem, loader=None)
    # mod = importlib.util.module_from_spec(spec)
    # exec(code, mod.__dict__)
    # return mod


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_sdfg_routine_copy(here, frontend):

    fcode = """
subroutine routine_copy(n, x, y)
  ! A simple routine that copies the values of x to y
  use iso_fortran_env, only: real64
  implicit none
  real(kind=real64), intent(in) :: x(n)
  real(kind=real64), intent(out) :: y(n)
  integer, intent(in) :: n
  integer :: i

  do i=1,n
    y(i) = x(i)
  enddo
end subroutine routine_copy
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Test the reference solution
    filepath = here/('routine_copy_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_copy')

    n = 64
    x_ref = np.array(range(n), dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    x[:] = x_ref[:]
    y = np.zeros(n, dtype=np.float64)
    function(n=n, x=x, y=y)
    assert all(x_ref == y)

    # Create the SDFG
    trafo = FortranSDFGTransformation()
    routine.apply(trafo, path=here)

    mod = load_module(trafo.py_path)
    function = getattr(mod, '{}_py'.format(routine.name))
    sdfg = function.to_sdfg()
    assert sdfg.validate() is None

    # Compile the SDFG
    csdfg = sdfg.compile()
    assert csdfg

    # Run the SDFG
    x[:] = x_ref[:]
    csdfg(n=np.int32(n), x=x, y=y)
    assert all(x_ref == y)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_sdfg_routine_axpy_scalar(here, frontend):

    fcode = """
subroutine routine_axpy_scalar(a, x, y)
  ! A simple standard routine that computes x = a * x + y for
  ! scalar arguments
  use iso_fortran_env, only: real64
  implicit none
  real(kind=real64), intent(in) :: a, y
  real(kind=real64), intent(inout) :: x

  x = a * x + y
end subroutine routine_axpy_scalar
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    trafo = FortranSDFGTransformation()
    routine.apply(trafo, path=here)

    mod = load_module(trafo.py_path)
    function = getattr(mod, '{}_py'.format(routine.name))
    sdfg = function.to_sdfg()

    assert sdfg.validate() is None

