import importlib
import itertools
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


def create_sdfg(routine, here):
    trafo = FortranSDFGTransformation()
    routine.apply(trafo, path=here)

    mod = load_module(trafo.py_path)
    function = getattr(mod, '{}_py'.format(routine.name))
    return function.to_sdfg()


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

    # Create and compile the SDFG
    sdfg = create_sdfg(routine, here)
    assert sdfg.validate() is None

    csdfg = sdfg.compile()
    assert csdfg

    # Run the SDFG
    x[:] = x_ref[:]
    csdfg(n=np.int32(n), x=x, y=y)
    assert all(x_ref == y)

    clean_test(filepath)


@pytest.mark.xfail(reason='Scalar inout arguments do not work in dace')
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

    # Test the reference solution
    filepath = here/('sdfg_routine_axpy_scalar_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_axpy_scalar')

    a = np.float64(23)
    x = np.float64(42)
    x_out = np.array([x], dtype=np.float64)
    y = np.float64(5)
    function(a=a, x=x_out, y=y)
    assert x_out == a * x + y

    # Create and compile the SDFG
    sdfg = create_sdfg(routine, here)
    assert sdfg.validate() is None

    csdfg = sdfg.compile()
    assert csdfg

    # Run the SDFG
    x_out = np.array([x], dtype=np.float64)
    csdfg(a=a, x=x_out, y=y)
    assert x_out == a * x + y

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_sdfg_routine_copy_stream(here, frontend):

    fcode = """
subroutine routine_copy_stream(length, alpha, vector_in, vector_out)
  implicit none
  ! A simple standard looking routine to test argument declarations
  ! and generator toolchain
  integer, intent(in) :: length, alpha(1), vector_in(length)
  integer, intent(out) :: vector_out(length)
  integer :: i

  !$loki dataflow
  do i=1, length
    vector_out(i) = vector_in(i) + alpha(1)
  end do
end subroutine routine_copy_stream
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # TODO: make alpha a true scalar, which doesn't seem to work with SDFG at the moment???

    # Test the reference solution
    filepath = here/('sdfg_routine_copy_stream_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_copy_stream')

    length = 32
    alpha = np.array([7], dtype=np.int32)
    vector_in = np.array(range(length), order='F', dtype=np.int32)
    vector_out = np.zeros(length, order='F', dtype=np.int32)
    function(length=length, alpha=alpha, vector_in=vector_in, vector_out=vector_out)
    assert np.all(vector_out == np.array(range(length)) + alpha)

    # Create and compile the SDFG
    sdfg = create_sdfg(routine, here)
    assert sdfg.validate() is None

    csdfg = sdfg.compile()
    assert csdfg

    # Run the SDFG
    vec_in = np.array(range(length), order='F', dtype=np.intc)
    vec_out = np.zeros(length, order='F', dtype=np.intc)
    csdfg(length=length, alpha=alpha, vector_in=vec_in, vector_out=vec_out)
    assert np.all(vec_out == np.array(range(length)) + alpha)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_sdfg_routine_fixed_loop(here, frontend):

    fcode = """
subroutine routine_fixed_loop(scalar, vector, vector_out, tensor, tensor_out)
  use iso_fortran_env, only: real64
  implicit none
  ! integer :: n=6, m=4
  real(kind=real64), intent(in) :: scalar(1)
  real(kind=real64), intent(in) :: tensor(6, 4), vector(6)
  real(kind=real64), intent(out) :: tensor_out(4, 6), vector_out(6)
  integer :: i, j

  ! For testing, the operation is:
  !$loki dataflow
  do j=1, 6
     vector_out(j) = vector(j) + tensor(j, 1) + 1.0
     !$loki dataflow
     do i=1, 4
        tensor_out(i, j) = tensor(j, i)
     end do
  end do
end subroutine routine_fixed_loop
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('sdfg_routine_fixed_loop_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_fixed_loop')

    # Test the reference solution
    n, m = 6, 4
    scalar = np.array([2.0], dtype=np.float64)
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.array([list(range(i, i+m)) for i in range(n)], order='F', dtype=np.float64)
    tensor_out = np.zeros(shape=(m, n), order='F')
    ref_vector = vector + np.array(list(range(n)), dtype=np.float64) + 1.
    ref_tensor = np.transpose(tensor)
    function(scalar=scalar, vector=vector, vector_out=vector, tensor=tensor, tensor_out=tensor_out)
    assert np.all(vector == ref_vector)
    assert np.all(tensor_out == ref_tensor)

    # Create and compile the SDFG
    sdfg = create_sdfg(routine, here)
    assert sdfg.validate() is None

    csdfg = sdfg.compile()
    assert csdfg

    # Test the transpiled kernel
    n, m = 6, 4
    scalar = np.array([2.0], dtype=np.float64)
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    tensor = np.array([list(range(i, i+m)) for i in range(n)], order='F', dtype=np.float64)
    tensor_out = np.zeros(shape=(m, n), order='F')
    csdfg(scalar=scalar, vector=vector, vector_out=vector, tensor=tensor, tensor_out=tensor_out)
    assert np.all(vector == ref_vector)
    assert np.all(tensor_out == ref_tensor)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_sdfg_routine_loop_carried_dependency(here, frontend):

    fcode = """
subroutine routine_loop_carried_dependency(vector)
  use iso_fortran_env, only: real64
  implicit none
  real(kind=real64), intent(inout) :: vector(32)
  integer :: i

  !$loki dataflow
  do i=2, 32
     vector(i) = vector(i) + vector(i-1)
  end do
end subroutine routine_loop_carried_dependency
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('sdfg_routine_loop_carried_dependency_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='routine_loop_carried_dependency')

    # Test the reference solution
    n = 32
    vector = np.zeros(shape=(n,), order='F') + 3.
    ref_vector = np.array(list(itertools.accumulate(vector)))
    function(vector=vector)
    assert np.all(vector == ref_vector)

    # Create and compile the SDFG
    sdfg = create_sdfg(routine, here)
    assert sdfg.validate() is None

    csdfg = sdfg.compile()
    assert csdfg

    # Test the transpiled kernel
    n = 32
    vector = np.zeros(shape=(n,), order='F') + 3.
    ref_vector = np.array(list(itertools.accumulate(vector)))
    csdfg(vector=vector)
    assert np.all(vector == ref_vector)
