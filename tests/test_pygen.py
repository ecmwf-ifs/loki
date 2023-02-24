# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys
from pathlib import Path
from importlib import import_module, reload, invalidate_caches
import pytest
import numpy as np

from conftest import available_frontends, jit_compile, clean_test
from loki import Subroutine, FortranPythonTransformation, pygen, OFP


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


def load_module(here, module):
    """
    A helper routine that loads the given module from the current path.
    """
    modpath = str(Path(here).absolute())
    if modpath not in sys.path:
        sys.path.insert(0, modpath)
    if module in sys.modules:
        reload(sys.modules[module])
        return sys.modules[module]

    # Trigger the actual module import
    try:
        return import_module(module)
    except ModuleNotFoundError:
        # If module caching interferes, try again with clean caches
        invalidate_caches()
        return import_module(module)


@pytest.mark.parametrize('frontend', available_frontends())
def test_pygen_simple_loops(here, frontend):
    """
    A simple test routine to test Python transpilation of loops
    """

    fcode = """
subroutine pygen_simple_loops(n, m, scalar, vector, tensor)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n, m
  real(kind=real64), intent(inout) :: scalar
  real(kind=real64), intent(inout) :: vector(n), tensor(n, m)

  integer :: i, j

  ! For testing, the operation is:
  do i=1, n
     vector(i) = vector(i) + tensor(i, 1) + 1.0
  end do

  do j=1, m
     do i=1, n
        tensor(i, j) = 10.* j + i
     end do
  end do
end subroutine pygen_simple_loops
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'pygen_simple_loops_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='pygen_simple_loops')

    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    function(n, m, scalar, vector, tensor)

    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])

    # Rename routine to avoid problems with module import caching
    routine.name = f'{routine.name}_{str(frontend)}'

    # Generate and test the transpiled Python kernel
    f2p = FortranPythonTransformation(suffix='_py')
    f2p.apply(source=routine, path=here)
    mod = load_module(here, f2p.mod_name)
    func = getattr(mod, f2p.mod_name)

    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    func(n, m, scalar, vector, tensor)

    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])

    clean_test(filepath)
    f2p.py_path.unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_pygen_arguments(here, frontend):
    """
    Test the correct exchange of arguments with varying intents
    """

    fcode = """
subroutine pygen_arguments(n, array, array_io, a, b, c, a_io, b_io, c_io)
  use iso_fortran_env, only: real32, real64
  implicit none

  integer, intent(in) :: n
  real(kind=real64), intent(inout) :: array(n)
  real(kind=real64), intent(out) :: array_io(n)

  integer, intent(out) :: a
  real(kind=real32), intent(out) :: b
  real(kind=real64), intent(out) :: c
  integer, intent(inout) :: a_io
  real(kind=real32), intent(inout) :: b_io
  real(kind=real64), intent(inout) :: c_io

  integer :: i

  do i=1, n
     array(i) = 3.
     array_io(i) = array_io(i) + 3.
  end do

  a = 2**3
  b = 3.2_real32
  c = 4.1_real64

  a_io = a_io + 2
  b_io = b_io + real(3.2, kind=real32)
  c_io = c_io + 4.1
end subroutine pygen_arguments
"""

    # Test the reference solution
    n = 3
    array = np.zeros(shape=(n,), order='F')
    array_io = np.zeros(shape=(n,), order='F') + 3.
    # To do scalar inout we allocate data in single-element arrays
    a_io = np.zeros(shape=(1,), order='F', dtype=np.int32) + 1
    b_io = np.zeros(shape=(1,), order='F', dtype=np.float32) + 2.
    c_io = np.zeros(shape=(1,), order='F', dtype=np.float64) + 3.

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'pygen_arguments_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='pygen_arguments')
    a, b, c = function(n, array, array_io, a_io, b_io, c_io)

    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 8 and np.isclose(b, 3.2) and np.isclose(c, 4.1)

    # Rename routine to avoid problems with module import caching
    routine.name = f'{routine.name}_{str(frontend)}'

    # Generate and test the transpiled Python kernel
    f2p = FortranPythonTransformation(suffix='_py')
    f2p.apply(source=routine, path=here)
    mod = load_module(here, f2p.mod_name)
    func = getattr(mod, f2p.mod_name)

    array = np.zeros(shape=(n,), order='F')
    array_io = np.zeros(shape=(n,), order='F') + 3.
    a_io = np.zeros(shape=(1,), order='F', dtype=np.int32) + 1
    b_io = np.zeros(shape=(1,), order='F', dtype=np.float32) + 2.
    c_io = np.zeros(shape=(1,), order='F', dtype=np.float64) + 3.
    a, b, c, a_io, b_io, c_io = func(n, array, array_io, a_io, b_io, c_io)

    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 8 and np.isclose(b, 3.2) and np.isclose(c, 4.1)

    clean_test(filepath)
    f2p.py_path.unlink()


# TODO: implement and test transpilation of derived types

# TODO: implement and test transpilation of associates

# TODO: implement and test transpilation of modules


@pytest.mark.parametrize('frontend', available_frontends())
def test_pygen_vectorization(here, frontend):
    """
    Tests vector-notation conversion and local multi-dimensional arrays.
    """

    fcode = """
subroutine pygen_vectorization(n, m, scalar, v1, v2)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n, m
  real(kind=real64), intent(inout) :: scalar
  real(kind=real64), intent(inout) :: v1(n), v2(n)

  real(kind=real64) :: matrix(n, m)

  integer :: i

  v1(:) = scalar + 1.0
  matrix(:, 1:m) = scalar + 2.
  v2(:n) = matrix(:, 2)
  v2(1) = 1.
end subroutine pygen_vectorization
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'pygen_vectorization_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='pygen_vectorization')

    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')
    function(n, m, scalar, v1, v2)

    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)

    # Rename routine to avoid problems with module import caching
    routine.name = f'{routine.name}_{str(frontend)}'

    # Generate and test the transpiled Python kernel
    f2p = FortranPythonTransformation(suffix='_py')
    f2p.apply(source=routine, path=here)
    mod = load_module(here, f2p.mod_name)
    func = getattr(mod, f2p.mod_name)

    # Test the transpiled Python kernel
    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')
    scalar = func(n, m, scalar, v1, v2)

    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)

    clean_test(filepath)
    f2p.py_path.unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_pygen_intrinsics(here, frontend):
    """
    A simple test routine to test supported intrinsic functions
    """

    fcode = """
subroutine pygen_intrinsics(v1, v2, v3, v4, vmin, vmax, vabs, vmin_nested, vmax_nested, vexp, vsqrt, vsign)
  ! Test supported intrinsic functions
  use iso_fortran_env, only: real64
  real(kind=real64), intent(in) :: v1, v2, v3, v4
  real(kind=real64), intent(out) :: vmin, vmax, vabs, vmin_nested, vmax_nested, vexp, vsqrt, vsign

  vmin = MIN(v1, v2)
  vmax = MAX(v1, v2)
  vabs = ABS(v1 - v2)
  vmin_nested = MIN(MIN(MAX(v1, -1._real64), v2), MIN(v3, v4))
  vmax_nested = MAX(MAX(v1, v2), MAX(v3, v4))
  vexp = EXP(v2)
  vsqrt = SQRT(v2)
  vsign = SIGN(v4, v1-v2)
end subroutine pygen_intrinsics
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'pygen_intrinsics_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='pygen_intrinsics')

    # Test the reference solution
    v1, v2, v3, v4 = 2., 4., 1., 5.
    vmin, vmax, vabs, vmin_nested, vmax_nested, vexp, vsqrt, vsign = function(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.
    assert vexp == np.exp(4.) and vsqrt == 2.
    assert vsign == -5.

    # Rename routine to avoid problems with module import caching
    routine.name = f'{routine.name}_{str(frontend)}'

    # Generate and test the transpiled Python kernel
    f2p = FortranPythonTransformation(suffix='_py')
    f2p.apply(source=routine, path=here)
    mod = load_module(here, f2p.mod_name)
    func = getattr(mod, f2p.mod_name)

    vmin, vmax, vabs, vmin_nested, vmax_nested, vexp, vsqrt, vsign = func(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.
    assert vexp == np.exp(4.) and vsqrt == 2.
    assert vsign == -5.

    clean_test(filepath)
    f2p.py_path.unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_pygen_loop_indices(here, frontend):
    """
    Test to ensure loop indexing translates correctly
    """

    fcode = """
subroutine pygen_loop_indices(n, idx, mask1, mask2, mask3)
  ! Test to ensure loop indexing translates correctly
  use iso_fortran_env, only: real64
  integer, intent(in) :: n, idx
  integer, intent(inout) :: mask1(n), mask2(n)
  real(kind=real64), intent(inout) :: mask3(n)

  integer :: i

  do i=1, n
     if (i < idx) then
        mask1(i) = 1
     end if

     if (i == idx) then
        mask1(i) = 2
     end if

     mask2(i) = i
  end do
  mask3(n) = 3.0
end subroutine pygen_loop_indices
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'pygen_loop_indices_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='pygen_loop_indices')

    # Test the reference solution
    n = 6
    cidx, fidx = 3, 4
    mask1 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask3 = np.zeros(shape=(n,), order='F', dtype=np.float64)

    function(n=n, idx=fidx, mask1=mask1, mask2=mask2, mask3=mask3)
    assert np.all(mask1[:cidx-1] == 1)
    assert mask1[cidx] == 2
    assert np.all(mask1[cidx+1:] == 0)
    assert np.all(mask2 == np.arange(n, dtype=np.int32) + 1)
    assert np.all(mask3[:-1] == 0.)
    assert mask3[-1] == 3.

    # Rename routine to avoid problems with module import caching
    routine.name = f'{routine.name}_{str(frontend)}'

    # Generate and test the transpiled Python kernel
    f2p = FortranPythonTransformation(suffix='_py')
    f2p.apply(source=routine, path=here)
    mod = load_module(here, f2p.mod_name)
    func = getattr(mod, f2p.mod_name)

    mask1 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask3 = np.zeros(shape=(n,), order='F', dtype=np.float64)
    func(n=n, idx=fidx, mask1=mask1, mask2=mask2, mask3=mask3)
    assert np.all(mask1[:cidx-1] == 1)
    assert mask1[cidx] == 2
    assert np.all(mask1[cidx+1:] == 0)
    assert np.all(mask2 == np.arange(n, dtype=np.int32) + 1)
    assert np.all(mask3[:-1] == 0.)
    assert mask3[-1] == 3.

    clean_test(filepath)
    f2p.py_path.unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_pygen_logical_statements(here, frontend):
    """
    A simple test routine to test logical statements
    """

    fcode = """
subroutine pygen_logical_statements(v1, v2, v_xor, v_xnor, v_nand, v_neqv, v_val)
  logical, intent(in) :: v1, v2
  logical, intent(out) :: v_xor, v_nand, v_xnor, v_neqv, v_val(2)

  v_xor = (v1 .and. .not. v2) .or. (.not. v1 .and. v2)
  v_xnor = v1 .eqv. v2
  v_nand = .not. (v1 .and. v2)
  v_neqv = v1 .neqv. v2
  v_val(1) = .true.
  v_val(2) = .false.

end subroutine pygen_logical_statements
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'pygen_logical_statements_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='pygen_logical_statements')

    # Test the reference solution
    for v1 in range(2):
        for v2 in range(2):
            v_val = np.zeros(shape=(2,), order='F', dtype=np.int32)
            v_xor, v_xnor, v_nand, v_neqv = function(v1, v2, v_val)
            assert v_xor == (v1 and not v2) or (not v1 and v2)
            assert v_xnor == (v1 and v2) or not (v1 or v2)
            assert v_nand == (not (v1 and v2))
            assert v_neqv == ((not (v1 and v2)) and (v1 or v2))
            assert v_val[0] and not v_val[1]

    # Rename routine to avoid problems with module import caching
    routine.name = f'{routine.name}_{str(frontend)}'

    # Generate and test the transpiled Python kernel
    f2p = FortranPythonTransformation(suffix='_py')
    f2p.apply(source=routine, path=here)
    mod = load_module(here, f2p.mod_name)
    func = getattr(mod, f2p.mod_name)

    for v1 in range(2):
        for v2 in range(2):
            v_val = np.zeros(shape=(2,), order='F', dtype=np.int32)
            v_xor, v_xnor, v_nand, v_neqv = func(v1, v2, v_val)
            assert v_xor == (v1 and not v2) or (not v1 and v2)
            assert v_xnor == (v1 and v2) or not (v1 or v2)
            assert v_nand == (not (v1 and v2))
            assert v_neqv == ((not (v1 and v2)) and (v1 or v2))
            assert v_val[0] and not v_val[1]

    clean_test(filepath)
    f2p.py_path.unlink()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OFP, 'OFP cannot handle stmt functions')]))
def test_pygen_downcasing(here, frontend):
    """
    A simple test routine to test the conversion to lower case.
    """

    fcode = """
subroutine pygen_downcasing(n, ScalaR, VectOr)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: N
  real(kind=real64), intent(inout) :: scalar
  real(kind=real64), intent(inout) :: vector(n)

  integer :: i
  real(kind=real64) :: a, tmp

  real(kind=real64) :: sTmT_F
  sTmT_F(a) = a + 2.

  do i=1, n
     tmp = stmt_F(scalar)
     veCtor(i) = vecTor(i) + tmp
  end do

end subroutine pygen_downcasing
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/(f'pygen_downcasing_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='pygen_downcasing')

    n = 3
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 2.
    function(n, scalar, vector)
    assert np.all(vector == 6.)

    # Rename routine to avoid problems with module import caching
    routine.name = f'{routine.name}_{str(frontend)}'

    # Generate and test the transpiled Python kernel
    f2p = FortranPythonTransformation(suffix='_py')
    f2p.apply(source=routine, path=here)
    mod = load_module(here, f2p.mod_name)
    func = getattr(mod, f2p.mod_name)

    assert pygen(routine).islower()

    n = 3
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 2.
    func(n, scalar, vector)
    assert np.all(vector == 6.)
