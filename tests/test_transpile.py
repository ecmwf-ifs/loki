from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, jit_compile_lib, clean_test
from loki import SourceFile, Subroutine, Module, OFP, OMNI, FP, FortranCTransformation
from loki.build import Builder, Obj, Lib


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(here):
    return Builder(source_dirs=here, build_dir=here/'build')


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_simple_loops(here, builder, frontend):
    """
    A simple test routine to test C transpilation of loops
    """

    fcode = """
subroutine transpile_simple_loops(n, m, scalar, vector, tensor)
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
end subroutine transpile_simple_loops
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('transpile_simple_loops_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='transpile_simple_loops')

    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    function(n, m, scalar, vector, tensor)

    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transpile_simple_loops_fc_mod.transpile_simple_loops_fc

    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    fc_function(n, m, scalar, vector, tensor)

    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])

    builder.clean()
    clean_test(filepath)
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_arguments(here, builder, frontend):
    """
    A test the correct exchange of arguments with varying intents
    """

    fcode = """
subroutine transpile_arguments(n, array, array_io, a, b, c, a_io, b_io, c_io)
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
end subroutine transpile_arguments
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
    filepath = here/('transpile_arguments_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='transpile_arguments')
    a, b, c = function(n, array, array_io, a_io, b_io, c_io)

    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 8 and np.isclose(b, 3.2) and np.isclose(c, 4.1)

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transpile_arguments_fc_mod.transpile_arguments_fc

    array = np.zeros(shape=(n,), order='F')
    array_io = np.zeros(shape=(n,), order='F') + 3.
    a_io = np.zeros(shape=(1,), order='F', dtype=np.int32) + 1
    b_io = np.zeros(shape=(1,), order='F', dtype=np.float32) + 2.
    c_io = np.zeros(shape=(1,), order='F', dtype=np.float64) + 3.
    a, b, c = fc_function(n, array, array_io, a_io, b_io, c_io)

    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 8 and np.isclose(b, 3.2) and np.isclose(c, 4.1)

    builder.clean()
    clean_test(filepath)
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_derived_type(here, builder, frontend):
    """
    Tests handling and type-conversion of various argument types
    """

    fcode_type = """
module transpile_type_mod
  use iso_fortran_env, only: real32, real64
  implicit none

  type my_struct
     integer :: a
     real(kind=real32) :: b
     real(kind=real64) :: c
  end type my_struct
end module transpile_type_mod
"""

    fcode_routine = """
subroutine transpile_derived_type(a_struct)
  use transpile_type_mod, only: my_struct
  implicit none
  type(my_struct), intent(inout) :: a_struct

  a_struct%a = a_struct%a + 4
  a_struct%b = a_struct%b + 5.
  a_struct%c = a_struct%c + 6.
end subroutine transpile_derived_type
"""
    builder.clean()

    module = Module.from_source(fcode_type, frontend=frontend)
    routine = Subroutine.from_source(fcode_routine, definitions=module, frontend=frontend)
    refname = 'ref_%s_%s' % (routine.name, frontend)
    reference = jit_compile_lib([module, routine], path=here, name=refname, builder=builder)

    # Test the reference solution
    a_struct = reference.transpile_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    reference.transpile_derived_type(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.

    # Translate the header module to expose parameters
    mod2c = FortranCTransformation()
    mod2c.apply(source=module, path=here, role='header')

    # Create transformation object and apply
    f2c = FortranCTransformation(header_modules=[module])
    f2c.apply(source=routine, path=here, role='kernel')

    # Build and wrap the cross-compiled library
    sources = [module, f2c.wrapperpath, f2c.c_path]
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib(sources=sources, path=here, name=libname, builder=builder)

    a_struct = c_kernel.transpile_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    function = c_kernel.transpile_derived_type_fc_mod.transpile_derived_type_fc
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.

    builder.clean()
    mod2c.wrapperpath.unlink()
    mod2c.c_path.unlink()
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()
    (here/'{}.f90'.format(module.name)).unlink()


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_associates(here, builder, frontend):
    """
    Tests C-transpilation of associate statements
    """

    fcode_type = """
module transpile_type_mod
  use iso_fortran_env, only: real32, real64
  implicit none

  type my_struct
     integer :: a
     real(kind=real32) :: b
     real(kind=real64) :: c
  end type my_struct
end module transpile_type_mod
"""

    fcode_routine = """
subroutine transpile_associates(a_struct)
  use transpile_type_mod, only: my_struct
  implicit none
  type(my_struct), intent(inout) :: a_struct

  associate(a_struct_a=>a_struct%a, a_struct_b=>a_struct%b,&
   & a_struct_c=>a_struct%c)
  a_struct%a = a_struct_a + 4.
  a_struct_b = a_struct%b + 5.
  a_struct_c = a_struct_a + a_struct%b + a_struct_c
  end associate
end subroutine transpile_associates
"""
    builder.clean()

    module = Module.from_source(fcode_type, frontend=frontend)
    routine = Subroutine.from_source(fcode_routine, definitions=module, frontend=frontend)
    refname = 'ref_%s_%s' % (routine.name, frontend)
    reference = jit_compile_lib([module, routine], path=here, name=refname, builder=builder)

    # Test the reference solution
    a_struct = reference.transpile_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    reference.transpile_associates(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 24.

    # Translate the header module to expose parameters
    mod2c = FortranCTransformation()
    mod2c.apply(source=module, path=here, role='header')

    # Create transformation object and apply
    f2c = FortranCTransformation(header_modules=[module])
    f2c.apply(source=routine, path=here, role='kernel')

    # Build and wrap the cross-compiled library
    sources = [module, f2c.wrapperpath, f2c.c_path]
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib(sources=sources, path=here, name=libname, builder=builder)

    a_struct = c_kernel.transpile_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    function = c_kernel.transpile_associates_fc_mod.transpile_associates_fc
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 24.

    builder.clean()
    mod2c.wrapperpath.unlink()
    mod2c.c_path.unlink()
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()
    (here/'{}.f90'.format(module.name)).unlink()


@pytest.mark.skip(reason='More thought needed on how to test structs-of-arrays')
def test_transpile_derived_type_array(here, builder, frontend):
    """
    Tests handling of multi-dimensional arrays and pointers.

    a_struct%scalar = 3.
    a_struct%vector(i) = a_struct%scalar + 2.
    a_struct%matrix(j,i) = a_struct%vector(i) + 1.

! subroutine transpile_derived_type_array(a_struct)
!   use transpile_type, only: array_struct
!   implicit none
!      ! real(kind=real64) :: vector(:)
!      ! real(kind=real64) :: matrix(:,:)
!   type(array_struct), intent(inout) :: a_struct
!   integer :: i, j

!   a_struct%scalar = 3.
!   do i=1, 3
!     a_struct%vector(i) = a_struct%scalar + 2.
!   end do
!   do i=1, 3
!     do j=1, 3
!       a_struct%matrix(j,i) = a_struct%vector(i) + 1.
!     end do
!   end do

! end subroutine transpile_derived_type_array
    """

@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_module_variables(here, builder, frontend):
    """
    Tests the use of imported module variables (via getter routines in C)
    """

    fcode_type = """
module transpile_type_mod
  use iso_fortran_env, only: real32, real64
  implicit none

  save

  integer :: PARAM1
  real(kind=real32) :: param2
  real(kind=real64) :: param3
end module transpile_type_mod
"""

    fcode_routine = """
subroutine transpile_module_variables(a, b, c)
  use iso_fortran_env, only: real32, real64
  use transpile_type_mod, only: PARAM1, param2, param3

  integer, intent(out) :: a
  real(kind=real32), intent(out) :: b
  real(kind=real64), intent(out) :: c

  a = 1 + PARAM1  ! Ensure downcasing is done right
  b = 1. + param2
  c = 1. + param3
end subroutine transpile_module_variables
"""

    module = Module.from_source(fcode_type, frontend=frontend)
    routine = Subroutine.from_source(fcode_routine, definitions=module, frontend=frontend)
    refname = 'ref_%s_%s' % (routine.name, frontend)
    reference = jit_compile_lib([module, routine], path=here, name=refname, builder=builder)

    reference.transpile_type_mod.param1 = 2
    reference.transpile_type_mod.param2 = 4.
    reference.transpile_type_mod.param3 = 3.
    a, b, c = reference.transpile_module_variables()
    assert a == 3 and b == 5. and c == 4.

    # Translate the header module to expose parameters
    mod2c = FortranCTransformation()
    mod2c.apply(source=module, path=here, role='header')

    # Create transformation object and apply
    f2c = FortranCTransformation(header_modules=[module])
    f2c.apply(source=routine, path=here, role='kernel')

    # Build and wrap the cross-compiled library
    sources = [module, mod2c.wrapperpath, f2c.wrapperpath, f2c.c_path]
    wrap = [here/'transpile_type_mod.f90', f2c.wrapperpath.name]
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib(sources=sources, wrap=wrap, path=here, name=libname, builder=builder)

    c_kernel.transpile_type_mod.param1 = 2
    c_kernel.transpile_type_mod.param2 = 4.
    c_kernel.transpile_type_mod.param3 = 3.
    a, b, c = c_kernel.transpile_module_variables_fc_mod.transpile_module_variables_fc()
    assert a == 3 and b == 5. and c == 4.

    builder.clean()
    mod2c.wrapperpath.unlink()
    mod2c.c_path.unlink()
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()
    (here/'{}.f90'.format(module.name)).unlink()


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_vectorization(here, builder, frontend):
    """
    Tests vector-notation conversion and local multi-dimensional arrays.
    """

    fcode = """
subroutine transpile_vectorization(n, m, scalar, v1, v2)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n, m
  real(kind=real64), intent(inout) :: scalar
  real(kind=real64), intent(inout) :: v1(n), v2(n)

  real(kind=real64) :: matrix(n, m)

  integer :: i

  v1(:) = scalar + 1.0
  matrix(:, :) = scalar + 2.
  v2(:) = matrix(:, 2)
  v2(1) = 1.
end subroutine transpile_vectorization
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('transpile_vectorization_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='transpile_vectorization')

    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')
    function(n, m, scalar, v1, v2)

    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transpile_vectorization_fc_mod.transpile_vectorization_fc

    # Test the trnapiled C kernel
    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')
    fc_function(n, m, scalar, v1, v2)

    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)

    builder.clean()
    clean_test(filepath)
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_intrinsics(here, builder, frontend):
    """
    A simple test routine to test supported intrinsic functions
    """

    fcode = """
subroutine transpile_intrinsics(v1, v2, v3, v4, vmin, vmax, vabs, vmin_nested, vmax_nested)
  ! Test supported intrinsic functions
  use iso_fortran_env, only: real64
  real(kind=real64), intent(in) :: v1, v2, v3, v4
  real(kind=real64), intent(out) :: vmin, vmax, vabs, vmin_nested, vmax_nested

  vmin = min(v1, v2)
  vmax = max(v1, v2)
  vabs = abs(v1 - v2)
  vmin_nested = min(min(v1, v2), min(v3, v4))
  vmax_nested = max(max(v1, v2), max(v3, v4))
end subroutine transpile_intrinsics
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('transpile_intrinsics_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='transpile_intrinsics')

    # Test the reference solution
    v1, v2, v3, v4 = 2., 4., 1., 5.
    vmin, vmax, vabs, vmin_nested, vmax_nested = function(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transpile_intrinsics_fc_mod.transpile_intrinsics_fc

    vmin, vmax, vabs, vmin_nested, vmax_nested = fc_function(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.

    builder.clean()
    clean_test(filepath)
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_loop_indices(here, builder, frontend):
    """
    Test to ensure loop indexing translates correctly
    """

    fcode = """
subroutine transpile_loop_indices(n, idx, mask1, mask2, mask3)
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
end subroutine transpile_loop_indices
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('transpile_loop_indices_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='transpile_loop_indices')

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

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transpile_loop_indices_fc_mod.transpile_loop_indices_fc

    mask1 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    mask3 = np.zeros(shape=(n,), order='F', dtype=np.float64)
    fc_function(n=n, idx=fidx, mask1=mask1, mask2=mask2, mask3=mask3)
    assert np.all(mask1[:cidx-1] == 1)
    assert mask1[cidx] == 2
    assert np.all(mask1[cidx+1:] == 0)
    assert np.all(mask2 == np.arange(n, dtype=np.int32) + 1)
    assert np.all(mask3[:-1] == 0.)
    assert mask3[-1] == 3.

    builder.clean()
    clean_test(filepath)
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()


@pytest.mark.parametrize('frontend', [OMNI, OFP, FP])
def test_transpile_logical_statements(here, builder, frontend):
    """
    A simple test routine to test logical statements
    """

    fcode = """
subroutine transpile_logical_statements(v1, v2, v_xor, v_xnor, v_nand, v_neqv, v_val)
  logical, intent(in) :: v1, v2
  logical, intent(out) :: v_xor, v_nand, v_xnor, v_neqv, v_val(2)

  v_xor = (v1 .and. .not. v2) .or. (.not. v1 .and. v2)
  v_xnor = v1 .eqv. v2
  v_nand = .not. (v1 .and. v2)
  v_neqv = v1 .neqv. v2
  v_val(1) = .true.
  v_val(2) = .false.

end subroutine transpile_logical_statements
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('transpile_logical_statements_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='transpile_logical_statements')

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

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transpile_logical_statements_fc_mod.transpile_logical_statements_fc

    for v1 in range(2):
        for v2 in range(2):
            v_val = np.zeros(shape=(2,), order='F', dtype=np.int32)
            v_xor, v_xnor, v_nand, v_neqv = fc_function(v1, v2, v_val)
            assert v_xor == (v1 and not v2) or (not v1 and v2)
            assert v_xnor == (v1 and v2) or not (v1 or v2)
            assert v_nand == (not (v1 and v2))
            assert v_neqv == ((not (v1 and v2)) and (v1 or v2))
            assert v_val[0] and not v_val[1]

    builder.clean()
    clean_test(filepath)
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transpile_multibody_conditionals(here, builder, frontend):
    """
    Test correct transformation of multi-body conditionals.
    """
    fcode = """
subroutine transpile_multibody_conditionals(in1, out1, out2)
  integer, intent(in) :: in1
  integer, intent(out) :: out1, out2

  if (in1 > 5) then
    out1 = 5
  else
    out1 = 1
  end if

  if (in1 < 0) then
    out2 = 0
  else if (in1 > 5) then
    out2 = 6
    out2 = out2 - 1
  else if (3 < in1 .and. in1 <= 5) then
    out2 = 4
  else
    out2 = in1
  end if
end subroutine transpile_multibody_conditionals
"""
    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('transpile_multibody_conditionals_%s.f90' % frontend)
    function = jit_compile(routine, filepath=filepath, objname='transpile_multibody_conditionals')

    out1, out2 = function(5)
    assert out1 == 1 and out2 == 4

    out1, out2 = function(2)
    assert out1 == 1 and out2 == 2

    out1, out2 = function(-1)
    assert out1 == 1 and out2 == 0

    out1, out2 = function(10)
    assert out1 == 5 and out2 == 5

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_function = c_kernel.transpile_multibody_conditionals_fc_mod.transpile_multibody_conditionals_fc

    out1, out2 = fc_function(5)
    assert out1 == 1 and out2 == 4

    out1, out2 = fc_function(2)
    assert out1 == 1 and out2 == 2

    out1, out2 = fc_function(-1)
    assert out1 == 1 and out2 == 0

    out1, out2 = fc_function(10)
    assert out1 == 5 and out2 == 5
    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transpile_inline_elemental_functions(here, builder, frontend):
    """
    Test correct transformation of multi-body conditionals.
    """
    fcode_module = """
module multiply_mod
  use iso_fortran_env, only: real64
  implicit none
contains

  elemental function multiply(a, b)
    real(kind=real64) :: multiply
    real(kind=real64), intent(in) :: a, b

    multiply = a * b
  end function multiply
end module multiply_mod
"""

    fcode = """
subroutine transpile_inline_elemental_functions(v1, v2, v3)
  use iso_fortran_env, only: real64
  use multiply_mod, only: multiply
  real(kind=real64), intent(in) :: v1
  real(kind=real64), intent(out) :: v2, v3

  v2 = multiply(v1, 6._real64)
  v3 = 600. + multiply(6._real64, 11._real64)
end subroutine transpile_inline_elemental_functions
"""
    # Generate reference code, compile run and verify
    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    refname = 'ref_%s_%s' % (routine.name, frontend)
    reference = jit_compile_lib([module, routine], path=here, name=refname, builder=builder)

    v2, v3 = reference.transpile_inline_elemental_functions(11.)
    assert v2 == 66.
    assert v3 == 666.

    (here/'{}.f90'.format(routine.name)).unlink()

    # Now transpile with supplied elementals but without module
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)

    f2c = FortranCTransformation(inline_elementals=True)
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_mod = c_kernel.transpile_inline_elemental_functions_fc_mod

    v2, v3 = fc_mod.transpile_inline_elemental_functions_fc(11.)
    assert v2 == 66.
    assert v3 == 666.

    builder.clean()
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()
    (here/'{}.f90'.format(module.name)).unlink()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transpile_inline_elementals_recursive(here, builder, frontend):
    """
    Test correct transformation of multi-body conditionals.
    """
    fcode_module = """
module multiply_plus_one_mod
  use iso_fortran_env, only: real64
  implicit none
contains

  elemental function multiply(a, b)
    real(kind=real64) :: multiply
    real(kind=real64), intent(in) :: a, b

    multiply = a * b
  end function multiply

  elemental function plus_one(a)
    real(kind=real64) :: plus_one
    real(kind=real64), intent(in) :: a

    plus_one = a + 1._real64
  end function plus_one

  elemental function multiply_plus_one(a, b)
    real(kind=real64) :: multiply_plus_one
    real(kind=real64), intent(in) :: a, b

    ! TODO: Add temporary variables...
    multiply_plus_one = multiply(plus_one(a), b)
  end function multiply_plus_one
end module multiply_plus_one_mod
"""

    fcode = """
subroutine transpile_inline_elementals_recursive(v1, v2, v3)
  use iso_fortran_env, only: real64
  use multiply_plus_one_mod, only: multiply_plus_one
  real(kind=real64), intent(in) :: v1
  real(kind=real64), intent(out) :: v2, v3

  v2 = multiply_plus_one(v1, 6._real64)
  v3 = 600. + multiply_plus_one(5._real64, 11._real64)
end subroutine transpile_inline_elementals_recursive
"""
    # Generate reference code, compile run and verify
    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    refname = 'ref_%s_%s' % (routine.name, frontend)
    reference = jit_compile_lib([module, routine], path=here, name=refname, builder=builder)

    v2, v3 = reference.transpile_inline_elementals_recursive(10.)
    assert v2 == 66.
    assert v3 == 666.

    (here/'{}.f90'.format(routine.name)).unlink()

    # Now transpile with supplied elementals but without module
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)

    f2c = FortranCTransformation(inline_elementals=True)
    f2c.apply(source=routine, path=here)
    libname = 'fc_{}_{}'.format(routine.name, frontend)
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=here, name=libname, builder=builder)
    fc_mod = c_kernel.transpile_inline_elementals_recursive_fc_mod

    v2, v3 = fc_mod.transpile_inline_elementals_recursive_fc(10.)
    assert v2 == 66.
    assert v3 == 666.

    builder.clean()
    f2c.wrapperpath.unlink()
    f2c.c_path.unlink()
    (here/'{}.f90'.format(module.name)).unlink()
