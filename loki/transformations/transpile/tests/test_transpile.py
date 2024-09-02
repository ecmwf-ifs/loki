# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
# from shutil import rmtree
import pytest
import numpy as np

from loki import Subroutine, Module, cgen, cppgen, cudagen, FindNodes, Dimension, Scheduler, read_file
from loki.build import jit_compile, jit_compile_lib, clean_test, Builder, Obj
import loki.expression.symbols as sym
from loki.frontend import available_frontends, OFP
from loki import ir, fgen

from loki.transformations.array_indexing import normalize_range_indexing
from loki.transformations.transpile import FortranCTransformation
from loki.transformations.single_column import SCCLowLevelHoist, SCCLowLevelParametrise

@pytest.fixture(scope='function', name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()

# @pytest.mark.parametrize('case_sensitive', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
# @pytest.mark.parametrize('language', ('c', 'cuda'))
def test_transpile_redo_inline_call(tmp_path, frontend): # , case_sensitive, language):
    """
    A simple test for testing lowering the case and case-sensitivity
    for specific symbols.
    """

    fcode = """
subroutine transpile_redo_inline_call(a, x)
    integer, intent(in) :: a
    real, intent(in) :: x
    if (mod(a, 2) /= 0) then

    endif

    if (mod(x, 2.0) /= 0.0 then

    endif
    
end subroutine transpile_redo_inline_call
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    language = 'cuda'
    f2c = FortranCTransformation(language=language, use_c_ptr=language=='cuda')
    f2c.apply(source=routine, path=tmp_path)
    ccode = f2c.c_path.read_text().replace(' ', '').replace('\n', ' ').replace('\r', '').replace('\t', '')
    print(f"{ccode}")

@pytest.mark.parametrize('case_sensitive', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('language', ('c', 'cuda'))
def test_transpile_case_sensitivity(tmp_path, frontend, case_sensitive, language):
    """
    A simple test for testing lowering the case and case-sensitivity
    for specific symbols.
    """

    fcode = """
subroutine transpile_case_sensitivity(a)
    integer, intent(in) :: a

end subroutine transpile_case_sensitivity
"""
    def convert_case(_str, case_sensitive):
        return _str.lower() if not case_sensitive else _str

    routine = Subroutine.from_source(fcode, frontend=frontend)

    var_thread_idx = sym.Variable(name="threadIdx", case_sensitive=case_sensitive)
    var_x = sym.Variable(name="x", parent=var_thread_idx, case_sensitive=case_sensitive)
    assignment = ir.Assignment(lhs=routine.variable_map['a'], rhs=var_x)
    routine.arguments=routine.arguments + (routine.arguments[0].clone(name='sOmE_vAr', case_sensitive=case_sensitive),
            sym.Variable(name="oTher_VaR", case_sensitive=case_sensitive, type=routine.arguments[0].type.clone()))

    call = ir.CallStatement(sym.Variable(name='somE_cALl', case_sensitive=case_sensitive),
            arguments=(routine.variable_map['a'],))
    inline_call = sym.InlineCall(function=sym.Variable(name='somE_InlINeCaLl', case_sensitive=case_sensitive),
            parameters=(sym.IntLiteral(1),))
    inline_call_assignment = ir.Assignment(lhs=routine.variable_map['a'], rhs=inline_call)
    routine.body = (routine.body, assignment, call, inline_call_assignment)

    f2c = FortranCTransformation(language=language, use_c_ptr=language=='cuda')
    f2c.apply(source=routine, path=tmp_path)
    ccode = f2c.c_path.read_text().replace(' ', '').replace('\n', ' ').replace('\r', '').replace('\t', '')
    assert convert_case('transpile_case_sensitivity_c(inta,intsOmE_vAr,intoTher_VaR)', case_sensitive) in ccode
    assert convert_case('a=threadIdx%x;', case_sensitive) in ccode
    assert convert_case('somE_cALl(a);', case_sensitive) in ccode
    assert convert_case('a=somE_InlINeCaLl(1);', case_sensitive) in ccode


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_simple_loops(tmp_path, builder, frontend, use_c_ptr):
    """
    A simple test routine to test C transpilation of loops
    """

    fcode = """
subroutine simple_loops(n, m, scalar, vector, tensor)
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
end subroutine simple_loops
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    normalize_range_indexing(routine) # Fix OMNI nonsense
    filepath = tmp_path/(f'simple_loops{"_c_ptr" if use_c_ptr else ""}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='simple_loops')

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
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.simple_loops_fc_mod.simple_loops_fc

    # check the generated F2C wrapper
    with open(f2c.wrapperpath, 'r') as f2c_f:
        f2c_str = f2c_f.read().upper().replace(' ', '')
        if use_c_ptr:
            assert f2c_str.count('TARGET') == 2
            assert f2c_str.count('C_LOC') == 3
            assert 'VECTOR(:)' in f2c_str
            assert 'TENSOR(:,:)' in f2c_str
        else:
            assert f2c_str.count('TARGET') == 0
            assert f2c_str.count('C_LOC') == 0
            assert 'VECTOR(N)' in f2c_str
            assert 'TENSOR(N,M)' in f2c_str

    n, m = 3, 4
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F') + 3.
    tensor = np.zeros(shape=(n, m), order='F') + 4.
    fc_function(n, m, scalar, vector, tensor)

    assert np.all(vector == 8.)
    assert np.all(tensor == [[11., 21., 31., 41.],
                             [12., 22., 32., 42.],
                             [13., 23., 33., 43.]])


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_arguments(tmp_path, builder, frontend, use_c_ptr):
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
    normalize_range_indexing(routine) # Fix OMNI nonsense
    filepath = tmp_path/(f'transpile_arguments{"_c_ptr" if use_c_ptr else ""}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='transpile_arguments')
    a, b, c = function(n, array, array_io, a_io, b_io, c_io)

    assert np.all(array == 3.) and array.size == n
    assert np.all(array_io == 6.)
    assert a_io[0] == 3. and np.isclose(b_io[0], 5.2) and np.isclose(c_io[0], 7.1)
    assert a == 8 and np.isclose(b, 3.2) and np.isclose(c, 4.1)

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.transpile_arguments_fc_mod.transpile_arguments_fc

    # check the generated F2C wrapper
    with open(f2c.wrapperpath, 'r') as f2c_f:
        f2c_str = f2c_f.read().upper().replace(' ', '')
        if use_c_ptr:
            assert f2c_str.count('TARGET') == 2
            assert f2c_str.count('C_LOC') == 3
            assert 'ARRAY(:)' in f2c_str
            assert 'ARRAY_IO(:)' in f2c_str
        else:
            assert f2c_str.count('TARGET') == 0
            assert f2c_str.count('C_LOC') == 0
            assert 'ARRAY(N)' in f2c_str
            assert 'ARRAY_IO(N)' in f2c_str

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


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_derived_type(tmp_path, builder, frontend, use_c_ptr):
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
subroutine transp_der_type(a_struct)
  use transpile_type_mod, only: my_struct
  implicit none
  type(my_struct), intent(inout) :: a_struct

  a_struct%a = a_struct%a + 4
  a_struct%b = a_struct%b + 5.
  a_struct%c = a_struct%c + 6.
end subroutine transp_der_type
"""
    builder.clean()

    module = Module.from_source(fcode_type, frontend=frontend)
    routine = Subroutine.from_source(fcode_routine, definitions=module, frontend=frontend)
    refname = f'ref_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)

    # Test the reference solution
    a_struct = reference.transpile_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    reference.transp_der_type(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.

    # Translate the header module to expose parameters
    mod2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    mod2c.apply(source=module, path=tmp_path, role='header')

    # Create transformation object and apply
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path, role='kernel')

    # Build and wrap the cross-compiled library
    sources = [module, f2c.wrapperpath, f2c.c_path]
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib(sources=sources, path=tmp_path, name=libname, builder=builder)

    a_struct = c_kernel.transpile_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    function = c_kernel.transp_der_type_fc_mod.transp_der_type_fc
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 12.


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_associates(tmp_path, builder, frontend, use_c_ptr):
    """
    Tests C-transpilation of associate statements
    """

    fcode_type = """
module assoc_type_mod
  use iso_fortran_env, only: real32, real64
  implicit none

  type my_struct
     integer :: a
     real(kind=real32) :: b
     real(kind=real64) :: c
  end type my_struct
end module assoc_type_mod
"""

    fcode_routine = """
subroutine transp_assoc(a_struct)
  use assoc_type_mod, only: my_struct
  implicit none
  type(my_struct), intent(inout) :: a_struct

  associate(a_struct_a=>a_struct%a, a_struct_b=>a_struct%b,&
   & a_struct_c=>a_struct%c)
  a_struct%a = a_struct_a + 4.
  a_struct_b = a_struct%b + 5.
  a_struct_c = a_struct_a + a_struct%b + a_struct_c
  end associate
end subroutine transp_assoc
"""

    module = Module.from_source(fcode_type, frontend=frontend)
    routine = Subroutine.from_source(fcode_routine, definitions=module, frontend=frontend)
    refname = f'ref_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)

    # Test the reference solution
    a_struct = reference.assoc_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    reference.transp_assoc(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 24.

    # Translate the header module to expose parameters
    mod2c = FortranCTransformation()
    mod2c.apply(source=module, path=tmp_path, role='header')

    # Create transformation object and apply
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path, role='kernel')

    # Build and wrap the cross-compiled library
    sources = [module, f2c.wrapperpath, f2c.c_path]
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib(sources=sources, path=tmp_path, name=libname, builder=builder)

    a_struct = c_kernel.assoc_type_mod.my_struct()
    a_struct.a = 4
    a_struct.b = 5.
    a_struct.c = 6.
    function = c_kernel.transp_assoc_fc_mod.transp_assoc_fc
    function(a_struct)
    assert a_struct.a == 8
    assert a_struct.b == 10.
    assert a_struct.c == 24.


@pytest.mark.skip(reason='More thought needed on how to test structs-of-arrays')
def test_transpile_derived_type_array():
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


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_module_variables(tmp_path, builder, frontend, use_c_ptr):
    """
    Tests the use of imported module variables (via getter routines in C)
    """

    fcode_type = """
module mod_var_type_mod
  use iso_fortran_env, only: real32, real64
  implicit none

  save

  integer :: PARAM1
  real(kind=real32) :: param2
  real(kind=real64) :: param3
end module mod_var_type_mod
"""

    fcode_routine = """
subroutine transp_mod_var(a, b, c)
  use iso_fortran_env, only: real32, real64
  use mod_var_type_mod, only: PARAM1, param2, param3

  integer, intent(out) :: a
  real(kind=real32), intent(out) :: b
  real(kind=real64), intent(out) :: c

  a = 1 + PARAM1  ! Ensure downcasing is done right
  b = 1. + param2
  c = 1. + param3
end subroutine transp_mod_var
"""

    module = Module.from_source(fcode_type, frontend=frontend)
    routine = Subroutine.from_source(fcode_routine, definitions=module, frontend=frontend)
    refname = f'ref_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)

    reference.mod_var_type_mod.param1 = 2
    reference.mod_var_type_mod.param2 = 4.
    reference.mod_var_type_mod.param3 = 3.
    a, b, c = reference.transp_mod_var()
    assert a == 3 and b == 5. and c == 4.

    # Translate the header module to expose parameters
    mod2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    mod2c.apply(source=module, path=tmp_path, role='header')

    # Create transformation object and apply
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path, role='kernel')

    # Build and wrap the cross-compiled library
    sources = [module, mod2c.wrapperpath, f2c.wrapperpath, f2c.c_path]
    wrap = [tmp_path/'mod_var_type_mod.f90', f2c.wrapperpath.name]
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib(sources=sources, wrap=wrap, path=tmp_path, name=libname, builder=builder)

    c_kernel.mod_var_type_mod.param1 = 2
    c_kernel.mod_var_type_mod.param2 = 4.
    c_kernel.mod_var_type_mod.param3 = 3.
    a, b, c = c_kernel.transp_mod_var_fc_mod.transp_mod_var_fc()
    assert a == 3 and b == 5. and c == 4.


@pytest.mark.parametrize('use_c_ptr', (False,)) # False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_vectorization(tmp_path, builder, frontend, use_c_ptr):
    """
    Tests vector-notation conversion and local multi-dimensional arrays.
    """

    fcode = """
subroutine transp_vect(n, m, scalar, v1, v2)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n, m
  real(kind=real64), intent(inout) :: scalar
  real(kind=real64), intent(inout) :: v1(n), v2(n)

  real(kind=real64) :: matrix(n, m)
  integer :: test1(n)
  integer :: test2(0:10)

  integer :: i

  do i=1,10
  test1(:) = 0
  test2(:) = 0 
  end do

  v1(:) = scalar + 1.0
  matrix(:, :) = scalar + 2.
  v2(:) = matrix(:, 2)
  v2(1) = 1.
end subroutine transp_vect
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'transp_vect{"_c_ptr" if use_c_ptr else ""}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='transp_vect')

    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')
    function(n, m, scalar, v1, v2)

    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    ccode = f2c.c_path.read_text()
    print(ccode)
    """
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.transp_vect_fc_mod.transp_vect_fc

    # Test the trnapiled C kernel
    n, m = 3, 4
    scalar = 2.0
    v1 = np.zeros(shape=(n,), order='F')
    v2 = np.zeros(shape=(n,), order='F')
    fc_function(n, m, scalar, v1, v2)

    assert np.all(v1 == 3.)
    assert v2[0] == 1. and np.all(v2[1:] == 4.)
    """

@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_intrinsics(tmp_path, builder, frontend, use_c_ptr):
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
    filepath = tmp_path/(f'transpile_intrinsics{"_c_ptr" if use_c_ptr else ""}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='transpile_intrinsics')

    # Test the reference solution
    v1, v2, v3, v4 = 2., 4., 1., 5.
    vmin, vmax, vabs, vmin_nested, vmax_nested = function(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.transpile_intrinsics_fc_mod.transpile_intrinsics_fc

    vmin, vmax, vabs, vmin_nested, vmax_nested = fc_function(v1, v2, v3, v4)
    assert vmin == 2. and vmax == 4. and vabs == 2.
    assert vmin_nested == 1. and vmax_nested == 5.


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_loop_indices(tmp_path, builder, frontend, use_c_ptr):
    """
    Test to ensure loop indexing translates correctly
    """

    fcode = """
subroutine transp_loop_ind(n, idx, mask1, mask2, mask3)
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
end subroutine transp_loop_ind
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'transp_loop_ind{"_c_ptr" if use_c_ptr else ""}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='transp_loop_ind')

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
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.transp_loop_ind_fc_mod.transp_loop_ind_fc

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


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_logical_statements(tmp_path, builder, frontend, use_c_ptr):
    """
    A simple test routine to test logical statements
    """

    fcode = """
subroutine logical_stmts(v1, v2, v_xor, v_xnor, v_nand, v_neqv, v_val)
  logical, intent(in) :: v1, v2
  logical, intent(out) :: v_xor, v_nand, v_xnor, v_neqv, v_val(2)

  v_xor = (v1 .and. .not. v2) .or. (.not. v1 .and. v2)
  v_xnor = v1 .eqv. v2
  v_nand = .not. (v1 .and. v2)
  v_neqv = v1 .neqv. v2
  v_val(1) = .true.
  v_val(2) = .false.

end subroutine logical_stmts
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'logical_stmts{"_c_ptr" if use_c_ptr else ""}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='logical_stmts')

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
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.logical_stmts_fc_mod.logical_stmts_fc

    for v1 in range(2):
        for v2 in range(2):
            v_val = np.zeros(shape=(2,), order='F', dtype=np.int32)
            v_xor, v_xnor, v_nand, v_neqv = fc_function(v1, v2, v_val)
            assert v_xor == (v1 and not v2) or (not v1 and v2)
            assert v_xnor == (v1 and v2) or not (v1 or v2)
            assert v_nand == (not (v1 and v2))
            assert v_neqv == ((not (v1 and v2)) and (v1 or v2))
            assert v_val[0] and not v_val[1]


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_multibody_conditionals(tmp_path, builder, frontend, use_c_ptr):
    """
    Test correct transformation of multi-body conditionals.
    """
    fcode = """
subroutine multibody_cond(in1, out1, out2)
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
end subroutine multibody_cond
"""
    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'multibody_cond{"_c_ptr" if use_c_ptr else ""}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='multibody_cond')

    out1, out2 = function(5)
    assert out1 == 1 and out2 == 4

    out1, out2 = function(2)
    assert out1 == 1 and out2 == 2

    out1, out2 = function(-1)
    assert out1 == 1 and out2 == 0

    out1, out2 = function(10)
    assert out1 == 5 and out2 == 5

    clean_test(filepath)

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.multibody_cond_fc_mod.multibody_cond_fc

    out1, out2 = fc_function(5)
    assert out1 == 1 and out2 == 4

    out1, out2 = fc_function(2)
    assert out1 == 1 and out2 == 2

    out1, out2 = fc_function(-1)
    assert out1 == 1 and out2 == 0

    out1, out2 = fc_function(10)
    assert out1 == 5 and out2 == 5


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OFP, 'Prefix/elemental support not implemented')]
))
def test_transpile_inline_elemental_functions(tmp_path, builder, frontend, use_c_ptr):
    """
    Test correct inlining of elemental functions in C transpilation.
    """
    fcode_module = """
module multiply_mod_c
  use iso_fortran_env, only: real64
  implicit none
contains

  elemental function multiply(a, b)
    real(kind=real64) :: multiply
    real(kind=real64), intent(in) :: a, b

    multiply = a * b
  end function multiply
end module multiply_mod_c
"""

    fcode = """
subroutine inline_elemental(v1, v2, v3)
  use iso_fortran_env, only: real64
  use multiply_mod_c, only: multiply
  real(kind=real64), intent(in) :: v1
  real(kind=real64), intent(out) :: v2, v3

  v2 = multiply(v1, 6._real64)
  v3 = 600. + multiply(6._real64, 11._real64)
end subroutine inline_elemental
"""
    # Generate reference code, compile run and verify
    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    refname = f'ref_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)

    v2, v3 = reference.inline_elemental(11.)
    assert v2 == 66.
    assert v3 == 666.

    (tmp_path/f'{module.name}.f90').unlink()
    (tmp_path/f'{routine.name}.f90').unlink()

    # Now transpile with supplied elementals but without module
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)

    f2c = FortranCTransformation(inline_elementals=True, use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_mod = c_kernel.inline_elemental_fc_mod

    v2, v3 = fc_mod.inline_elemental_fc(11.)
    assert v2 == 66.
    assert v3 == 666.


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OFP, 'Prefix/elemental support not implemented')]
))
def test_transpile_inline_elementals_recursive(tmp_path, builder, frontend, use_c_ptr):
    """
    Test correct inlining of nested elemental functions.
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

    multiply_plus_one = multiply(plus_one(a), b)
  end function multiply_plus_one
end module multiply_plus_one_mod
"""

    fcode = """
subroutine inline_elementals_rec(v1, v2, v3)
  use iso_fortran_env, only: real64
  use multiply_plus_one_mod, only: multiply_plus_one
  real(kind=real64), intent(in) :: v1
  real(kind=real64), intent(out) :: v2, v3

  v2 = multiply_plus_one(v1, 6._real64)
  v3 = 600. + multiply_plus_one(5._real64, 11._real64)
end subroutine inline_elementals_rec
"""
    # Generate reference code, compile run and verify
    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    refname = f'ref_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    reference = jit_compile_lib([module, routine], path=tmp_path, name=refname, builder=builder)

    v2, v3 = reference.inline_elementals_rec(10.)
    assert v2 == 66.
    assert v3 == 666.

    (tmp_path/f'{module.name}.f90').unlink()
    (tmp_path/f'{routine.name}.f90').unlink()

    # Now transpile with supplied elementals but without module
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)

    f2c = FortranCTransformation(inline_elementals=True, use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_mod = c_kernel.inline_elementals_rec_fc_mod

    v2, v3 = fc_mod.inline_elementals_rec_fc(10.)
    assert v2 == 66.
    assert v3 == 666.


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_expressions(tmp_path, builder, frontend, use_c_ptr):
    """
    A simple test to verify expression parenthesis and resolution
    of minus sign
    """

    fcode = """
subroutine transpile_expressions(n, scalar, vector)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n
  real(kind=real64), intent(in) :: scalar
  real(kind=real64), intent(inout) :: vector(n)

  integer :: i

  vector(1) = scalar
  do i=2, n
     vector(i) = vector(i-1) - (-scalar)
  end do
end subroutine transpile_expressions
"""

    # Generate reference code, compile run and verify
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/f'{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend!s}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 10
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F')
    function(n, scalar, vector)

    assert np.all(vector == [i * scalar for i in range(1, n+1)])

    # Generate and test the transpiled C kernel
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr)
    f2c.apply(source=routine, path=tmp_path)
    libname = f'fc_{routine.name}{"_c_ptr" if use_c_ptr else ""}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.transpile_expressions_fc_mod.transpile_expressions_fc

    # Make sure minus signs are represented correctly in the C code
    ccode = f2c.c_path.read_text()
    # double minus due to index shift to 0
    assert 'vector[i - 1 - 1]' in ccode or 'vector[-1 + i - 1]' in ccode
    assert 'vector[i - 1]' in ccode
    assert '-scalar' in ccode  # scalar with negative sign

    n = 10
    scalar = 2.0
    vector = np.zeros(shape=(n,), order='F')
    fc_function(n, scalar, vector)

    assert np.all(vector == [i * scalar for i in range(1, n+1)])


@pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('language', ('c', 'cuda'))
@pytest.mark.parametrize('chevron', (False, True))
def test_transpile_call(tmp_path, frontend, use_c_ptr, language, chevron):
    fcode_module = """
module transpile_call_kernel_mod
  implicit none
contains

  subroutine transpile_call_kernel(a, b, c, arr1, arr2, len)
    integer, intent(inout) :: a, c
    integer, intent(in) :: b
    integer, intent(in) :: len
    integer, intent(inout) :: arr1(len, len)
    integer, intent(in) :: arr2(len, len)
    a = b
    c = b
  end subroutine transpile_call_kernel
end module transpile_call_kernel_mod
"""

    fcode = """
subroutine transpile_call_driver(a)
  use transpile_call_kernel_mod, only: transpile_call_kernel
    integer, intent(inout) :: a
    integer, parameter :: len = 5
    integer :: arr1(len, len)
    integer :: arr2(len, len)
    integer :: b
    b = 2 * len
    call transpile_call_kernel(a, b, arr2(1, 1), arr1, arr2, len)
end subroutine transpile_call_driver
"""
    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=module)
    if chevron:
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        calls[0]._update(chevron=(sym.IntLiteral(1), sym.IntLiteral(1)))
    f2c = FortranCTransformation(use_c_ptr=use_c_ptr, path=tmp_path, language=language)
    f2c.apply(source=module.subroutine_map['transpile_call_kernel'], path=tmp_path, role='kernel')
    ccode_kernel = f2c.c_path.read_text().replace(' ', '').replace('\n', '')
    f2c.apply(source=routine, path=tmp_path, role='kernel')
    ccode_driver = f2c.c_path.read_text().replace(' ', '').replace('\n', '')

    assert "int*a,intb,int*c" in ccode_kernel
    # check for applied Dereference
    assert "(*a)=b;" in ccode_kernel
    assert "(*c)=b;" in ccode_kernel
    # check for applied 'const' and 'restrict'/'__restrict__'
    if language == 'cuda':
        assert 'int*a,intb,int*c,int*__restrict__arr1,constint*__restrict__arr2' in ccode_kernel
    else:
        assert 'int*a,intb,int*c,int*restrictarr1,int*restrictarr2' in ccode_kernel
    # check for applied Reference and chevron
    if chevron and language == 'cuda':
        assert "transpile_call_kernel<<<1,1>>>((&a),b,(&arr2[" in ccode_driver
    else:
        assert "transpile_call_kernel((&a),b,(&arr2[" in ccode_driver

# @pytest.mark.parametrize('use_c_ptr', (False, True))
@pytest.mark.parametrize('frontend', available_frontends())
# @pytest.mark.parametrize('language', ('c', 'cuda'))
# @pytest.mark.parametrize('chevron', (False, True))
def test_transpile_call_intent(tmp_path, frontend): # , use_c_ptr, language, chevron):
    fcode_nested_kernel = """
module transpile_call_nested_kernel_mod
  implicit none
contains

  subroutine transpile_call_nested_kernel(a2, b2)
    integer, intent(in) :: a2
    integer, intent(inout) :: b2
  end subroutine transpile_call_nested_kernel
end module transpile_call_nested_kernel_mod
"""

    fcode_kernel = """
module transpile_call_kernel_mod
  implicit none
contains

  subroutine transpile_call_kernel(a, b)
    use transpile_call_nested_kernel_mod, only: transpile_call_nested_kernel
    integer, intent(inout) :: a
    integer, intent(inout) :: b
    call transpile_call_nested_kernel(a,b)
  end subroutine transpile_call_kernel
end module transpile_call_kernel_mod
"""

    fcode = """
subroutine transpile_call_driver(a, b)
  use transpile_call_kernel_mod, only: transpile_call_kernel
    integer, intent(inout) :: a
    integer, intent(inout) :: b
    call transpile_call_kernel(a, b)
end subroutine transpile_call_driver
"""
    nested_kernel_mod = Module.from_source(fcode_nested_kernel, frontend=frontend)
    kernel_mod = Module.from_source(fcode_kernel, frontend=frontend, definitions=nested_kernel_mod)
    routine = Subroutine.from_source(fcode, frontend=frontend, definitions=kernel_mod)

    f2c = FortranCTransformation(use_c_ptr=True, path=tmp_path, language='c')
    f2c.apply(source=nested_kernel_mod.subroutine_map['transpile_call_nested_kernel'], path=tmp_path, role='kernel')
    ccode_nested_kernel = f2c.c_path.read_text() # .replace(' ', '').replace('\n', '')
    f2c.apply(source=kernel_mod.subroutine_map['transpile_call_kernel'], path=tmp_path, role='kernel')
    ccode_kernel = f2c.c_path.read_text() # .replace(' ', '').replace('\n', '')
    f2c.apply(source=routine, path=tmp_path, role='kernel')
    ccode_driver = f2c.c_path.read_text() # .replace(' ', '').replace('\n', '')

    print(f"{ccode_driver}")
    print(f"-----")
    print(f"{ccode_kernel}")
    print(f"----")
    print(f"{ccode_nested_kernel}")

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('f_type', ['integer', 'real'])
@pytest.mark.parametrize('codegen', (cgen, cppgen, cudagen))
def test_transpile_inline_functions(tmp_path, frontend, f_type, codegen):
    """
    Test correct transpilation of functions in C transpilation.
    """

    fcode = f"""
function add(a, b)
    {f_type} :: add
    {f_type}, intent(in) :: a, b

    add = a + b
end function add
""".format(f_type)

    routine = Subroutine.from_source(fcode, frontend=frontend)
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=tmp_path)

    f_type_map = {'integer': 'int', 'real': 'double'}
    c_routine = codegen(routine)
    assert 'return add;' in c_routine
    assert f'{f_type_map[f_type]} add(' in c_routine


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('f_type', ['integer', 'real'])
@pytest.mark.parametrize('codegen', (cgen, cppgen, cudagen))
def test_transpile_inline_functions_return(tmp_path, frontend, f_type, codegen):
    """
    Test correct transpilation of functions in C transpilation.
    """

    fcode = f"""
function add(a, b) result(res)
    {f_type} :: res
    {f_type}, intent(in) :: a, b

    res = a + b
end function add
""".format(f_type)

    routine = Subroutine.from_source(fcode, frontend=frontend)
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=tmp_path)

    f_type_map = {'integer': 'int', 'real': 'double'}
    c_routine = codegen(routine)
    assert 'return res;' in c_routine
    assert f'{f_type_map[f_type]} add(' in c_routine


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('codegen', (cgen, cppgen, cudagen))
def test_transpile_multiconditional(tmp_path, builder, frontend, codegen):
    """
    A simple test to verify multiconditionals/select case statements.
    """

    fcode = """
subroutine multi_cond(in, out)
  implicit none
  integer, intent(in) :: in
  integer, intent(inout) :: out

  select case (in)
    case (1)
        out = 10
    case (2)
        out = 20
    case default
        out = 100
  end select

end subroutine multi_cond
""".strip()

    # for testing purposes
    in_var = 0
    test_vals = [0, 1, 2, 5]
    expected_results = [100, 10, 20, 100]
    out_var = np.int_([0])

    # compile original Fortran version
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/f'{routine.name}_{frontend!s}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    # test Fortran version
    for i, val in enumerate(test_vals):
        in_var = val
        function(in_var, out_var)
        assert out_var == expected_results[i]

    # apply F2C trafo
    f2c = FortranCTransformation()
    f2c.apply(source=routine, path=tmp_path)

    # check whether 'switch' statement is within C code
    assert 'switch' in codegen(routine)

    # compile C version
    libname = f'fc_{routine.name}_{frontend}'
    c_kernel = jit_compile_lib([f2c.wrapperpath, f2c.c_path], path=tmp_path, name=libname, builder=builder)
    fc_function = c_kernel.multi_cond_fc_mod.multi_cond_fc
    # test C version
    for i, val in enumerate(test_vals):
        in_var = val
        fc_function(in_var, out_var)
        assert out_var == expected_results[i]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transpile_multiconditional_range(tmp_path, frontend):
    """
    A simple test to verify multiconditionals/select case statements.
    """

    fcode = """
subroutine transpile_multi_conditional_range(in, out)
  implicit none
  integer, intent(in) :: in
  integer, intent(inout) :: out

  select case (in)
    case (1:5)
        out = 10
    case default
        out = 100
  end select

end subroutine transpile_multi_conditional_range
""".strip()

    # for testing purposes
    in_var = 0
    test_vals = [0, 1, 2, 5, 6]
    expected_results = [100, 10, 10, 10, 100]
    out_var = np.int_([0])

    # compile original Fortran version
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/f'{routine.name}_{frontend!s}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    # test Fortran version
    for i, val in enumerate(test_vals):
        in_var = val
        function(in_var, out_var)
        assert out_var == expected_results[i]

    clean_test(filepath)

    # apply F2C trafo
    # TODO: RangeIndex as case is not yet implemented!
    #  'NotImplementedError' is raised
    f2c = FortranCTransformation()
    with pytest.raises(NotImplementedError):
        f2c.apply(source=routine, path=tmp_path)

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'iend'))


@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk')


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(name='config')
def fixture_config():
    """
    Default configuration dict with basic options.
    """
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': False,  # cudafor import
        },
        'routines': {
            'driver': {'role': 'driver'}
        }
    }

def remove_whitespace_linebreaks(text):
    return text.replace(' ', '').replace('\n', ' ').replace('\r', '').replace('\t', '').lower()

def remove_whitespace_linebreaks_2(text):
    return text
    # return text.replace(' ', '').replace('\n', ' ').replace('\r', '').replace('\t', '').lower()

@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuda_parametrise(tmp_path, here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 0, thus including parametrising (array dimension(s))
    """

    proj = here / '../../tests/sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    dic2p = {'nz': 137}
    cuda_transform = SCCLowLevelParametrise(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='parametrise',
        dim_vars=(vertical.size,), as_kwarguments=True, remove_vector_section=True,
        use_c_ptr=True, dic2p=dic2p, path=here, mode='cuda'
    )
    scheduler.process(transformation=cuda_transform)
    f2c_transformation = FortranCTransformation(path=tmp_path, language='cuda', use_c_ptr=True)
    scheduler.process(transformation=f2c_transformation)

    # f_driver = remove_whitespace_linebreaks(fgen(scheduler["driver_mod#driver"].ir))
    fc_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_fc.F90'))
    c_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.c'))
    c_kernel_header = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.h'))
    c_kernel_launch = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c_launch.h'))
    c_device = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.c'))
    # c_device_header = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.h'))
    c_elemental_device = remove_whitespace_linebreaks(read_file(tmp_path/'elemental_device_c.c'))

    calls = FindNodes(ir.CallStatement).visit(scheduler["driver_mod#driver"].ir.body)
    assert len(calls) == 3
    for call in calls:
        assert str(call.name).lower() == 'kernel'
        assert call.pragma[0].keyword == 'loki'
        assert 'removed_loop' in call.pragma[0].content
    # kernel_fc.F90
    assert '!$acchost_datause_device(q,t,z)' in fc_kernel
    assert 'kernel_iso_c(start,nlon,c_loc(q),c_loc(t),c_loc(z),nb,tot,iend)' in fc_kernel
    assert 'bind(c,name="kernel_c_launch")' in fc_kernel
    assert 'useiso_c_binding' in fc_kernel
    # kernel_c.c
    assert '#include<cuda.h>' in c_kernel
    assert '#include<cuda_runtime.h>' in c_kernel
    assert '#include"kernel_c.h"' in c_kernel
    assert '#include"kernel_c_launch.h"' in c_kernel
    assert '__global__voidkernel_c' in c_kernel
    assert 'jl=threadidx.x;' in c_kernel
    assert 'b=blockidx.x;' in c_kernel
    # kernel_c.h
    assert '__global__voidkernel_c' in c_kernel_header
    assert 'jl=threadidx.x;' not in c_kernel_header
    assert 'b=blockidx.x;' not in c_kernel_header
    # kernel_c_launch.h
    assert 'extern"c"' in c_kernel_launch
    assert 'voidkernel_c_launch(' in c_kernel_launch
    assert 'structdim3blockdim;' in c_kernel_launch
    assert 'structdim3griddim;' in c_kernel_launch
    assert 'griddim=dim3(' in c_kernel_launch
    assert 'blockdim=dim3(' in c_kernel_launch
    assert 'kernel_c<<<griddim,blockdim>>>(' in c_kernel_launch
    assert 'cudadevicesynchronize();' in c_kernel_launch
    # device_c.c
    assert '#include<cuda.h>' in c_device
    assert '#include<cuda_runtime.h>' in c_device
    assert '#include"device_c.h"' in c_device
    # assert '__device__voiddevice_c' in c_device # TODO
    # elemental_device_c.c
    assert '#include<cuda.h>' in c_elemental_device
    assert '#include<cuda_runtime.h>' in c_elemental_device
    assert '#include"elemental_device_c.h"' in c_elemental_device
    # assert '__device__voidelemental_device_c' in c_elemental_device # TODO

@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuda_hoist(tmp_path, here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 0, thus including parametrising (array dimension(s))
    """

    proj = here / '../../tests/sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuda_transform = SCCLowLevelHoist(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='parametrise',
        dim_vars=(vertical.size,), as_kwarguments=True, remove_vector_section=True,
        use_c_ptr=True, path=here, mode='cuda'
    )
    scheduler.process(transformation=cuda_transform)
    f2c_transformation = FortranCTransformation(path=tmp_path, language='cuda', use_c_ptr=True)
    scheduler.process(transformation=f2c_transformation)

    # f_driver = remove_whitespace_linebreaks(fgen(scheduler["driver_mod#driver"].ir))
    fc_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_fc.F90'))
    c_kernel = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.c'))
    c_kernel_header = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c.h'))
    c_kernel_launch = remove_whitespace_linebreaks(read_file(tmp_path/'kernel_c_launch.h'))
    c_device = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.c'))
    # c_device_header = remove_whitespace_linebreaks(read_file(tmp_path/'device_c.h'))
    c_elemental_device = remove_whitespace_linebreaks(read_file(tmp_path/'elemental_device_c.c'))

    calls = FindNodes(ir.CallStatement).visit(scheduler["driver_mod#driver"].ir.body)
    assert len(calls) == 3
    for call in calls:
        assert str(call.name).lower() == 'kernel'
        assert call.pragma[0].keyword == 'loki'
        assert 'removed_loop' in call.pragma[0].content
    # kernel_fc.F90
    assert '!$acchost_datause_device(q,t,z,local_z,device_local_x)' in fc_kernel
    assert 'kernel_iso_c(start,nlon,nz,c_loc(q),c_loc(t),c_loc(z)' in fc_kernel
    assert 'c_loc(z),nb,tot,iend,c_loc(local_z),c_loc(device_local_x))' in fc_kernel
    assert 'bind(c,name="kernel_c_launch")' in fc_kernel
    assert 'useiso_c_binding' in fc_kernel
    # kernel_c.c
    assert '#include<cuda.h>' in c_kernel
    assert '#include<cuda_runtime.h>' in c_kernel
    assert '#include"kernel_c.h"' in c_kernel
    assert '#include"kernel_c_launch.h"' in c_kernel
    assert '__global__voidkernel_c' in c_kernel
    assert 'jl=threadidx.x;' in c_kernel
    assert 'b=blockidx.x;' in c_kernel
    # kernel_c.h
    assert '__global__voidkernel_c' in c_kernel_header
    assert 'jl=threadidx.x;' not in c_kernel_header
    assert 'b=blockidx.x;' not in c_kernel_header
    # kernel_c_launch.h
    assert 'extern"c"' in c_kernel_launch
    assert 'voidkernel_c_launch(' in c_kernel_launch
    assert 'structdim3blockdim;' in c_kernel_launch
    assert 'structdim3griddim;' in c_kernel_launch
    assert 'griddim=dim3(' in c_kernel_launch
    assert 'blockdim=dim3(' in c_kernel_launch
    assert 'kernel_c<<<griddim,blockdim>>>(' in c_kernel_launch
    assert 'cudadevicesynchronize();' in c_kernel_launch
    # device_c.c
    assert '#include<cuda.h>' in c_device
    assert '#include<cuda_runtime.h>' in c_device
    assert '#include"device_c.h"' in c_device
    # assert '__device__voiddevice_c' in c_device # TODO
    # elemental_device_c.c
    assert '#include<cuda.h>' in c_elemental_device
    assert '#include<cuda_runtime.h>' in c_elemental_device
    assert '#include"elemental_device_c.h"' in c_elemental_device
    # assert '__device__voidelemental_device_c' in c_elemental_device # TODO

@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_cuda_hoist_extended(tmp_path, here, frontend, config, horizontal, vertical, blocking):
    """
    Test SCC-CUF transformation type 0, thus including parametrising (array dimension(s))
    """

    proj = here / '../../tests/sources/projSccCuf/module'

    scheduler = Scheduler(paths=[proj], config=config, seed_routines=['driver'], frontend=frontend)

    cuda_transform = SCCLowLevelHoist(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        transformation_type='parametrise',
        dim_vars=(vertical.size,), as_kwarguments=True, remove_vector_section=True,
        use_c_ptr=True, path=here, mode='cuda'
    )
    scheduler.process(transformation=cuda_transform)
    f2c_transformation = FortranCTransformation(path=tmp_path, language='cuda', use_c_ptr=True)
    scheduler.process(transformation=f2c_transformation)

    f_driver = remove_whitespace_linebreaks_2(fgen(scheduler["driver_mod#driver"].ir))
    fc_kernel = remove_whitespace_linebreaks_2(read_file(tmp_path/'kernel_fc.F90'))
    c_kernel = remove_whitespace_linebreaks_2(read_file(tmp_path/'kernel_c.c'))
    c_kernel_header = remove_whitespace_linebreaks_2(read_file(tmp_path/'kernel_c.h'))
    c_kernel_launch = remove_whitespace_linebreaks_2(read_file(tmp_path/'kernel_c_launch.h'))
    c_device = remove_whitespace_linebreaks_2(read_file(tmp_path/'device_c.c'))
    # c_device_header = remove_whitespace_linebreaks_2(read_file(tmp_path/'device_c.h'))
    c_elemental_device = remove_whitespace_linebreaks_2(read_file(tmp_path/'elemental_device_c.c'))

    print(f"driver\n----------------")
    print(f"{f_driver}")
    print(f"fc_kernel\n----------------")
    print(f"{fc_kernel}")
    print(f"kernel\n----------------")
    print(f"{c_kernel}")
