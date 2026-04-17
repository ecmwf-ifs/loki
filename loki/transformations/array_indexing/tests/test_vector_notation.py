# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines

import pytest
import numpy as np

from loki import Module, Subroutine, Dimension
from loki.jit_build import jit_compile, jit_compile_lib, Builder, Obj
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes, FindVariables

from loki.transformations.array_indexing.vector_notation import (
    resolve_vector_notation, resolve_vector_dimension,
    remove_explicit_array_dimensions, add_explicit_array_dimensions
)


@pytest.fixture(scope='function', name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_promote_resolve_vector_notation(tmp_path, frontend):
    """
    Apply and test resolve vector notation utility.
    """
    fcode = """
subroutine transform_resolve_vector_notation(ret1, ret2)
  implicit none
  integer, parameter :: param1 = 3
  integer, parameter :: param2 = 5
  integer, intent(out) :: ret1(param1, param1), ret2(param1, param2)
  integer :: tmp, jk

  ret1(:, :) = 11
  ret2(:, :) = 42

end subroutine transform_resolve_vector_notation
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine)

    loops = FindNodes(ir.Loop).visit(routine.body)
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]

    assert len(loops) == 4
    assert loops[0].variable == 'i_ret1_1'
    assert loops[0].bounds == '1:param1' if frontend != OMNI else '1:3:1'
    assert loops[1].variable == 'i_ret1_0'
    assert loops[1].bounds == '1:param1' if frontend != OMNI else '1:3:1'
    assert loops[2].variable == 'i_ret2_1'
    assert loops[2].bounds == '1:param2' if frontend != OMNI else '1:5:1'
    assert loops[3].variable == 'i_ret2_0'
    assert loops[3].bounds == '1:param1' if frontend != OMNI else '1:3:1'

    assert len(arrays) == 2
    assert arrays[0].dimensions == ('i_ret1_0', 'i_ret1_1')
    assert arrays[1].dimensions == ('i_ret2_0', 'i_ret2_1')

    ret1 = np.zeros(shape=(3, 3), order='F', dtype=np.int32)
    ret2 = np.zeros(shape=(3, 5), order='F', dtype=np.int32)

    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    function(ret1, ret2)

    assert np.all(ret1 == 11)
    assert np.all(ret2 == 42)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('kidia_loop', (True, False))
def test_transform_resolve_vector_notation_common_loops(tmp_path, frontend, kidia_loop):
    """
    Apply and test resolve vector notation utility with already
    available/appropriate loops.
    """
    fcode = f"""
subroutine transform_resolve_vector_notation_common_loops(scalar, vector, vector_2, matrix, n, m, l, kidia, kfdia)
  implicit none
  integer, intent(in) :: n, m, l, kidia, kfdia
  integer, intent(inout) :: scalar, vector(n), vector_2(n), matrix(l, n)
  integer :: tmp_scalar, tmp_vector(n, m), tmp_matrix(l, m, n), tmp_dummy(n, 0:4)
  integer :: jl, jk, jm

  tmp_dummy(:,:) = 0
  tmp_vector(:, 1) = tmp_dummy(:, 1)
  tmp_vector(:, :) = 0
  tmp_matrix(:, :, :) = 0
  matrix(:, :) = 0

  do jl={'kidia,kfdia' if kidia_loop else '1,n'}
    do jm=1,m
      tmp_vector(jl, jm) = scalar + jl
    end do
  end do

  do jm=1,m
    do jl=1,n
      scalar = jl
      vector(jl) = tmp_vector(jl, jm) + tmp_vector(jl, jm)

      do jk=1,l
        tmp_matrix(jk, jm, jl) = vector(jl) + jk
      end do
    end do
  end do


  do jk=1,l
    matrix(jk, :) = 0
    do jm=1,m
      do jl=1,n
        matrix(jk, jl) = tmp_matrix(jk, jm, jl)
      end do
    end do
  end do

  vector_2(:) = 1
  vector_2(kidia:kfdia) = 2

end subroutine transform_resolve_vector_notation_common_loops
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Test the original implementation
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    n = 3
    m = 2
    l = 3
    kidia = 1
    kfdia = n
    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    vector_2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    function(scalar, vector, vector_2, matrix, n, m, l, kidia, kfdia)

    assert scalar == 3
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))

    resolve_vector_notation(routine)
    loops = FindNodes(ir.Loop).visit(routine.body)
    arrays = [var for var in FindVariables(unique=False).visit(routine.body) if isinstance(var, sym.Array)]
    assert len(loops) == 21
    assert loops[0].variable == 'i_tmp_dummy_1' and loops[0].bounds == '0:4'
    assert loops[1].variable == 'jl' and loops[1].bounds == '1:n'
    assert loops[2].variable == 'jl' and loops[2].bounds == '1:n'
    assert loops[3].variable == 'jm' and loops[3].bounds == '1:m'
    assert loops[4].variable == 'jl' and loops[4].bounds == '1:n'
    assert loops[5].variable == 'jl' and loops[5].bounds == '1:n'
    assert loops[6].variable == 'jm' and loops[6].bounds == '1:m'
    assert loops[7].variable == 'jk' and loops[7].bounds == '1:l'
    assert loops[8].variable == 'jl' and loops[8].bounds == '1:n'
    assert loops[9].variable == 'jk' and loops[9].bounds == '1:l'
    assert loops[10].variable == 'jl'
    if kidia_loop:
        assert loops[10].bounds == 'kidia:kfdia'
    else:
        assert loops[10].bounds == '1:n'
    assert loops[11].variable == 'jm' and loops[11].bounds == '1:m'
    assert loops[12].variable == 'jm' and loops[12].bounds == '1:m'
    assert loops[13].variable == 'jl' and loops[13].bounds == '1:n'
    assert loops[14].variable == 'jk' and loops[14].bounds == '1:l'
    assert loops[15].variable == 'jk' and loops[15].bounds == '1:l'
    assert loops[16].variable == 'jl' and loops[16].bounds == '1:n'
    assert loops[17].variable == 'jm' and loops[17].bounds == '1:m'
    assert loops[18].variable == 'jl' and loops[18].bounds == '1:n'
    assert loops[19].variable == 'jl' and loops[19].bounds == '1:n'
    if kidia_loop:
        assert loops[20].variable == 'jl'
        assert loops[20].bounds == 'kidia:kfdia'
    else:
        assert loops[20].variable == 'i_vector_2_0'
        assert loops[20].bounds == 'kidia:kfdia'

    assert len(arrays) == 17
    assert arrays[0].name.lower() == 'tmp_dummy' and arrays[0].dimensions == ('jl', 'i_tmp_dummy_1')
    assert arrays[1].name.lower() == 'tmp_vector' and arrays[1].dimensions == ('jl', 1)
    assert arrays[2].name.lower() == 'tmp_dummy' and arrays[2].dimensions == ('jl', 1)
    assert arrays[3].name.lower() == 'tmp_vector' and arrays[3].dimensions == ('jl', 'jm')
    assert arrays[4].name.lower() == 'tmp_matrix' and arrays[4].dimensions == ('jk', 'jm', 'jl')
    assert arrays[15].name.lower() == 'vector_2' and arrays[15].dimensions == ('jl',)
    assert arrays[16].name.lower() == 'vector_2'
    if kidia_loop:
        assert arrays[16].dimensions == ('jl',)
    else:
        assert arrays[16].dimensions == ('i_vector_2_0',)

    # Test promoted routine
    resolved_filepath = tmp_path/(f'{routine.name}_resolved_{frontend}.f90')
    resolved_function = jit_compile(routine, filepath=resolved_filepath, objname=routine.name)

    n = 3
    m = 2
    l = 3
    kidia = 1
    kfdia = n
    scalar = np.array(0)
    vector = np.zeros(shape=(n,), order='F', dtype=np.int32)
    vector_2 = np.zeros(shape=(n,), order='F', dtype=np.int32)
    matrix = np.zeros(shape=(n, n), order='F', dtype=np.int32)
    resolved_function(scalar, vector, vector_2, matrix, n, m, l, kidia, kfdia)

    assert scalar == 3
    assert np.all(vector == np.arange(1, n + 1)*2)
    assert np.all(matrix == np.sum(np.mgrid[1:4,2:8:2], axis=0))


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'OMNI does not like missing information')]))
def test_transform_inline_call_resolve_vector_notation(frontend):
    """
    Apply and test resolve vector notation utility to not apply to a inline call
    although Loki needs to assume it is an array.
    """
    fcode = """
subroutine transform_resolve_vector_notation_inline_call(x)
  use some_mod, only: some_func
  implicit none
  integer, parameter :: param1 = 3
  integer, parameter :: param2 = 5
  integer, intent(in) :: x(param1, param2)

  ! should stay like that
  tmp = some_func(ret1(1, 1))

end subroutine transform_resolve_vector_notation_inline_call
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine)
    var_map = {var.name.lower(): var for var in FindVariables(unique=False).visit(routine.body)
            if isinstance(var, sym.Array)}
    # Fortran's questionable choice of having the same syntax for a inline call and array access ...
    assert 'some_func' in var_map
    assert var_map['some_func'].dimensions == ('ret1(1, 1)',)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('calls_only', (False, True))
def test_transform_explicit_dimensions(tmp_path, frontend, builder, calls_only):
    """
    Test making dimensions of arrays explicit and undoing this,
    thus removing colon notation from array dimensions either for all
    or for arrays within (inline) calls only.
    """
    fcode_driver = """
  SUBROUTINE driver_routine(nlon, nlev, a, b)
    use kernel_explicit_dimensions_mod, only: kernel_routine
    INTEGER, INTENT(IN)    :: nlon, nlev
    INTEGER, INTENT(INOUT) :: a(nlon,nlev)
    INTEGER, INTENT(INOUT)  :: b(nlon,nlev)

    call kernel_routine(nlon, a, b=b, nlev=nlev)

  END SUBROUTINE driver_routine
    """

    fcode_kernel = """
  module kernel_explicit_dimensions_mod
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE kernel_routine(nlon, a, b, nlev)
    INTEGER, INTENT(IN)    :: nlon, nlev
    INTEGER, INTENT(INOUT) :: a(nlon,nlev)
    INTEGER, INTENT(INOUT) :: b(nlon,nlev)

    A = MYADD(A, B=B)
  END SUBROUTINE kernel_routine

  PURE ELEMENTAL FUNCTION MYADD(A, B)
    INTEGER :: MYADD
    INTEGER, INTENT(IN) :: A, B

    MYADD = A + B
  END FUNCTION
  end module kernel_explicit_dimensions_mod
    """

    def init_arguments(nlon, nlev):
        a = 2*np.ones(shape=(nlon,nlev,), order='F', dtype=np.int32)
        b = 3*np.ones(shape=(nlon,nlev,), order='F', dtype=np.int32)
        return a, b

    kernel_module = Module.from_source(fcode_kernel, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path],
                                     definitions=[kernel_module])
    kernel = kernel_module.subroutines[0]

    # compile and test reference
    refname = f'ref_explicit_dims_{driver.name}_{frontend}'
    reference = jit_compile_lib([kernel_module, driver], path=tmp_path, name=refname, builder=builder)
    ref_function = reference.driver_routine

    nlon = 10
    nlev = 12
    a_ref, b_ref = init_arguments(nlon, nlev)
    ref_function(nlon, nlev, a_ref, b_ref)
    builder.clean()

    # add explicit array dimensions
    add_explicit_array_dimensions(driver)
    add_explicit_array_dimensions(kernel)
    kernel_call = FindNodes(ir.CallStatement).visit(driver.body)[0]
    kernel_call_array_args = [arg for arg in kernel_call.arguments if isinstance(arg, sym.Array)]
    assert all(len(arg.dimensions) == 2 for arg in kernel_call_array_args)

    # remove explicit array dimensions (possibly only for calls)
    remove_explicit_array_dimensions(driver, calls_only=calls_only)
    remove_explicit_array_dimensions(kernel, calls_only=calls_only)

    kernel_call = FindNodes(ir.CallStatement).visit(driver.body)[0]
    kernel_call_array_args = [arg for arg in kernel_call.arguments if isinstance(arg, sym.Array)]
    assert all(not arg.dimensions for arg in kernel_call_array_args)
    if calls_only:
        assignments = FindNodes(ir.Assignment).visit(kernel.body)
        assert len(assignments) == 1
        assert len(assignments[0].lhs.dimensions) == 2
        parameters = (assignments[0].rhs.parameters[0],)
        parameters += (assignments[0].rhs.kwarguments[0][1],)
        assert not parameters[0].dimensions
        assert not parameters[1].dimensions
    else:
        kernel_arrays = FindVariables().visit(kernel.body)
        assert all(not arr.dimensions for arr in kernel_arrays)

    # compile and test the resulting code
    testname = f'test_explicit_dims_{"calls_only_" if calls_only else ""}_{driver.name}_{frontend}'
    test = jit_compile_lib([kernel_module, driver], path=tmp_path, name=testname, builder=builder)
    test_function = test.driver_routine

    a_test, b_test = init_arguments(nlon, nlev)
    test_function(nlon, nlev, a_test, b_test)

    # check whether reference and flattened variant(s) produce same result
    assert (a_test == a_ref).all()
    assert (b_test == b_ref).all()

    builder.clean()


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_dimension(frontend):
    """ Test vector resolution utility for a single dimension. """

    fcode = """
subroutine kernel(start, end, nlon, nlev, z, work, play, sleep, repeat)
  integer, intent(in) :: start, end, nlon, nlev
  real, intent(in) :: z
  real, intent(out) :: work(nlon), play(nlon, nlev), sleep(nlev,nlev), repeat(nlev,nlon)
  integer :: jl
  real :: work_maxval

  work(start:end) = 0.
  work_maxval = maxval(work(start:end))

  play(:,1:nlev) = 42.
  sleep(:, :) = z * z * z
  repeat(:,start:end) = 6.66
end subroutine kernel
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    horizontal = Dimension(name='horizontal', index='jl', lower='start', upper='end')
    resolve_vector_dimension(routine, dimension=horizontal)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 2

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 5

    # Check that the first expression has been wrapped
    assert assigns[0] in loops[0].body
    assert assigns[0].lhs == 'work(jl)'

    # Ensure that none of the other sections has been wrapped
    assert not assigns[1] in loops[0].body
    assert not assigns[1] in loops[1].body
    assert 'maxval' == assigns[1].rhs.name.lower()
    assert 'start:end' in assigns[1].rhs.parameters[0].dimensions

    assert not assigns[2] in loops[0].body
    assert not assigns[2] in loops[1].body
    assert assigns[2].lhs == 'play(:,1:nlev)'

    assert not assigns[3] in loops[0].body
    assert not assigns[3] in loops[1].body
    assert assigns[3].lhs == 'sleep(:,:)'

    # Check that the last expression has been partially wrapped
    assert assigns[4] in loops[1].body
    assert assigns[4].lhs == 'repeat(:,jl)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_dimension_extended(frontend):
    """
    Test vector resolution for multi-dimensional arrays with mixed
    explicit/implicit ranges and IFS-like patterns including derived
    types, pointers, literal lists, and shifted ranges.

    Combines coverage from the original test_resolve_vector_dimension_2
    and test_resolve_vector_dimension_3.
    """

    fcode = """
subroutine test_extended(klon, klev, ngpblks, nproma)
  implicit none
  integer, intent(in) :: klon, klev, ngpblks, nproma
  real :: var(klon, 4, 3, klev, 5, ngpblks)
  real :: local_var1(nproma, klev, ngpblks)
  real :: local_var2(nproma, klev, ngpblks)
  real :: local_var3(nproma, klev, ngpblks)
  real :: local_var4(nproma, klev, ngpblks)
  real :: ptr_src(nproma, klev, ngpblks)
  real :: shifted_src(nproma, 0:klev, ngpblks)
  integer :: jl, ibl
  integer :: start, end

  ! Part A: Multi-dimensional arrays with mixed explicit/implicit ranges
  do ibl=1, ngpblks
    start = 1
    end = klon

    ! A1: 4 implicit range dims inside an explicit jl loop
    do jl=start,end
      var(jl, :, :, :, :, ibl) = 0
    enddo

    ! A2: 5 implicit range dims (first dim should resolve by resolve_vector_notation)
    var(:, :, :, :, :, ibl) = 0
  enddo

  ! Part B: IFS-like patterns inside a block loop
  do ibl=1, ngpblks
    start = 1
    end = nproma

    ! B1: 2D array zeroing (both dims implicit)
    local_var1(:, :, ibl) = 0

    ! B2: Literal list assignment -- should NOT be resolved
    local_var1(1:2, 1, ibl) = (/ 2.739, 4.043 /)

    ! B3: Multi-term RHS with matching ranges
    local_var2(:, :, ibl) = ptr_src(:, :, ibl) + local_var3(:, :, ibl)

    ! B4: RHS with shifted range (0:klev-1 on RHS vs 1:klev on LHS)
    local_var4(:, 1:klev, ibl) = shifted_src(:, 0:klev-1, ibl)
  enddo
end subroutine test_extended
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    horizontal = Dimension(
        name='horizontal', index='jl',
        lower=['start'], upper=['end'], size=['klon', 'nproma']
    )
    resolve_vector_dimension(routine, dimension=horizontal, derive_qualified_ranges=True)
    resolve_vector_notation(routine)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # -- Part A assertions --
    # A1: var(jl,:,:,:,:,ibl) = 0 should resolve 4 implicit dims into 4 loops
    a1_loops = [l for l in loops if l.variable.name.startswith('i_var_')]
    assert len(a1_loops) >= 4, f"Expected at least 4 new loops for var, got {len(a1_loops)}"

    # A2: var(:,:,:,:,:,ibl) = 0 should first resolve horizontal dim to jl,
    # then resolve remaining 4 implicit dims
    # Check that jl loop was created for the horizontal dimension
    jl_loops = [l for l in loops if l.variable.name == 'jl']
    assert len(jl_loops) >= 1, "Expected at least one jl loop"

    # -- Part B assertions --
    # B1: local_var1(:,:,ibl) = 0 — resolve_vector_dimension resolves the
    # horizontal dim (1:nproma) as jl; resolve_vector_notation then resolves
    # the remaining vertical dim (1:klev) as i_local_var1_0.
    b1_loops = [l for l in loops if l.variable.name.startswith('i_local_var1_')]
    assert len(b1_loops) >= 1, f"Expected at least 1 loop for local_var1 vertical dim, got {len(b1_loops)}"
    # The horizontal dim should be resolved via a jl loop
    b1_jl_loops = [l for l in jl_loops
                   if any('local_var1' in str(a.lhs)
                          for a in FindNodes(ir.Assignment).visit(l.body))]
    assert len(b1_jl_loops) >= 1, "Expected jl loop containing local_var1 assignment"

    # B2: Literal list assignment should be unchanged (no loop wrapping)
    generated_loop_bodies = []
    for l in loops:
        if l.variable.name.startswith('i_'):
            generated_loop_bodies.extend(FindNodes(ir.Assignment).visit(l.body))
    b2_assigns = [a for a in assigns
                  if hasattr(a.rhs, 'elements') or 'LiteralList' in type(a.rhs).__name__]
    for b2a in b2_assigns:
        assert b2a not in generated_loop_bodies, \
            "Literal list assignment should not be inside a generated loop"

    # B3: Multi-term RHS arrays should all have loop indices, no RangeIndex left
    b3_assigns = [a for a in assigns
                  if str(a.lhs).startswith('local_var2(') and 'ptr_src' in str(a.rhs)]
    assert len(b3_assigns) >= 1, "Expected B3 assignment"
    for b3a in b3_assigns:
        rhs_arrays = [v for v in FindVariables(unique=False).visit(b3a.rhs)
                      if isinstance(v, sym.Array)]
        for arr in rhs_arrays:
            for dim in arr.dimensions:
                assert not isinstance(dim, sym.RangeIndex), \
                    f"RHS array {arr.name} still has RangeIndex: {dim}"

    # B4: Shifted range should produce offset index expression on RHS
    b4_assigns = [a for a in assigns
                  if str(a.lhs).startswith('local_var4(') and 'shifted_src' in str(a.rhs)]
    assert len(b4_assigns) >= 1, "Expected B4 assignment"
    for b4a in b4_assigns:
        rhs_arrays = [v for v in FindVariables(unique=False).visit(b4a.rhs)
                      if isinstance(v, sym.Array) and v.name.lower() == 'shifted_src']
        for arr in rhs_arrays:
            # shifted_src should NOT have a plain loop index matching LHS;
            # it should have an offset expression (loop_var - lower + rhs_lower)
            for dim in arr.dimensions:
                assert not isinstance(dim, sym.RangeIndex), \
                    f"shifted_src still has RangeIndex: {dim}"

@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_masked_statements(frontend):
    """
    Test resolving of masked statements in kernel.
    """

    fcode = """
subroutine test_resolve_where(start, end, nlon, nz, q, t)
  INTEGER, INTENT(IN) :: start, end  ! Iteration indices
  INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
  REAL, INTENT(INOUT) :: t(nlon,nz)
  REAL, INTENT(INOUT) :: q(nlon,nz)
  INTEGER :: jk

  DO jk = 2, nz
    WHERE (q(start:end, jk) > 1.234)
      q(start:end, jk) = q(start:end, jk-1) + t(start:end, jk)
    ELSEWHERE
      q(start:end, jk) = t(start:end, jk)
    END WHERE
  END DO
end subroutine test_resolve_where
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    horizontal = Dimension(
        name='horizontal', index='jl', lower='start', upper='end'
    )
    resolve_vector_dimension(routine, dimension=horizontal)

    # Ensure horizontal loop variable has been declared
    assert 'jl' in routine.variables

    # Ensure we have three loops in the kernel,
    # horizontal loops should be nested within vertical
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 2
    assert loops[1] in FindNodes(ir.Loop).visit(loops[0].body)
    assert loops[1].variable == 'jl'
    assert loops[1].bounds == 'start:end'
    assert loops[0].variable == 'jk'
    assert loops[0].bounds == '2:nz'

    # Ensure that the respective conditional has been inserted correctly
    conds = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds) == 1
    assert conds[0] in FindNodes(ir.Conditional).visit(loops[1])
    assert conds[0].condition == 'q(jl, jk) > 1.234'

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0] in conds[0].body
    assert assigns[0].lhs == 'q(jl, jk)' and assigns[0].rhs == 'q(jl, jk - 1) + t(jl, jk)'
    assert assigns[1] in conds[0].else_body
    assert assigns[1].lhs == 'q(jl, jk)' and assigns[1].rhs == 't(jl, jk)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_masked_inferred_bounds(frontend):
    """ Test the resolution of WHERE stmts with inferred bounds """

    fcode = """
subroutine test_masked_inferred(n, m, x, y, z)
  implicit none
  integer, intent(in) :: n, m
  real(kind=8), intent(inout) :: x(n), y(n), z(m)
  integer :: i

  do i=1,n
    x(i) = i
  end do
  y(:) = 0.0
  z(:) = 0.0

  where( (x > 5.0) )
    x = y
  end where
end subroutine test_masked_inferred
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    dim = Dimension(name='n', index='i', lower='1', upper='n')
    resolve_vector_dimension(
        routine, dimension=dim, derive_qualified_ranges=True
    )

    # Check only assignments over ``n`` have been resolved
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 4
    assert assigns[0].lhs == 'x(i)' and assigns[0].rhs == 'i'
    assert assigns[1].lhs == 'y(i)' and assigns[1].rhs == '0.0'
    assert assigns[2].lhs == 'z(1:m)' and assigns[2].rhs == '0.0'
    assert assigns[3].lhs == 'x(i)' and assigns[3].rhs == 'y(i)'

    # Check the WHERE has been resolved to IF
    conds = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds) == 1
    assert conds[0].condition == 'x(i) > 5.0'
    assert assigns[3] in conds[0].body

    # Check that new loops have been inserted
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 3
    assert assigns[0] in loops[0].body
    assert assigns[1] in loops[1].body
    assert conds[0] in loops[2].body


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_ifs_patterns(frontend):
    """
    Test vector notation resolution on patterns found in real IFS code,
    including scalar broadcasts, element-wise operations, 1D-to-2D
    broadcasts, partial ranges, fixed third indices, shifted ranges,
    zero-based ranges, and nested full-colon resolution.
    """

    fcode = """
subroutine test_ifs_patterns(kidia, kfdia, klon, klev, klevsn)
  implicit none
  integer, intent(in) :: kidia, kfdia, klon, klev, klevsn
  real :: work(klon), play(klon, klev)
  real :: pssn(klon, klevsn), ptsn(klon, klevsn), pwsn(klon, klevsn)
  real :: pdhtss(klon, klevsn, 15)
  real :: zdsng(klon, 0:klevsn), zremap(klon, klevsn, klevsn)
  real :: ztmpl(klon, klevsn), zsrc(klon, 0:klevsn)
  real :: global_min(klevsn), zmin(klon, klevsn)
  integer :: jl

  ! Pattern 1: Simple scalar broadcast with KIDIA:KFDIA
  work(kidia:kfdia) = 1.0

  ! Pattern 2: 2D element-wise operation inside JL loop
  do jl=kidia,kfdia
    pssn(jl, 1:klevsn) = ptsn(jl, 1:klevsn) / pwsn(jl, 1:klevsn)
  enddo

  ! Pattern 3: 1D-to-2D broadcast inside JL loop
  do jl=kidia,kfdia
    zmin(jl, 1:klevsn) = global_min(1:klevsn)
  enddo

  ! Pattern 4: Partial range (non-1 start)
  do jl=kidia,kfdia
    pssn(jl, 2:klevsn) = 0.0
  enddo

  ! Pattern 5: Multi-dim with fixed third index
  do jl=kidia,kfdia
    pdhtss(jl, 1:klevsn, 1) = 0.0
    pdhtss(jl, 1:klevsn, 2) = 273.15
  enddo

  ! Pattern 6: Shifted/offset range on RHS
  do jl=kidia,kfdia
    ztmpl(jl, 1:klevsn) = zsrc(jl, 0:klevsn-1)
  enddo

  ! Pattern 7: Zero-based range
  do jl=kidia,kfdia
    zdsng(jl, 0:klevsn) = 0.0
  enddo

  ! Pattern 8: Nested 2D full-colon
  do jl=kidia,kfdia
    zremap(jl, :, :) = 0.0
  enddo

end subroutine test_ifs_patterns
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    horizontal = Dimension(
        name='horizontal', index='jl', lower='kidia', upper='kfdia'
    )
    resolve_vector_dimension(routine, dimension=horizontal)
    resolve_vector_notation(routine)

    loops = FindNodes(ir.Loop).visit(routine.body)

    # --- Pattern 1: work(kidia:kfdia) = 1.0 -> DO jl=kidia,kfdia; work(jl)=1.0 ---
    p1_loops = [l for l in loops
                if l.variable.name == 'jl' and str(l.bounds) == 'kidia:kfdia']
    assert len(p1_loops) >= 1, "Pattern 1: expected jl loop with kidia:kfdia bounds"
    p1_assigns = FindNodes(ir.Assignment).visit(p1_loops[0].body)
    assert len(p1_assigns) >= 1
    assert str(p1_assigns[0].lhs) == 'work(jl)'

    # --- Pattern 2: element-wise pssn(jl,1:klevsn) = ptsn/pwsn ---
    # Should create inner loop for dimension 2
    p2_loops = [l for l in loops if l.variable.name.startswith('i_pssn_')]
    assert len(p2_loops) >= 1, "Pattern 2: expected generated loop for pssn"
    p2_assigns = FindNodes(ir.Assignment).visit(p2_loops[0].body)
    assert len(p2_assigns) >= 1
    # LHS and all RHS arrays should use same loop index, no RangeIndex left
    for a in p2_assigns:
        for v in FindVariables(unique=False).visit(a):
            if isinstance(v, sym.Array) and v.name.lower() in ('pssn', 'ptsn', 'pwsn'):
                for dim in v.dimensions:
                    assert not isinstance(dim, sym.RangeIndex), \
                        f"Pattern 2: {v.name} still has RangeIndex {dim}"

    # --- Pattern 3: 1D-to-2D broadcast: zmin(jl,1:klevsn) = global_min(1:klevsn) ---
    p3_loops = [l for l in loops if l.variable.name.startswith('i_zmin_')]
    assert len(p3_loops) >= 1, "Pattern 3: expected generated loop for zmin"
    p3_assigns = FindNodes(ir.Assignment).visit(p3_loops[0].body)
    assert len(p3_assigns) >= 1
    # global_min should use the same loop index
    for a in p3_assigns:
        rhs_arrays = [v for v in FindVariables(unique=False).visit(a.rhs)
                      if isinstance(v, sym.Array) and v.name.lower() == 'global_min']
        for arr in rhs_arrays:
            for dim in arr.dimensions:
                assert not isinstance(dim, sym.RangeIndex), \
                    f"Pattern 3: global_min still has RangeIndex {dim}"

    # --- Pattern 4: Partial range pssn(jl, 2:klevsn) = 0.0 ---
    # Should create loop starting at 2
    p4_loop_names = [l for l in loops if l.variable.name.startswith('i_pssn_')]
    p4_loops_from2 = [l for l in p4_loop_names if '2' in str(l.bounds)]
    assert len(p4_loops_from2) >= 1, "Pattern 4: expected loop with lower bound of 2"

    # --- Pattern 5: pdhtss(jl, 1:klevsn, 1) = 0.0 and pdhtss(jl, 1:klevsn, 2) = 273.15 ---
    # Third index should stay as literal, only dim 2 gets a loop
    p5_loops = [l for l in loops if l.variable.name.startswith('i_pdhtss_')]
    assert len(p5_loops) >= 2, "Pattern 5: expected at least 2 loops for pdhtss"
    for p5l in p5_loops:
        p5_assigns = FindNodes(ir.Assignment).visit(p5l.body)
        for a in p5_assigns:
            # Third dimension should be a literal (1 or 2), not a loop index
            if isinstance(a.lhs, sym.Array) and a.lhs.name.lower() == 'pdhtss':
                assert len(a.lhs.dimensions) == 3
                third_dim = a.lhs.dimensions[2]
                assert not isinstance(third_dim, sym.RangeIndex), \
                    f"Pattern 5: third dim should not be RangeIndex, got {third_dim}"

    # --- Pattern 6: Shifted range ztmpl(jl,1:klevsn) = zsrc(jl,0:klevsn-1) ---
    p6_loops = [l for l in loops if l.variable.name.startswith('i_ztmpl_')]
    assert len(p6_loops) >= 1, "Pattern 6: expected generated loop for ztmpl"
    p6_assigns = FindNodes(ir.Assignment).visit(p6_loops[0].body)
    assert len(p6_assigns) >= 1
    # zsrc should have an offset expression, not a plain loop variable
    for a in p6_assigns:
        rhs_arrays = [v for v in FindVariables(unique=False).visit(a.rhs)
                      if isinstance(v, sym.Array) and v.name.lower() == 'zsrc']
        for arr in rhs_arrays:
            for dim in arr.dimensions:
                assert not isinstance(dim, sym.RangeIndex), \
                    f"Pattern 6: zsrc still has RangeIndex {dim}"

    # --- Pattern 7: Zero-based range zdsng(jl, 0:klevsn) = 0.0 ---
    p7_loops = [l for l in loops if l.variable.name.startswith('i_zdsng_')]
    assert len(p7_loops) >= 1, "Pattern 7: expected generated loop for zdsng"
    # Loop should start at 0
    assert '0' in str(p7_loops[0].bounds), \
        f"Pattern 7: expected 0 in loop bounds, got {p7_loops[0].bounds}"

    # --- Pattern 8: Nested 2D full-colon zremap(jl, :, :) = 0.0 ---
    p8_loops = [l for l in loops if l.variable.name.startswith('i_zremap_')]
    assert len(p8_loops) >= 2, \
        f"Pattern 8: expected 2 nested loops for zremap, got {len(p8_loops)}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_early_exits(frontend):
    """
    Test that certain patterns are correctly skipped by the resolver:
    literal list assignments, SUM intrinsic, and scalar assignments.
    """

    fcode = """
subroutine test_early_exits(n, arr, arr2, scalar)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: arr(n), arr2(n)
  real, intent(inout) :: scalar
  real :: total

  ! Skip: literal list assignment
  arr(1:3) = (/ 1.0, 2.0, 3.0 /)

  ! Skip: SUM intrinsic in RHS
  total = sum(arr(1:n))

  ! Skip: non-array LHS (scalar assignment)
  scalar = arr(1) + arr(2)

  ! Should resolve: normal vector notation
  arr(1:n) = arr2(1:n) + 1.0

end subroutine test_early_exits
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # Literal list assignment should remain unchanged
    literal_assigns = [a for a in assigns
                       if hasattr(a.rhs, 'elements') or 'LiteralList' in type(a.rhs).__name__]
    assert len(literal_assigns) >= 1, "Literal list assignment should still exist"
    # It should NOT be inside any generated loop
    for la in literal_assigns:
        for l in loops:
            assert la not in FindNodes(ir.Assignment).visit(l.body), \
                "Literal list assignment should not be inside a generated loop"

    # SUM assignment should remain unchanged (total = sum(...))
    sum_assigns = [a for a in assigns if str(a.lhs) == 'total']
    assert len(sum_assigns) == 1, "SUM assignment should still exist"
    for l in loops:
        assert sum_assigns[0] not in FindNodes(ir.Assignment).visit(l.body), \
            "SUM assignment should not be inside a generated loop"

    # Scalar assignment should remain unchanged
    scalar_assigns = [a for a in assigns if str(a.lhs) == 'scalar']
    assert len(scalar_assigns) == 1, "Scalar assignment should still exist"
    for l in loops:
        assert scalar_assigns[0] not in FindNodes(ir.Assignment).visit(l.body), \
            "Scalar assignment should not be inside a generated loop"

    # Normal vector notation should be resolved
    resolved_loops = [l for l in loops if l.variable.name.startswith('i_arr_')]
    assert len(resolved_loops) >= 1, "Normal vector notation should be resolved to a loop"
    resolved_assigns = FindNodes(ir.Assignment).visit(resolved_loops[0].body)
    assert len(resolved_assigns) >= 1
    # LHS should have loop index, not RangeIndex
    assert not any(isinstance(d, sym.RangeIndex) for d in resolved_assigns[0].lhs.dimensions), \
        "Resolved assignment LHS should not have RangeIndex"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_broadcast_and_nesting(frontend):
    """
    Test 1D-to-2D broadcasts and nested full-colon resolution,
    including generating multiple nested loops from multi-dim colons.
    """

    fcode = """
subroutine test_broadcast_nesting(n, m, l)
  implicit none
  integer, intent(in) :: n, m, l
  real :: arr1d(n), arr2d(m, n), arr3d(m, n, l)
  integer :: jl

  ! 1D array assigned to 2D slice
  do jl=1,m
    arr2d(jl, 1:n) = arr1d(1:n)
  enddo

  ! Nested full-colon generating nested loops
  do jl=1,m
    arr3d(jl, :, :) = 0.0
  enddo

  ! 2D full-colon outside any explicit loop (both dims get loops)
  arr2d(:, :) = 0.0

end subroutine test_broadcast_nesting
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine)

    loops = FindNodes(ir.Loop).visit(routine.body)

    # --- Case 1: arr2d(jl, 1:n) = arr1d(1:n) ---
    # Should create inner loop for dim 2, RHS and LHS use same index
    c1_loops = [l for l in loops if l.variable.name.startswith('i_arr2d_')]
    assert len(c1_loops) >= 1, "Case 1: expected generated loop for arr2d"
    c1_assigns = FindNodes(ir.Assignment).visit(c1_loops[0].body)
    assert len(c1_assigns) >= 1
    # Both arr2d and arr1d should use the same loop index
    for a in c1_assigns:
        for v in FindVariables(unique=False).visit(a):
            if isinstance(v, sym.Array) and v.name.lower() in ('arr2d', 'arr1d'):
                for dim in v.dimensions:
                    assert not isinstance(dim, sym.RangeIndex), \
                        f"Case 1: {v.name} still has RangeIndex {dim}"

    # --- Case 2: arr3d(jl, :, :) = 0.0 ---
    # Should create two nested inner loops (one per ':' dimension)
    c2_loops = [l for l in loops if l.variable.name.startswith('i_arr3d_')]
    assert len(c2_loops) >= 2, \
        f"Case 2: expected 2 nested loops for arr3d, got {len(c2_loops)}"

    # --- Case 3: arr2d(:, :) = 0.0 outside any explicit loop ---
    # Should create two nested loops (one per dimension)
    # These should be the i_arr2d_ loops that are NOT inside a jl loop
    all_arr2d_loops = [l for l in loops if l.variable.name.startswith('i_arr2d_')]
    assert len(all_arr2d_loops) >= 2, \
        f"Case 3: expected at least 2 total arr2d loops, got {len(all_arr2d_loops)}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_derived_type_existing_scalar(frontend):
    """
    When shape-derived loop bounds reference derived-type members and a scalar
    assignment for those members already exists in the routine (e.g.
    ``KLEVS = KDIM%KLEVS``), the generated loops must use the plain scalar
    rather than the derived-type expression.

    This is the typical IFS pattern: scalar extractions are performed once
    outside the block loop, so the generated inner loops can reference
    device-safe plain scalars.
    """
    fcode = """
subroutine test_dt_existing_scalar(kdim, nb)
  implicit none
  type :: dims_t
    integer :: klon
    integer :: klevs
  end type dims_t
  type(dims_t), intent(in) :: kdim
  integer, intent(in) :: nb
  real :: arr(kdim%klon, kdim%klevs, nb)
  integer :: klon, klevs, ibl

  klon  = kdim%klon
  klevs = kdim%klevs

  do ibl = 1, nb
    arr(:, :, ibl) = 0.0
  enddo
end subroutine test_dt_existing_scalar
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine, substitute_derived_type_bounds=True)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # Two inner loops should be generated (one per ':' dimension)
    inner_loops = [l for l in loops if l.variable.name.startswith('i_arr')]
    assert len(inner_loops) == 2, \
        f"Expected 2 inner loops, got {len(inner_loops)}: {[l.variable.name for l in inner_loops]}"

    # Collect the upper bounds of all generated loops
    upper_bounds = {str(l.bounds.stop).lower() for l in inner_loops}

    # The bounds must use the plain scalars klon and klevs, NOT the
    # derived-type members kdim%klon and kdim%klevs
    assert 'klon' in upper_bounds, \
        f"Expected loop bound 'klon', got: {upper_bounds}"
    assert 'klevs' in upper_bounds, \
        f"Expected loop bound 'klevs', got: {upper_bounds}"
    assert 'kdim%klon' not in upper_bounds, \
        "Loop bound must not reference derived-type member kdim%klon"
    assert 'kdim%klevs' not in upper_bounds, \
        "Loop bound must not reference derived-type member kdim%klevs"

    # No new scalar assignment statements should have been prepended
    # (the existing klon=... and klevs=... are sufficient)
    new_scalar_assigns = [
        a for a in assigns
        if isinstance(a.rhs, sym.Scalar) and a.rhs.parent is not None
        and a.lhs.name.lower() not in ('klon', 'klevs')
    ]
    assert not new_scalar_assigns, \
        f"No extra scalar assignments should be created, got: {new_scalar_assigns}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_derived_type_new_scalar(frontend):
    """
    When shape-derived loop bounds reference derived-type members and NO
    scalar assignment exists for those members, new scalars and corresponding
    assignments must be created and inserted before the generated loop.

    The new scalar takes the name of the derived-type member's basename
    (e.g., ``KLEVS`` for ``KDIM%KLEVS``).
    """
    fcode = """
subroutine test_dt_new_scalar(kdim, nb)
  implicit none
  type :: dims_t
    integer :: klon
    integer :: klevs
  end type dims_t
  type(dims_t), intent(in) :: kdim
  integer, intent(in) :: nb
  real :: arr(kdim%klon, kdim%klevs, nb)
  integer :: ibl

  ! No scalar extractions here — the transformer must create them.
  do ibl = 1, nb
    arr(:, :, ibl) = 0.0
  enddo
end subroutine test_dt_new_scalar
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine, substitute_derived_type_bounds=True)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # Two inner loops should be generated
    inner_loops = [l for l in loops if l.variable.name.startswith('i_arr')]
    assert len(inner_loops) == 2, \
        f"Expected 2 inner loops, got {len(inner_loops)}"

    # Bounds must use plain scalars, not derived-type members
    upper_bounds = {str(l.bounds.stop).lower() for l in inner_loops}
    assert 'kdim%klon' not in upper_bounds, \
        "Loop bound must not reference derived-type member kdim%klon"
    assert 'kdim%klevs' not in upper_bounds, \
        "Loop bound must not reference derived-type member kdim%klevs"

    # New scalar assignment statements should have been created and inserted
    # (of the form  SCALAR = KDIM%MEMBER  immediately before the loop nest)
    new_scalar_assigns = [
        a for a in assigns
        if isinstance(a.rhs, sym.Scalar) and a.rhs.parent is not None
    ]
    assert len(new_scalar_assigns) == 2, \
        f"Expected 2 new scalar assignments, got {len(new_scalar_assigns)}: {new_scalar_assigns}"

    # Each new scalar name must equal the member basename
    new_scalar_names = {str(a.lhs).lower() for a in new_scalar_assigns}
    assert 'klon' in new_scalar_names, \
        f"Expected new scalar 'klon', got: {new_scalar_names}"
    assert 'klevs' in new_scalar_names, \
        f"Expected new scalar 'klevs', got: {new_scalar_names}"

    # The new scalars must be declared in the routine
    declared = {v.name.lower() for v in routine.variables}
    assert 'klon' in declared, "New scalar 'klon' must be declared"
    assert 'klevs' in declared, "New scalar 'klevs' must be declared"

    # Bounds must equal the new scalar names
    assert 'klon' in upper_bounds, \
        f"Expected loop bound 'klon', got: {upper_bounds}"
    assert 'klevs' in upper_bounds, \
        f"Expected loop bound 'klevs', got: {upper_bounds}"

    # Scalar extraction assignments must appear in the routine body BEFORE any loop
    body_nodes = list(routine.body.body)
    first_loop_idx = next(
        (i for i, n in enumerate(body_nodes) if isinstance(n, ir.Loop)), None
    )
    assert first_loop_idx is not None, "Expected at least one loop in routine body"
    for assign in new_scalar_assigns:
        assign_idx = next(
            (i for i, n in enumerate(body_nodes) if n is assign), None
        )
        assert assign_idx is not None, f"Could not locate assignment {assign} in body"
        assert assign_idx < first_loop_idx, (
            f"Scalar extraction assignment {assign} (index {assign_idx}) must appear "
            f"before the first loop (index {first_loop_idx})"
        )


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_derived_type_explicit_range_not_substituted(frontend):
    """
    Explicit source-code ranges that come from loop_map (resolved by
    ``resolve_vector_dimension``) must NOT have their bounds substituted,
    even if those bounds happen to be plain scalars that were themselves
    extracted from a derived type.

    Here a horizontal ``DO jl=kidia,kfdia`` loop already exists, and the
    assignment ``arr(kidia:kfdia) = 1.0`` matches that loop range.
    The resulting ``DO jl=kidia,kfdia`` loop should use the original
    ``kidia``/``kfdia`` scalars unchanged, without any extra substitution.
    """
    fcode = """
subroutine test_dt_no_subst_explicit(dims, arr, n)
  implicit none
  type :: dims_t
    integer :: kidia
    integer :: kfdia
  end type dims_t
  type(dims_t), intent(in) :: dims
  real, intent(inout) :: arr(n)
  integer, intent(in) :: n
  integer :: kidia, kfdia

  kidia = dims%kidia
  kfdia = dims%kfdia

  arr(kidia:kfdia) = 1.0
end subroutine test_dt_no_subst_explicit
    """.strip()
    horizontal = Dimension(name='horizontal', index='jl', lower='kidia', upper='kfdia')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # resolve_vector_dimension handles arr(kidia:kfdia) via loop_map -> jl
    resolve_vector_dimension(routine, dimension=horizontal, derive_qualified_ranges=True)
    resolve_vector_notation(routine, substitute_derived_type_bounds=True)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # Exactly one loop should exist: DO jl = kidia, kfdia
    assert len(loops) == 1, f"Expected exactly 1 loop, got {len(loops)}"
    loop = loops[0]
    assert loop.variable.name.lower() == 'jl', \
        f"Expected loop variable 'jl', got '{loop.variable.name}'"
    assert str(loop.bounds.start).lower() == 'kidia', \
        f"Expected lower bound 'kidia', got '{loop.bounds.start}'"
    assert str(loop.bounds.stop).lower() == 'kfdia', \
        f"Expected upper bound 'kfdia', got '{loop.bounds.stop}'"

    # The bounds must remain as plain scalars (not derived-type members)
    for bound in (loop.bounds.start, loop.bounds.stop):
        bound_vars = FindVariables().visit(bound)
        for v in bound_vars:
            assert v.parent is None, \
                f"Loop bound '{bound}' must not reference a derived-type member, got '{v}'"

    # No extra scalar assignment statements should have been added
    # (only the original kidia=dims%kidia and kfdia=dims%kfdia should exist)
    scalar_assigns = [
        a for a in assigns
        if isinstance(a.rhs, sym.Scalar) and a.rhs.parent is not None
    ]
    assert len(scalar_assigns) == 2, \
        f"Expected exactly 2 scalar assignments (kidia/kfdia extractions), got {len(scalar_assigns)}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_derived_type_name_collision(frontend):
    """
    When the basename of a derived-type member collides with an existing,
    *different* variable in scope, the substitution must be skipped and the
    loop bound should retain the derived-type member expression.

    Here ``klevs`` is declared as ``REAL`` in the routine scope, which is a
    different variable from the INTEGER member ``kdim%klevs``.  The transformer
    must not substitute ``kdim%klevs`` with the ``REAL :: klevs`` variable.
    """
    fcode = """
subroutine test_dt_collision(kdim, nb)
  implicit none
  type :: dims_t
    integer :: klevs
  end type dims_t
  type(dims_t), intent(in) :: kdim
  integer, intent(in) :: nb
  real :: klevs                        ! REAL -- different from INTEGER kdim%klevs
  real :: arr(kdim%klevs, nb)
  integer :: ibl

  do ibl = 1, nb
    arr(:, ibl) = 0.0
  enddo
end subroutine test_dt_collision
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine, substitute_derived_type_bounds=True)

    loops = FindNodes(ir.Loop).visit(routine.body)

    # An inner loop should still be generated for the ':' dimension
    inner_loops = [l for l in loops if l.variable.name.startswith('i_arr')]
    assert len(inner_loops) == 1, \
        f"Expected 1 inner loop, got {len(inner_loops)}"

    # The loop bound must NOT use the REAL klevs variable (name collision);
    # it must retain the derived-type member kdim%klevs.
    upper_bound = str(inner_loops[0].bounds.stop).lower().replace(' ', '')
    assert 'kdim%klevs' in upper_bound, \
        f"Expected loop bound to retain 'kdim%klevs' due to collision, got: {upper_bound}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_derived_type_nested(frontend):
    """
    Regression test for nested derived-type bounds in
    ``_substitute_derived_type_bounds``.

    The array ``arr`` has shape ``(ydg%yrdimv%nflevg, ydm%ndhcvsun)`` where
    ``ydg%yrdimv%nflevg`` is a two-level chain (``ydg`` -> ``yrdimv`` ->
    ``nflevg``) and ``ydm%ndhcvsun`` is a single-level chain.

    **Kernel path** (``substitute_derived_type_bounds=False``, the default):
    no scalar-extraction assignments must be introduced; the generated loop
    bounds must reference the derived-type members directly.

    **Driver path** (``substitute_derived_type_bounds=True``):
    only the *leaf* integer members must be extracted (``nflevg`` and
    ``ndhcvsun``); the intermediate struct ``yrdimv`` must NOT appear as a
    new variable; correct assignments of the form
    ``nflevg = ydg%yrdimv%nflevg`` and ``ndhcvsun = ydm%ndhcvsun`` must be
    present; loop bounds must use the plain scalar names.
    """
    fcode = """
subroutine test_dt_nested(ydg, ydm, nb)
  implicit none
  type :: dimv_t
    integer :: nflevg
  end type dimv_t
  type :: geo_t
    type(dimv_t) :: yrdimv
  end type geo_t
  type :: met_t
    integer :: ndhcvsun
  end type met_t
  type(geo_t),  intent(in) :: ydg
  type(met_t),  intent(in) :: ydm
  integer,      intent(in) :: nb
  real :: arr(ydg%yrdimv%nflevg, ydm%ndhcvsun, nb)
  integer :: ibl

  do ibl = 1, nb
    arr(:, :, ibl) = 0.0
  enddo
end subroutine test_dt_nested
    """.strip()

    # --- Kernel path: no scalar substitution ---
    routine_k = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine_k)  # substitute_derived_type_bounds=False by default

    loops_k = FindNodes(ir.Loop).visit(routine_k.body)
    assigns_k = FindNodes(ir.Assignment).visit(routine_k.body)

    # Two inner loops must be generated
    inner_loops_k = [l for l in loops_k if l.variable.name.startswith('i_arr')]
    assert len(inner_loops_k) == 2, \
        f"[kernel] Expected 2 inner loops, got {len(inner_loops_k)}"

    # No new scalar-extraction assignments (only the original arr(:,:,ibl)=0.0 -> arr(...)=0.0)
    scalar_ext_k = [
        a for a in assigns_k
        if isinstance(a.rhs, sym.Scalar) and a.rhs.parent is not None
    ]
    assert not scalar_ext_k, \
        f"[kernel] No scalar-extraction assignments expected, got: {scalar_ext_k}"

    # Loop bounds must reference derived-type members directly
    upper_bounds_k = {str(l.bounds.stop).lower().replace(' ', '') for l in inner_loops_k}
    assert any('ydg%yrdimv%nflevg' in b for b in upper_bounds_k), \
        f"[kernel] Expected loop bound to reference 'ydg%yrdimv%nflevg', got: {upper_bounds_k}"
    assert any('ydm%ndhcvsun' in b for b in upper_bounds_k), \
        f"[kernel] Expected loop bound to reference 'ydm%ndhcvsun', got: {upper_bounds_k}"

    # --- Driver path: scalar substitution enabled ---
    routine_d = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine_d, substitute_derived_type_bounds=True)

    loops_d = FindNodes(ir.Loop).visit(routine_d.body)
    assigns_d = FindNodes(ir.Assignment).visit(routine_d.body)

    # Two inner loops must be generated
    inner_loops_d = [l for l in loops_d if l.variable.name.startswith('i_arr')]
    assert len(inner_loops_d) == 2, \
        f"[driver] Expected 2 inner loops, got {len(inner_loops_d)}"

    # Exactly two scalar-extraction assignments (nflevg and ndhcvsun)
    scalar_ext_d = [
        a for a in assigns_d
        if isinstance(a.rhs, sym.Scalar) and a.rhs.parent is not None
    ]
    assert len(scalar_ext_d) == 2, \
        f"[driver] Expected 2 scalar-extraction assignments, got: {scalar_ext_d}"

    scalar_lhs_names = {str(a.lhs).lower() for a in scalar_ext_d}
    assert 'nflevg' in scalar_lhs_names, \
        f"[driver] Expected scalar 'nflevg', got: {scalar_lhs_names}"
    assert 'ndhcvsun' in scalar_lhs_names, \
        f"[driver] Expected scalar 'ndhcvsun', got: {scalar_lhs_names}"

    # Intermediate struct yrdimv must NOT appear as a new variable or LHS
    declared_d = {v.name.lower() for v in routine_d.variables}
    assert 'yrdimv' not in declared_d, \
        "[driver] Intermediate struct 'yrdimv' must not be declared as a new variable"

    # Loop bounds must use plain scalar names
    upper_bounds_d = {str(l.bounds.stop).lower().replace(' ', '') for l in inner_loops_d}
    assert any('nflevg' in b for b in upper_bounds_d), \
        f"[driver] Expected loop bound 'nflevg', got: {upper_bounds_d}"
    assert any('ndhcvsun' in b for b in upper_bounds_d), \
        f"[driver] Expected loop bound 'ndhcvsun', got: {upper_bounds_d}"
    assert not any('ydg%' in b or 'ydm%' in b for b in upper_bounds_d), \
        f"[driver] Loop bounds must not reference derived-type members, got: {upper_bounds_d}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_no_implicit_rhs(frontend):
    """
    When ``resolve_implicit_rhs_ranges=False`` is passed to
    ``resolve_vector_dimension``, assignments whose RHS arrays use bare ``:``
    ranges (without explicit bounds) must NOT be resolved.

    Only assignments where all RHS ranges are explicit should be resolved.
    """
    fcode = """
subroutine test_no_implicit_rhs(start, end, n, a, b, c)
  implicit none
  integer, intent(in) :: start, end, n
  real, intent(inout) :: a(n), b(n), c(n)

  ! RHS uses bare ':' -- should NOT be resolved with resolve_implicit_rhs_ranges=False
  a(start:end) = b(:)

  ! Both sides use explicit range -- SHOULD be resolved
  a(start:end) = c(start:end)

end subroutine test_no_implicit_rhs
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    dim = Dimension(name='horizontal', index='jl', lower='start', upper='end')
    resolve_vector_dimension(routine, dimension=dim, resolve_implicit_rhs_ranges=False)

    loops = FindNodes(ir.Loop).visit(routine.body)
    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # Exactly one loop should be generated (for the explicit-range assignment only)
    assert len(loops) == 1, \
        f"Expected exactly 1 loop (for explicit-range assignment), got {len(loops)}"

    # The unresolved assignment (bare RHS ':') should remain as a vector assignment
    unresolved = [a for a in assigns if isinstance(a.lhs, sym.Array)
                  and any(isinstance(d, sym.RangeIndex) for d in a.lhs.dimensions)]
    assert len(unresolved) >= 1, \
        "Assignment with bare RHS ':' should remain unresolved (LHS still has RangeIndex)"

    # The resolved loop should contain the explicit-range assignment (c)
    loop_assigns = FindNodes(ir.Assignment).visit(loops[0].body)
    assert len(loop_assigns) == 1
    assert 'c(' in str(loop_assigns[0].rhs), \
        f"Expected resolved c(...) in loop body, got: {loop_assigns[0].rhs}"

    # --- Part 2: range dimension NOT in the first position ---
    # Ensures that qualified-position tracking works when the range
    # dimension sits at a non-leading position in the index tuple.
    fcode2 = """
subroutine test_no_implicit_rhs_pos(start, end, m, n, a, b, c)
  implicit none
  integer, intent(in) :: start, end, m, n
  real, intent(inout) :: a(m, n), b(m, n), c(m, n)

  ! RHS bare ':' in second dim -- should NOT be resolved
  a(1, start:end) = b(1, :)

  ! Both sides explicit in second dim -- SHOULD be resolved
  a(1, start:end) = c(1, start:end)

end subroutine test_no_implicit_rhs_pos
    """.strip()
    routine2 = Subroutine.from_source(fcode2, frontend=frontend)
    dim2 = Dimension(name='horizontal', index='jl', lower='start', upper='end')
    resolve_vector_dimension(routine2, dimension=dim2, resolve_implicit_rhs_ranges=False)

    loops2 = FindNodes(ir.Loop).visit(routine2.body)
    assigns2 = FindNodes(ir.Assignment).visit(routine2.body)

    # Exactly one loop for the explicit-range assignment
    assert len(loops2) == 1, \
        f"Expected exactly 1 loop, got {len(loops2)}"

    # The unresolved assignment should still have a RangeIndex
    unresolved2 = [a for a in assigns2 if isinstance(a.lhs, sym.Array)
                   and any(isinstance(d, sym.RangeIndex) for d in a.lhs.dimensions)]
    assert len(unresolved2) >= 1, \
        "Assignment with bare RHS ':' should remain unresolved"

    # The resolved loop body should reference c
    loop_assigns2 = FindNodes(ir.Assignment).visit(loops2[0].body)
    assert len(loop_assigns2) == 1
    assert 'c(' in str(loop_assigns2[0].rhs), \
        f"Expected resolved c(...) in loop body, got: {loop_assigns2[0].rhs}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_inline_conditional(frontend):
    """
    When an inline conditional (single-line ``IF``) contains a vector
    notation assignment, resolving it wraps the body in a loop.  The
    conditional must be demoted from ``inline=True`` to ``inline=False``
    so that the backend does not attempt to format it as a single line.
    """
    fcode = """
subroutine test_inline_cond(n, flag, a, b)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: flag
  real, intent(inout) :: a(n), b(n)

  if (flag) a(1:n) = b(1:n)

end subroutine test_inline_cond
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Before resolution: should have an inline conditional
    # (OMNI does not preserve the inline flag, so only check for fparser)
    conds_before = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds_before) == 1
    if frontend != OMNI:
        assert conds_before[0].inline is True

    resolve_vector_notation(routine)

    # After resolution: the conditional should no longer be inline
    conds_after = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds_after) == 1
    assert conds_after[0].inline is False

    # The body should contain a loop wrapping the assignment
    loops = FindNodes(ir.Loop).visit(conds_after[0].body)
    assert len(loops) == 1

    # The generated Fortran should be valid (block IF, not inline)
    fcode_out = routine.to_fortran()
    assert 'IF (flag) THEN' in fcode_out
    assert 'END IF' in fcode_out


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_dimension_empty_bounds_warning(frontend, caplog):
    """
    When ``resolve_vector_dimension`` is called with a dimension whose
    lower/upper bounds are not present in the routine's scope, it should
    emit a warning and leave the routine unchanged.
    """
    from loki.logging import WARNING  # pylint: disable=import-outside-toplevel

    fcode = """
subroutine test_empty_bounds(n, a)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n)

  a(1:n) = 0.0

end subroutine test_empty_bounds
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Use a dimension whose bounds ('xstart', 'xend') don't exist in the routine
    dim = Dimension(name='horizontal', index='jl', lower='xstart', upper='xend')

    caplog.set_level(WARNING)
    resolve_vector_dimension(routine, dimension=dim)

    # The routine body should be unchanged -- the range is still present
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 1
    assert any(isinstance(d, sym.RangeIndex) for d in assigns[0].lhs.dimensions), \
        "Assignment should remain unresolved when no valid bounds are found"

    # A warning should have been emitted
    assert any('No valid loop bounds' in r.message for r in caplog.records), \
        f"Expected a 'No valid loop bounds' warning, got: {[r.message for r in caplog.records]}"


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_shifted_range_in_intrinsic(frontend):
    """
    Regression test: when a vertical range on the RHS is shifted relative to
    the LHS range (e.g. ``zdsng(jl, 0:klevsn-1)`` vs LHS ``1:klevsn``), both
    ``resolve_vector_notation`` alone and the two-step pipeline
    ``resolve_vector_dimension`` + ``resolve_vector_notation`` must produce the
    same, correct result.

    Before the fix, the two-step pipeline would leave a corrupt ``RangeIndex``
    embedded in arithmetic (e.g. ``-1 + (1:klevsn)``) instead of substituting
    the loop index variable.
    """
    fcode = """
subroutine test_shifted_intrinsic(jl, kidia, kfdia, klevsn, jk, ztmpl1, zdsnnewg, zdsng)
  implicit none
  integer, intent(in) :: kidia, kfdia, klevsn, jl, jk
  real, intent(inout) :: ztmpl1(kfdia, klevsn)
  real, intent(in)    :: zdsnnewg(kfdia, klevsn)
  real, intent(in)    :: zdsng(kfdia, 0:klevsn)
  do jl = kidia, kfdia
    ztmpl1(jl, 1:klevsn) = MAX(zdsnnewg(jl, jk-1), zdsng(jl, 0:klevsn-1))
  end do
end subroutine test_shifted_intrinsic
    """.strip()

    # --- Approach 1: resolve_vector_dimension + resolve_vector_notation ---
    routine1 = Subroutine.from_source(fcode, frontend=frontend)
    dim = Dimension(name='horizontal', index='jl', lower='kidia', upper='kfdia')
    resolve_vector_dimension(routine1, dimension=dim, derive_qualified_ranges=True)
    resolve_vector_notation(routine1)

    # --- Approach 2: resolve_vector_notation only ---
    routine2 = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine2)

    for label, routine in [('pipeline', routine1), ('notation-only', routine2)]:
        loops = FindNodes(ir.Loop).visit(routine.body)
        # Outer jl-loop + inner vertical loop
        assert len(loops) == 2, f'[{label}] Expected 2 loops, got {len(loops)}'

        inner_loop = loops[1]
        assigns = FindNodes(ir.Assignment).visit(inner_loop.body)
        assert len(assigns) == 1, f'[{label}] Expected 1 assignment in inner loop'

        assign = assigns[0]
        rhs_str = str(assign.rhs)

        # The zdsng dimension must NOT contain a RangeIndex (e.g. '1:klevsn')
        rhs_arrays = [
            v for v in FindVariables(unique=False).visit(assign.rhs)
            if isinstance(v, sym.Array)
        ]
        zdsng_arr = next((a for a in rhs_arrays if 'zdsng' in a.name.lower()), None)
        assert zdsng_arr is not None, f'[{label}] zdsng not found in RHS arrays'
        assert not any(isinstance(d, sym.RangeIndex) for d in zdsng_arr.dimensions), \
            f'[{label}] zdsng still has RangeIndex dim after resolution: {zdsng_arr.dimensions}'

        # The zdsng index should be the loop variable offset by -1 (not a plain range)
        zdsng_dim1 = zdsng_arr.dimensions[1]
        dim1_str = str(zdsng_dim1).replace(' ', '')
        assert 'RangeIndex' not in type(zdsng_dim1).__name__, \
            f'[{label}] zdsng second dim is still a RangeIndex: {zdsng_dim1}'
        assert any(s in dim1_str for s in ['-1+', '+-1', 'i_', '-1']), \
            f'[{label}] expected offset index for zdsng, got: {zdsng_dim1}'

        # No corrupt range expressions anywhere in the body
        all_vars = FindVariables(unique=False).visit(inner_loop)
        assert not any(
            isinstance(v, sym.Array) and any(isinstance(d, sym.RangeIndex) for d in v.dimensions)
            for v in all_vars
        ), f'[{label}] Found arrays with remaining RangeIndex dims inside inner loop:\n{rhs_str}'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI cannot parse unresolved external function beta2alpha')]
))
def test_resolve_vector_notation_shifted_range_multi_arg(frontend):
    """
    Regression test for the ``overlap_alpha = beta2alpha(...)`` pattern where
    a function is called with multiple shifted array arguments:

    .. code-block:: fortran

       overlap_alpha(jcol, 1:nlev-1) = beta2alpha(overlap_param(jcol,:), &
            frac(jcol, 1:nlev-1), frac(jcol, 2:nlev))

    ``frac(jcol, 2:nlev)`` has its second dim shifted by +1 relative to
    ``1:nlev-1``.  Both the two-step pipeline and ``resolve_vector_notation``
    alone must produce ``frac(jcol, 1 + i_...)`` (not ``1 + (1:nlev-1)``).
    """
    fcode = """
subroutine test_shifted_multi(jcol, kstart, kend, nlev, overlap_alpha, overlap_param, frac)
  implicit none
  integer, intent(in) :: kstart, kend, nlev, jcol
  real, intent(inout) :: overlap_alpha(kend, nlev-1)
  real, intent(in)    :: overlap_param(kend, nlev)
  real, intent(in)    :: frac(kend, nlev)
  do jcol = kstart, kend
    overlap_alpha(jcol,1:nlev-1) = beta2alpha(overlap_param(jcol,:), &
         frac(jcol,1:nlev-1), frac(jcol,2:nlev))
  end do
end subroutine test_shifted_multi
    """.strip()

    # --- Approach 1: resolve_vector_dimension + resolve_vector_notation ---
    routine1 = Subroutine.from_source(fcode, frontend=frontend)
    dim = Dimension(name='horizontal', index='jcol', lower='kstart', upper='kend')
    resolve_vector_dimension(routine1, dimension=dim, derive_qualified_ranges=True)
    resolve_vector_notation(routine1)

    # --- Approach 2: resolve_vector_notation only ---
    routine2 = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine2)

    for label, routine in [('pipeline', routine1), ('notation-only', routine2)]:
        loops = FindNodes(ir.Loop).visit(routine.body)
        assert len(loops) == 2, f'[{label}] Expected 2 loops, got {len(loops)}'

        inner_loop = loops[1]
        assigns = FindNodes(ir.Assignment).visit(inner_loop.body)
        assert len(assigns) == 1

        assign = assigns[0]
        all_vars = FindVariables(unique=False).visit(assign.rhs)

        # No array in the RHS should still carry a RangeIndex dimension
        rhs_range_arrays = [
            v for v in all_vars
            if isinstance(v, sym.Array)
            and any(isinstance(d, sym.RangeIndex) for d in v.dimensions)
        ]
        assert not rhs_range_arrays, \
            f'[{label}] RHS arrays still have RangeIndex dims: {rhs_range_arrays}'

        # frac(jcol, 2:nlev) should have become frac(jcol, 1 + ivar)
        frac_arrs = [v for v in all_vars if isinstance(v, sym.Array) and 'frac' in v.name.lower()]
        assert len(frac_arrs) == 2, \
            f'[{label}] Expected 2 frac array references, got {frac_arrs}'

        frac_dims = [str(a.dimensions[1]).replace(' ', '') for a in frac_arrs]
        # One frac should be accessed with the plain loop index, the other with +1 offset
        assert any('1+' in d or '+1' in d for d in frac_dims), \
            f'[{label}] Expected one frac dim to have +1 offset, got: {frac_dims}'


@pytest.mark.parametrize('frontend', available_frontends())
def test_resolve_vector_notation_duplicate_dimensions(frontend):
    """
    Regression test: when an array has two range dimensions that map to the
    same loop variable (because both dimensions have equal extent), each
    position must receive its own distinct loop index variable.

    Example:  ``ZREMAP(KLON, KLEVSN, KLEVSN)`` with ``DO JK=1,KLEVSN``
    already in scope.  The assignment ``ZREMAP(JL,:,:)=0.0`` must produce
    two nested inner loops -- one for each ``:`` -- with **different** loop
    variables.  Before the fix both positions would silently reuse ``JK``
    and produce ``ZREMAP(JL,JK,JK)`` with only one inner loop.
    """
    fcode = """
subroutine test_dup_dims(klon, klevsn, zremap)
  implicit none
  integer, intent(in) :: klon, klevsn
  real, intent(inout) :: zremap(klon, klevsn, klevsn)
  integer :: jl, jk

  do jl = 1, klon
    ! Sibling loop that introduces JK -> 1:KLEVSN in loop_map
    do jk = 1, klevsn
      zremap(jl, jk, 1) = 0.0
    enddo
    ! Two bare ':' dimensions -- both map to range 1:KLEVSN, but each
    ! must get its own loop variable (not both reuse JK).
    zremap(jl, :, :) = 0.0
  enddo

end subroutine test_dup_dims
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)
    resolve_vector_notation(routine)

    assigns = FindNodes(ir.Assignment).visit(routine.body)

    # Find the resolved zremap assignment (no more RangeIndex dimensions)
    resolved_assigns = [
        a for a in assigns
        if isinstance(a.lhs, sym.Array)
        and 'zremap' in a.lhs.name.lower()
        and not any(isinstance(d, sym.RangeIndex) for d in a.lhs.dimensions)
        and len(a.lhs.dimensions) == 3
        and str(a.lhs.dimensions[2]).lower() != '1'  # not the original zremap(jl,jk,1)
    ]
    assert len(resolved_assigns) == 1, \
        f"Expected exactly 1 resolved zremap assignment, got {len(resolved_assigns)}: {resolved_assigns}"

    assign = resolved_assigns[0]
    dims = assign.lhs.dimensions
    # The two inner dimensions (dims[1] and dims[2]) must be distinct scalar variables
    assert str(dims[1]).lower() != str(dims[2]).lower(), (
        f"Both inner dimensions of zremap are identical ({dims[1]!s}), "
        f"indicating only one loop variable was used for two distinct dimensions; "
        f"expected two distinct loop variables"
    )
