# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# pylint: disable=too-many-lines
import itertools
import pytest
import numpy as np

from loki import Subroutine
from loki.jit_build import jit_compile, clean_test
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.ir import (
    is_loki_pragma, pragmas_attached, FindNodes, Loop, Conditional,
    Assignment, FindVariables, nodes as ir
)

from loki.transformations.transform_loop import (
    do_loop_interchange, do_loop_fusion, do_loop_fission, do_loop_unroll,
    TransformLoopsTransformation
)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_interchange_plain(tmp_path, frontend):
    """
    Apply loop interchange for two loops without further arguments.
    """
    fcode = """
subroutine transform_loop_interchange_plain(a, m, n)
  integer, intent(out) :: a(m, n)
  integer, intent(in) :: m, n
  integer :: i, j

  !$loki loop-interchange
  do i=1,n
    do j=1,m
      a(j, i) = i + j
    end do
  end do

  ! This loop is to make sure everything else stays as is
  do i=1,n
    do j=1,m
      a(j, i) = a(j, i) - 2
    end do
  end do
end subroutine transform_loop_interchange_plain
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    m, n = 10, 20
    ref = np.array([[i+j for i in range(n)] for j in range(m)], order='F')

    # Test the reference solution
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 4
    assert [str(loop.variable) for loop in loops] == ['i', 'j', 'i', 'j']

    a = np.zeros(shape=(m, n), dtype=np.int32, order='F')
    function(a=a, m=m, n=n)
    assert np.all(a == ref)

    # Apply transformation
    do_loop_interchange(routine)

    interchanged_filepath = tmp_path/(f'{routine.name}_interchanged_{frontend}.f90')
    interchanged_function = jit_compile(routine, filepath=interchanged_filepath, objname=routine.name)

    # Test transformation
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 4
    assert [str(loop.variable) for loop in loops] == ['j', 'i', 'i', 'j']

    a = np.zeros(shape=(m, n), dtype=np.int32, order='F')
    interchanged_function(a=a, m=m, n=n)
    assert np.all(a == ref)

    clean_test(filepath)
    clean_test(interchanged_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_interchange(tmp_path, frontend):
    """
    Apply loop interchange for three loops with specified order.
    """
    fcode = """
subroutine transform_loop_interchange(a, m, n, nclv)
  integer, intent(out) :: a(m, n, nclv)
  integer, intent(in) :: m, n, nclv
  integer :: i, j, k

!$loki loop-interchange (j, i, k)
!$loki some-pragma
  do k=1,nclv
!$loki more-pragma
    do i=1,n
!$loki other-pragma
      do j=1,m
        a(j, i, k) = i + j + k
      end do
    end do
  end do

  ! This loop is to make sure everything else stays as is
  do k=1,nclv
    do i=1,n
      do j=1,m
        a(j, i, k) = a(j, i, k) - 3
      end do
    end do
  end do
end subroutine transform_loop_interchange
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    m, n, nclv = 10, 20, 5
    ref = np.array([[[i+j+k for k in range(nclv)] for i in range(n)] for j in range(m)], order='F')

    # Test the reference solution
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['k', 'i', 'j', 'k', 'i', 'j']
    with pragmas_attached(routine, Loop):
        assert is_loki_pragma(loops[0].pragma, starts_with='some-pragma')
        assert is_loki_pragma(loops[1].pragma, starts_with='more-pragma')
        assert is_loki_pragma(loops[2].pragma, starts_with='other-pragma')

    a = np.zeros(shape=(m, n, nclv), dtype=np.int32, order='F')
    function(a=a, m=m, n=n, nclv=nclv)
    assert np.all(a == ref)

    # Apply transformation
    do_loop_interchange(routine)

    interchanged_filepath = tmp_path/(f'{routine.name}_interchanged_{frontend}.f90')
    interchanged_function = jit_compile(routine, filepath=interchanged_filepath, objname=routine.name)

    # Test transformation
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert [str(loop.variable) for loop in loops] == ['j', 'i', 'k', 'k', 'i', 'j']

    # Make sure other pragmas remain in place
    with pragmas_attached(routine, Loop):
        assert is_loki_pragma(loops[0].pragma, starts_with='some-pragma')
        assert is_loki_pragma(loops[1].pragma, starts_with='more-pragma')
        assert is_loki_pragma(loops[2].pragma, starts_with='other-pragma')

    a = np.zeros(shape=(m, n, nclv), dtype=np.int32, order='F')
    interchanged_function(a=a, m=m, n=n, nclv=nclv)
    assert np.all(a == ref)

    clean_test(filepath)
    clean_test(interchanged_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_interchange_project(tmp_path, frontend):
    """
    Apply loop interchange for two loops with bounds projection.
    """
    fcode = """
subroutine transform_loop_interchange_project(a, m, n)
  integer, intent(inout) :: a(m, n)
  integer, intent(in) :: m, n
  integer :: i, j

  !$loki loop-interchange
  do i=1,n
    do j=i,m
      a(j, i) = i + j
    end do
  end do
end subroutine transform_loop_interchange_project
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    m, n = 10, 20
    ref = np.array([[i+j if j>=i else 0 for i in range(1, n+1)]
                    for j in range(1, m+1)], order='F')

    # Test the reference solution
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert [str(loop.variable) for loop in loops] == ['i', 'j']

    a = np.zeros(shape=(m, n), dtype=np.int32, order='F')
    function(a=a, m=m, n=n)
    assert np.all(a == ref)

    # Apply transformation
    do_loop_interchange(routine, project_bounds=True)

    interchanged_filepath = tmp_path/(f'{routine.name}_interchanged_{frontend}.f90')
    interchanged_function = jit_compile(routine, filepath=interchanged_filepath, objname=routine.name)

    # Test transformation
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert [str(loop.variable) for loop in loops] == ['j', 'i']

    a = np.zeros(shape=(m, n), dtype=np.int32, order='F')
    interchanged_function(a=a, m=m, n=n)
    assert np.all(a == ref)

    clean_test(filepath)
    clean_test(interchanged_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('insert_loc', (False, True))
def test_transform_loop_fuse_ordering(frontend, insert_loc):
    """
    Apply loop fusion for two loops with matching iteration spaces.
    """
    fcode = f"""
subroutine transform_loop_fuse_ordering(a, b, c, n, m)
  integer, intent(out) :: a(m, n), b(m, n), c(m)
  integer, intent(in) :: n, m
  integer :: i

  !$loki loop-fusion group(1)
  !$loki loop-interchange
  do j=1,m
    do i=1,n
      a(j, i) = i + j
    enddo
  end do

  !$loki loop-fusion group(1)
  do i=1,n
    do j=1,m
      a(j, i) = i + j
    enddo
  end do

  do j=1,m
    c(j) = j
  enddo

  !$loki loop-fusion group(1) {'insert-loc' if insert_loc else ''}
  do i=1,n-1
    do j=1,m
      b(j, i) = n-i+1 + j
    enddo
  end do
end subroutine transform_loop_fuse_ordering
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(FindNodes(Loop).visit(routine.body)) == 7
    do_loop_interchange(routine)
    do_loop_fusion(routine)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 5
    loop_0_vars = [var.name.lower() for var in FindVariables().visit(loops[0].body)]
    if insert_loc:
        assert loops[0].variable.name.lower() == 'j'
        assert 'c' in loop_0_vars
    else:
        assert loops[0].variable.name.lower() == 'i'
        assert 'c' not in loop_0_vars

@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_matching(tmp_path, frontend):
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
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
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
    do_loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fused_function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_subranges(tmp_path, frontend):
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
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
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
    do_loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fused_function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_groups(tmp_path, frontend):
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
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
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
    do_loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 2

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
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


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_failures(frontend):
    """
    Test that loop-fusion fails for known mistakes.
    """
    fcode = """
subroutine transform_loop_fuse_failures(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: i

  !$loki loop-fusion group(1) range(1:n)
  do i=1,n
    a(i) = i
  end do

  !$loki loop-fusion group(1) range(0:n-1)
  do i=0,n-1
    b(i+1) = n-i
  end do
end subroutine transform_loop_fuse_failures
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    with pytest.raises(RuntimeError):
        do_loop_fusion(routine)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_alignment(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fuse_alignment(a, b, n)
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
end subroutine transform_loop_fuse_alignment
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
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
    do_loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fused_function(a=a, b=b, n=n)
    assert np.all(a == range(1, n+1))
    assert np.all(b == range(n, 0, -1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_nonmatching_lower(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fuse_nonmatching_lower(a, b, nclv, klev)
  integer, intent(out) :: a(klev), b(klev)
  integer, intent(in) :: nclv, klev
  integer :: jl

  !$loki loop-fusion group(1)
  do jl=1,klev
    a(jl) = jl
  end do

  !$loki loop-fusion group(1)
  do jl=nclv,klev
    b(jl) = jl - nclv
  end do
end subroutine transform_loop_fuse_nonmatching_lower
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klev, nclv = 100, 15
    a = np.zeros(shape=(klev,), dtype=np.int32)
    b = np.zeros(shape=(klev,), dtype=np.int32)
    function(a=a, b=b, klev=klev, nclv=nclv)
    assert np.all(a == range(1, klev+1))
    assert np.all(b[nclv:klev+1] == range(1, klev-nclv+1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert isinstance(loops[0].bounds.start, sym.InlineCall) and loops[0].bounds.start.name == 'min'
    assert loops[0].bounds.stop == 'klev'
    assert len(FindNodes(Conditional).visit(routine.body)) == 2

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    klev, nclv = 100, 15
    a = np.zeros(shape=(klev,), dtype=np.int32)
    b = np.zeros(shape=(klev,), dtype=np.int32)
    fused_function(a=a, b=b, klev=klev, nclv=nclv)
    assert np.all(a == range(1, klev+1))
    assert np.all(b[nclv:klev+1] == range(1, klev-nclv+1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_nonmatching_lower_annotated(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fuse_nonmatching_lower_annotated(a, b, nclv, klev)
  integer, intent(out) :: a(klev), b(klev)
  integer, intent(in) :: nclv, klev
  integer :: jl

  !$loki loop-fusion group(1)
  do jl=1,klev
    a(jl) = jl
  end do

  !$loki loop-fusion group(1) range(1:klev)
  do jl=nclv,klev
    b(jl) = jl - nclv
  end do
end subroutine transform_loop_fuse_nonmatching_lower_annotated
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klev, nclv = 100, 15
    a = np.zeros(shape=(klev,), dtype=np.int32)
    b = np.zeros(shape=(klev,), dtype=np.int32)
    function(a=a, b=b, klev=klev, nclv=nclv)
    assert np.all(a == range(1, klev+1))
    assert np.all(b[nclv:klev+1] == range(1, klev-nclv+1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].bounds.start == '1'
    assert loops[0].bounds.stop == 'klev'
    assert len(FindNodes(Conditional).visit(routine.body)) == 1

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    klev, nclv = 100, 15
    a = np.zeros(shape=(klev,), dtype=np.int32)
    b = np.zeros(shape=(klev,), dtype=np.int32)
    fused_function(a=a, b=b, klev=klev, nclv=nclv)
    assert np.all(a == range(1, klev+1))
    assert np.all(b[nclv:klev+1] == range(1, klev-nclv+1))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_nonmatching_upper(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fuse_nonmatching_upper(a, b, klev)
  integer, intent(out) :: a(klev), b(klev+1)
  integer, intent(in) :: klev
  integer :: jl

  !$loki loop-fusion group(1)
  do jl=1,klev
    a(jl) = jl
  end do

  !$loki loop-fusion group(1)
  do jl=1,klev+1
    b(jl) = 2*jl
  end do
end subroutine transform_loop_fuse_nonmatching_upper
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klev = 100
    a = np.zeros(shape=(klev,), dtype=np.int32)
    b = np.zeros(shape=(klev+1,), dtype=np.int32)
    function(a=a, b=b, klev=klev)
    assert np.all(a == range(1, klev+1))
    assert np.all(b == np.array(list(range(1, klev+2))) * 2)

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].bounds.start == '1'
    assert loops[0].bounds.stop == '1 + klev'
    assert len(FindNodes(Conditional).visit(routine.body)) == 1

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    klev = 100
    a = np.zeros(shape=(klev,), dtype=np.int32)
    b = np.zeros(shape=(klev+1,), dtype=np.int32)
    fused_function(a=a, b=b, klev=klev)
    assert np.all(a == range(1, klev+1))
    assert np.all(b == np.array(list(range(1, klev+2))) * 2)

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_collapse(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fuse_collapse(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl

!$loki loop-fusion collapse(2)
  do jk=1,klev
    do jl=1,klon
      a(jl, jk) = jk
    end do
  end do

!$loki loop-fusion collapse(2)
  do jk=1,klev
    do jl=1,klon
      b(jl, jk) = jl + jk
    end do
  end do
end subroutine transform_loop_fuse_collapse
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+1))] * klon, order='F'))
    assert np.all(b == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 4
    do_loop_fusion(routine)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == 'klev' for loop in loops) == 1
    assert sum(loop.bounds.stop == 'klon' for loop in loops) == 1

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    fused_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+1))] * klon, order='F'))
    assert np.all(b == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_collapse_nonmatching(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fuse_collapse_nonmatching(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev+1), b(klon+1, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl

!$loki loop-fusion collapse(2)
  do jk=1,klev+1
    do jl=1,klon
      a(jl, jk) = jk
    end do
  end do

!$loki loop-fusion collapse(2)
  do jk=1,klev
    do jl=1,klon+1
      b(jl, jk) = jl + jk
    end do
  end do
end subroutine transform_loop_fuse_collapse_nonmatching
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev+1), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon+1, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+2))] * klon, order='F'))
    assert np.all(b == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+2)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 4
    do_loop_fusion(routine)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == '1 + klev' for loop in loops) == 1
    assert sum(loop.bounds.stop == '1 + klon' for loop in loops) == 1
    assert len(FindNodes(Conditional).visit(routine.body)) == 2

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev+1), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon+1, klev), order='F', dtype=np.int32)
    fused_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+2))] * klon, order='F'))
    assert np.all(b == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+2)], order='F'))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_collapse_range(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fuse_collapse_range(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev+1), b(klon+1, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl, start = 15

!$loki loop-fusion collapse(2)
  do jk=1,klev+1
    do jl=1,klon
      a(jl, jk) = jk
    end do
  end do

!$loki loop-fusion collapse(2) range(1:1+klev,1:klon+1)
  do jk=start,klev
    do jl=1,klon+1
      b(jl, jk) = jl + jk
    end do
  end do
end subroutine transform_loop_fuse_collapse_range
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev+1), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon+1, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+2))] * klon, order='F'))
    assert np.all(b[..., 14:] == np.array([[jl + jk for jk in range(15, klev+1)]
                                           for jl in range(1, klon+2)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 4
    do_loop_fusion(routine)
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == '1 + klev' for loop in loops) == 1
    assert sum(loop.bounds.stop == 'klon + 1' for loop in loops) == 1
    assert len(FindNodes(Conditional).visit(routine.body)) == 2

    fused_filepath = tmp_path/(f'{routine.name}_fused_{frontend}.f90')
    fused_function = jit_compile(routine, filepath=fused_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev+1), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon+1, klev), order='F', dtype=np.int32)
    fused_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+2))] * klon, order='F'))
    assert np.all(b[..., 14:] == np.array([[jl + jk for jk in range(15, klev+1)]
                                           for jl in range(1, klon+2)], order='F'))

    clean_test(filepath)
    clean_test(fused_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_single(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_single(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: j

  do j=1,n
    a(j) = j
    !$loki loop-fission
    b(j) = n-j
  end do
end subroutine transform_loop_fission_single
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    for loop in loops:
        assert loop.bounds.start == '1'
        assert loop.bounds.stop == 'n'

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fissioned_function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_nested(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_nested(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: j, k

  do j=1,n+1
    if (j <= n) then
      a(j) = j
!$loki loop-fission
      b(j) = n-j
    end if
  end do
end subroutine transform_loop_fission_nested
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    assert len(FindNodes(Conditional).visit(routine.body)) == 1
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    for loop in loops:
        assert loop.bounds.start == '1'
        assert loop.bounds.stop == 'n + 1'
    assert len(FindNodes(Conditional).visit(routine.body)) == 2

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fissioned_function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_nested_promote(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_nested_promote(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: j, k, zqxfg(5)

  do j=1,n+1
    zqxfg(2) = j
!$loki loop-fission promote(zqxfg)
    if (j <= n) then
      if (zqxfg(2) <= n) then
        a(j) = zqxfg(2)
        b(j) = n-zqxfg(2)
      end if
    end if
  end do
end subroutine transform_loop_fission_nested_promote
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    assert len(FindNodes(Conditional).visit(routine.body)) == 2
    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    for loop in loops:
        assert loop.bounds.start == '1'
        assert loop.bounds.stop == 'n + 1'
    assert len(FindNodes(Conditional).visit(routine.body)) == 2
    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assert all(d == ref for d, ref in zip(routine.variable_map['zqxfg'].shape, ['5', '1 + n']))

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fissioned_function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_collapse(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_collapse(a, n)
  integer, intent(out) :: a(n, n+1)
  integer, intent(in) :: n
  integer :: j, k, tmp, tmp2

  tmp = 0
  do j=1,n+1
    tmp = j
    tmp2 = 0
!$loki loop-fission promote(tmp)
    do k=1,n
      tmp2 = tmp + k
!$loki loop-fission collapse(2) promote(tmp2)
      a(k, j) = tmp2
!$loki loop-fission
      a(k, j) = a(k, j) - 1
!$loki loop-fission collapse(2)
      a(k, j) = -1 + a(k, j)
    end do
    tmp = 0
  end do
end subroutine transform_loop_fission_collapse
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 11
    a = np.zeros(shape=(n, n+1), order='F', dtype=np.int32)
    function(a=a, n=n)
    assert np.all(a == np.array([[j+k for j in range(n+1)] for k in range(n)], dtype=np.int32))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    assert len(FindNodes(Assignment).visit(routine.body)) == 8
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 8
    for loop in loops:
        assert loop.bounds.start == '1'
        assert loop.bounds.stop == {'j': 'n + 1', 'k': 'n'}[str(loop.variable).lower()]
    assert len(FindNodes(Assignment).visit(routine.body)) == 8

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    n = 11
    a = np.zeros(shape=(n, n+1), order='F', dtype=np.int32)
    fissioned_function(a=a, n=n)
    assert np.all(a == np.array([[j+k for j in range(n+1)] for k in range(n)], dtype=np.int32))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_multiple(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_multiple(a, b, c, n)
  integer, intent(out) :: a(n), b(n), c(n)
  integer, intent(in) :: n
  integer :: j

  do j=1,n
    a(j) = j
    !$loki loop-fission
    b(j) = n-j
    !$loki loop-fission
    c(j) = a(j) + b(j)
  end do
end subroutine transform_loop_fission_multiple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    c = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, c=c, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))
    assert np.all(c == n)

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3
    for loop in loops:
        assert loop.bounds.start == '1'
        assert loop.bounds.stop == 'n'

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    c = np.zeros(shape=(n,), dtype=np.int32)
    fissioned_function(a=a, b=b, c=c, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n-1, -1, -1))
    assert np.all(c == n)

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_promote(a, b, n)
  integer, intent(out) :: a(n), b(n)
  integer, intent(in) :: n
  integer :: j, tmp

  do j=1,n
    a(j) = j
    tmp = j - 1
    !$loki loop-fission promote(tmp)
    b(j) = n-tmp
  end do
end subroutine transform_loop_fission_promote
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n, 0, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    for loop in loops:
        assert loop.bounds.start == '1'
        assert loop.bounds.stop == 'n'
    assert [str(d) for d in routine.variable_map['tmp'].shape] == ['n']

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n,), dtype=np.int32)
    fissioned_function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n, 0, -1))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote_conflicting_lengths(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_promote_conflicting_lengths(a, b, n)
  integer, intent(out) :: a(n), b(n+1)
  integer, intent(in) :: n
  integer :: j, tmp

  do j=1,n
    tmp = j - 1
    !$loki loop-fission promote(tmp)
    a(j) = tmp + 1
  end do

  do j=1,n+1
    tmp = j - 1
    !$loki loop-fission promote(tmp)
    b(j) = n-tmp
  end do
end subroutine transform_loop_fission_promote_conflicting_lengths
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n+1,), dtype=np.int32)
    function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n, -1, -1))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 4
    for loop in loops:
        assert loop.bounds.start == '1'
    assert loops[0].bounds.stop == 'n'
    assert loops[1].bounds.stop == 'n'
    assert loops[2].bounds.stop == 'n + 1'
    assert loops[3].bounds.stop == 'n + 1'
    assert [str(d) for d in routine.variable_map['tmp'].shape] == ['1 + n']

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    n = 100
    a = np.zeros(shape=(n,), dtype=np.int32)
    b = np.zeros(shape=(n+1,), dtype=np.int32)
    fissioned_function(a=a, b=b, n=n)
    assert np.all(a == range(1,n+1))
    assert np.all(b == range(n, -1, -1))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote_array(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_promote_array(a, klon, klev)
  integer, intent(inout) :: a(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl, zsupsat(klon)

  do jk=1,klev
    zsupsat(:) = 0
    do jl=1,klon
        zsupsat(jl) = jl
    end do
    !$loki loop-fission promote(ZSUPSAT)
    a(:, jk) = zsupsat(:)
  end do
end subroutine transform_loop_fission_promote_array
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, klon=klon, klev=klev)
    assert np.all(a == np.array([[jl] * klev for jl in range(1, klon+1)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == 'klev' for loop in loops) == 2
    assert routine.variable_map['zsupsat'].shape == ('klon', 'klev')

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    fissioned_function(a=a, klon=klon, klev=klev)
    assert np.all(a == np.array([[jl] * klev for jl in range(1, klon+1)], order='F'))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote_multiple(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_promote_multiple(a, klon, klev)
  integer, intent(inout) :: a(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl, zsupsat(klon), tmp

  do jk=1,klev
    zsupsat(:) = 0
    do jl=1,klon
        zsupsat(jl) = jl
    end do
    tmp = jk
    !$loki loop-fission promote(ZSUPSAT, tmp)
    a(:, jk) = zsupsat(:) + tmp
  end do
end subroutine transform_loop_fission_promote_multiple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, klon=klon, klev=klev)
    assert np.all(a == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == 'klev' for loop in loops) == 2
    assert routine.variable_map['zsupsat'].shape == ('klon', 'klev')
    assert routine.variable_map['tmp'].shape == ('klev',)

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    fissioned_function(a=a, klon=klon, klev=klev)
    assert np.all(a == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_multiple_promote(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_multiple_promote(a, b, klon, klev, nclv)
  integer, intent(inout) :: a(klon, klev), b(klon, klev, nclv)
  integer, intent(in) :: klon, klev, nclv
  integer :: jm, jk, jl, zsupsat(klon), zqxn(klon, nclv)

  do jk=1,klev
    zsupsat(:) = 0
    do jl=1,klon
        zsupsat(jl) = jl
    end do
    !$loki loop-fission
    do jm=1,nclv
        do jl=1,klon
            zqxn(jl, jm) = jm+jl
        end do
    end do
    !$loki loop-fission promote(ZSUPSAT)
    a(:, jk) = zsupsat(:)
    !$loki loop-fission promote( zQxN )
    do jm=1,nclv
        b(:, jk, jm) = zqxn(:, jm)
    end do
  end do
end subroutine transform_loop_fission_multiple_promote
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev, nclv = 32, 100, 5
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev, nclv), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev, nclv=nclv)
    assert np.all(a == np.array([[jl] * klev for jl in range(1, klon+1)], order='F'))
    assert np.all(b == np.array([[[jl + jm for jm in range(1, nclv+1)]] * klev
                                for jl in range(1, klon+1)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 5
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 8
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == 'klev' for loop in loops) == 4
    assert sum(loop.bounds.stop == 'klon' for loop in loops) == 2
    assert sum(loop.bounds.stop == 'nclv' for loop in loops) == 2
    assert routine.variable_map['zsupsat'].shape == ('klon', 'klev')
    assert routine.variable_map['zqxn'].shape == ('klon', 'nclv', 'klev')

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    klon, klev, nclv = 32, 100, 5
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev, nclv), order='F', dtype=np.int32)
    fissioned_function(a=a, b=b, klon=klon, klev=klev, nclv=nclv)
    assert np.all(a == np.array([[jl] * klev for jl in range(1, klon+1)], order='F'))
    assert np.all(b == np.array([[[jl + jm for jm in range(1, nclv+1)]] * klev
                                for jl in range(1, klon+1)], order='F'))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote_read_after_write(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_promote_read_after_write(a, klon, klev)
  integer, intent(inout) :: a(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl, zsupsat(klon), tmp

  do jk=1,klev
    zsupsat(:) = 0
    do jl=1,klon
        zsupsat(jl) = jl
    end do
    tmp = jk
    !$loki loop-fission
    a(:, jk) = zsupsat(:) + tmp
  end do
end subroutine transform_loop_fission_promote_read_after_write
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, klon=klon, klev=klev)
    assert np.all(a == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 3
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == 'klev' for loop in loops) == 2
    assert routine.variable_map['zsupsat'].shape == ('klon', 'klev')
    assert routine.variable_map['tmp'].shape == ('klev',)

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    fissioned_function(a=a, klon=klon, klev=klev)
    assert np.all(a == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote_multiple_read_after_write(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fission_promote_mult_r_a_w(a, b, klon, klev, nclv)
  integer, intent(inout) :: a(klon, klev), b(klon, klev, nclv)
  integer, intent(in) :: klon, klev, nclv
  integer :: jm, jk, jl, zsupsat(klon), zqxn(nclv, klon)
  ! Note the shape of zqxn, which is the reverse of the iteration space

  do jk=1,klev
    zsupsat(:) = 0
    do jl=1,klon
        zsupsat(jl) = jl
    end do
    !$loki loop-fission
    do jm=1,nclv
        do jl=1,klon
            zqxn(jm, jl) = jm+jl
        end do
    end do
    !$loki loop-fission
    a(:, jk) = zsupsat(:)
    !$loki loop-fission
    do jm=1,nclv
        b(:, jk, jm) = zqxn(jm, :)
    end do
  end do
end subroutine transform_loop_fission_promote_mult_r_a_w
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev, nclv = 32, 100, 5
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev, nclv), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev, nclv=nclv)
    assert np.all(a == np.array([[jl] * klev for jl in range(1, klon+1)], order='F'))
    assert np.all(b == np.array([[[jl + jm for jm in range(1, nclv+1)]] * klev
                                for jl in range(1, klon+1)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 5
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 8
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == 'klev' for loop in loops) == 4
    assert sum(loop.bounds.stop == 'klon' for loop in loops) == 2
    assert sum(loop.bounds.stop == 'nclv' for loop in loops) == 2
    assert routine.variable_map['zsupsat'].shape == ('klon', 'klev')
    assert routine.variable_map['zqxn'].shape == ('nclv', 'klon', 'klev')

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    klon, klev, nclv = 32, 100, 5
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev, nclv), order='F', dtype=np.int32)
    fissioned_function(a=a, b=b, klon=klon, klev=klev, nclv=nclv)
    assert np.all(a == np.array([[jl] * klev for jl in range(1, klon+1)], order='F'))
    assert np.all(b == np.array([[[jl + jm for jm in range(1, nclv+1)]] * klev
                                for jl in range(1, klon+1)], order='F'))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fusion_fission(tmp_path, frontend):
    fcode = """
subroutine transform_loop_fusion_fission(a, b, klon, klev)
  integer, intent(inout) :: a(klon, klev), b(klon, klev)
  integer, intent(in) :: klon, klev
  integer :: jk, jl, zsupsat(klon)

!$loki loop-fusion
  do jk=1,klev
    do jl=1,klon
      a(jl, jk) = jk
    end do
  end do

!$loki loop-fusion
  do jk=1,klev
    do jl=1,klon
      zsupsat(jl) = jl
    end do
    !$loki loop-fission promote(zsupsat)
    b(:, jk) = a(:, jk) + zsupsat(:)
  end do
end subroutine transform_loop_fusion_fission
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+1))] * klon, order='F'))
    assert np.all(b == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 4
    do_loop_fusion(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 3
    do_loop_fission(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 4
    assert all(loop.bounds.start == '1' for loop in loops)
    assert sum(loop.bounds.stop == 'klev' for loop in loops) == 2
    assert sum(loop.bounds.stop == 'klon' for loop in loops) == 2
    assert routine.variable_map['zsupsat'].shape == ('klon', 'klev')

    fissioned_filepath = tmp_path/(f'{routine.name}_fissioned_{frontend}.f90')
    fissioned_function = jit_compile(routine, filepath=fissioned_filepath, objname=routine.name)

    # Test transformation
    klon, klev = 32, 100
    a = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    b = np.zeros(shape=(klon, klev), order='F', dtype=np.int32)
    fissioned_function(a=a, b=b, klon=klon, klev=klev)
    assert np.all(a == np.array([list(range(1, klev+1))] * klon, order='F'))
    assert np.all(b == np.array([[jl + jk for jk in range(1, klev+1)]
                                for jl in range(1, klon+1)], order='F'))

    clean_test(filepath)
    clean_test(fissioned_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll(s)
    implicit none
    integer :: a
    integer, intent(inout) :: s

    !Loop A
    !$loki loop-unroll
    do a=1, 10
        s = s + a + 1
    end do

end subroutine test_transform_loop_unroll
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    assert s == sum(x + 1 for x in range(1, 11))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 0 and len(FindNodes(Assignment).visit(routine.body)) == 10

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == sum(x + 1 for x in range(1, 11))

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_step(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll_step(s)
    implicit none
    integer :: a
    integer, intent(inout) :: s

    !Loop A
    !$loki loop-unroll
    do a=-2, 7, 2
        s = s + a + 1
    end do

end subroutine test_transform_loop_unroll_step
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    assert s == sum(x + 1 for x in range(-2, 8, 2))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 0 and len(FindNodes(Assignment).visit(routine.body)) == 5

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == sum(x + 1 for x in range(-2, 8, 2))

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_non_literal_range(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll_non_literal_range(s)
    implicit none
    integer :: a, i
    integer, intent(inout) :: s

    i = 10

    !Loop A
    !$loki loop-unroll
    do a=1, i
        s = s + a + 1
    end do

end subroutine test_transform_loop_unroll_non_literal_range
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    assert s == sum(x + 1 for x in range(1, 11))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1 and len(FindNodes(Assignment).visit(routine.body)) == 2

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == sum(x + 1 for x in range(1, 11))

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll_nested(s)
    implicit none
    integer :: a, b
    integer, intent(inout) :: s

    !Loop A
    !$loki loop-unroll
    do a=1, 10
        !Loop B
        do b=1, 5
            s = s + a + b + 1
        end do
    end do

end subroutine test_transform_loop_unroll_nested
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    assert s == sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 0 and len(FindNodes(Assignment).visit(routine.body)) == 50

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested_restricted_depth(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll_nested_restricted_depth(s)
    implicit none
    integer :: a, b
    integer, intent(inout) :: s

    !Loop A
    !$loki loop-unroll depth(1)
    do a=1, 10
        !Loop B
        do b=1, 5
            s = s + a + b + 1
        end do
    end do

end subroutine test_transform_loop_unroll_nested_restricted_depth
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    assert s == sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 10 and len(FindNodes(Assignment).visit(routine.body)) == 10

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))

    # check unroll pragma has been removed
    assert not FindNodes(ir.Pragma).visit(routine.body)

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested_restricted_depth_unrollable(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll_nested_restricted_depth(s)
    implicit none
    integer :: a, b, i
    integer, intent(inout) :: s

    i = 10

    !Loop A
    !$loki loop-unroll depth(1)
    do a=1, i
        !Loop B
        do b=1, 5
            s = s + a + b + 1
        end do
    end do

end subroutine test_transform_loop_unroll_nested_restricted_depth
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    assert s == sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 1 and len(FindNodes(Assignment).visit(routine.body)) == 6

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested_counters(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll_nested_counters(s)
    implicit none

    integer :: a, b
    integer, intent(inout) :: s

    !Loop A
    !$loki loop-unroll
    do a=1, 10
        !Loop B
        do b=1, a
            s = s + a + b + 1
        end do
    end do

end subroutine test_transform_loop_unroll_nested_counters
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    tuples = [a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 11)) if b <= a]
    assert s == sum(tuples)

    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 0 and \
           len(FindNodes(Assignment).visit(routine.body)) == len(tuples)

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 11)) if b <= a)

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested_neighbours(tmp_path, frontend):
    fcode = """
subroutine test_transform_loop_unroll_nested_neighbours(s)
    implicit none

    integer :: a, b, c
    integer, intent(inout) :: s

    !Loop A
    !$loki loop-unroll depth(1)
    do a=1, 10
        !Loop B
        !$loki loop-unroll
        do b=1, 5
            s = s + a + b + 1
        end do
        !Loop C
        do c=1, 5
            s = s + a + c + 1
        end do
    end do

end subroutine test_transform_loop_unroll_nested_neighbours
 """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path / f'{routine.name}_{frontend}.f90'
    function = jit_compile(routine, filepath=filepath, objname=routine.name)

    # Test the reference solution
    s = np.array(0)
    function(s=s)
    assert s == 2 * sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))
    # Apply transformation
    assert len(FindNodes(Loop).visit(routine.body)) == 3
    do_loop_unroll(routine)
    assert len(FindNodes(Loop).visit(routine.body)) == 10 and len(FindNodes(Assignment).visit(routine.body)) == 60

    unrolled_filepath = tmp_path / f'{routine.name}_unrolled_{frontend}.f90'
    unrolled_function = jit_compile(routine, filepath=unrolled_filepath, objname=routine.name)

    # Test transformation
    s = np.array(0)
    unrolled_function(s=s)
    assert s == 2 * sum(a + b + 1 for (a, b) in itertools.product(range(1, 11), range(1, 6)))

    clean_test(filepath)
    clean_test(unrolled_filepath)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('loop_interchange', [False, True])
@pytest.mark.parametrize('loop_fusion', [False, True])
@pytest.mark.parametrize('loop_fission', [False, True])
@pytest.mark.parametrize('loop_unroll', [False, True])
def test_transform_loop_transformation(frontend, loop_interchange, loop_fusion, loop_fission, loop_unroll):
    fcode = """
subroutine transform_loop()
  integer, parameter :: m = 8
  integer, parameter :: n = 16

  integer :: array(m,n)
  integer :: a(n), b(n)
  integer :: i, j, s

  !$loki loop-interchange
  do i=1,n
    do j=1,m
      array(j, i) = i + j
    end do
  end do

  !$loki loop-fusion
  do i=1,n
    a(i) = i
  end do

  !$loki loop-fusion
  do i=1,n
    b(i) = n-i+1
  end do

  do j=1,n
    a(j) = j
    !$loki loop-fission
    b(j) = n-j
  end do

  !$loki loop-unroll
  do i=1, 10
      s = s + i + 1
  end do
end subroutine transform_loop
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)
    transform = TransformLoopsTransformation(loop_interchange=loop_interchange, loop_fusion=loop_fusion,
                                             loop_fission=loop_fission, loop_unroll=loop_unroll)

    num_pragmas = len(FindNodes(ir.Pragma).visit(routine.body))
    num_loops = len(FindNodes(ir.Loop).visit(routine.body))

    transform.apply(routine)
    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    loops = FindNodes(ir.Loop).visit(routine.body)

    if loop_interchange:
        num_pragmas -= 1
        assert loops[0].variable == 'j'
        assert not any('loop-interchange' in pragma.content for pragma in pragmas)
        assert FindNodes(ir.Loop).visit(loops[0].body)[0].variable == 'i'

    if loop_fusion:
        num_pragmas -= 1
        num_loops -= 1
        assert not any('loop-fusion' in pragma.content for pragma in pragmas)
        assert len(FindNodes(ir.Assignment).visit(loops[2].body)) == 2

    if loop_fission:
        num_pragmas -= 1
        num_loops += 1
        assert not any('loop-fission' in pragma.content for pragma in pragmas)

    if loop_unroll:
        num_pragmas -= 1
        num_loops -= 1
        assert not any('loop-unroll' in pragma.content for pragma in pragmas)

    assert len(loops) == num_loops
    assert len(pragmas) == num_pragmas


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_before_fuse(frontend):
    fcode = """
    subroutine test_loop_unroll_before_fuse(n, map, a, b)
       integer, intent(in) :: n
       integer, intent(in) :: map(3,3)
       real, intent(inout) :: a(n)
       real, intent(in) :: b(:)

       integer :: i,j,k

       !$loki loop-unroll
       do k=1,3
          !$loki loop-unroll
          do j=1,3
            !$loki loop-fusion
            do i=1,n
              a(i) = a(i) + b(map(j,k))
            enddo
          enddo
       enddo

    end subroutine test_loop_unroll_before_fuse
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3

    do_loop_unroll(routine)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 9
    assert all(loop.variable == 'i' for loop in loops)

    pragmas = FindNodes(ir.Pragma).visit(routine.body)
    assert len(pragmas) == 9
    assert all(p.content == 'loop-fusion' for p in pragmas)

    do_loop_fusion(routine)
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1
