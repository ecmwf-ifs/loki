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


def loop_var_names(node):
    return [str(loop.variable) for loop in FindNodes(Loop).visit(node)]


def loop_bounds(node):
    return [
        (
            str(loop.variable), str(loop.bounds.start), str(loop.bounds.stop),
            None if loop.bounds.step is None else str(loop.bounds.step)
        )
        for loop in FindNodes(Loop).visit(node)
    ]


def assignment_strs(node):
    return [(str(assign.lhs), str(assign.rhs)) for assign in FindNodes(Assignment).visit(node)]


def conditional_strs(node):
    return [str(cond.condition) for cond in FindNodes(Conditional).visit(node)]


def pragma_strs(node):
    return [pragma.content for pragma in FindNodes(ir.Pragma).visit(node)]


def variable_shape(routine, name):
    shape = routine.variable_map[name].shape
    return tuple(str(dim) for dim in shape) if shape else None


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_interchange_plain(frontend):
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

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 4
    assert loop_var_names(routine.body) == ['i', 'j', 'i', 'j']
    assert assignment_strs(routine.body) == [
        ('a(j, i)', 'i + j'), ('a(j, i)', 'a(j, i) - 2')
    ]

    do_loop_interchange(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 4
    assert loop_var_names(routine.body) == ['j', 'i', 'i', 'j']
    assert loop_bounds(routine.body) == [
        ('j', '1', 'm', None), ('i', '1', 'n', None),
        ('i', '1', 'n', None), ('j', '1', 'm', None)
    ]
    assert assignment_strs(routine.body) == [
        ('a(j, i)', 'i + j'), ('a(j, i)', 'a(j, i) - 2')
    ]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_interchange(frontend):
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

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert loop_var_names(routine.body) == ['k', 'i', 'j', 'k', 'i', 'j']
    assert assignment_strs(routine.body) == [
        ('a(j, i, k)', 'i + j + k'), ('a(j, i, k)', 'a(j, i, k) - 3')
    ]
    with pragmas_attached(routine, Loop):
        assert is_loki_pragma(loops[0].pragma, starts_with='some-pragma')
        assert is_loki_pragma(loops[1].pragma, starts_with='more-pragma')
        assert is_loki_pragma(loops[2].pragma, starts_with='other-pragma')

    do_loop_interchange(routine)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 6
    assert loop_var_names(routine.body) == ['j', 'i', 'k', 'k', 'i', 'j']
    assert loop_bounds(routine.body) == [
        ('j', '1', 'm', None), ('i', '1', 'n', None), ('k', '1', 'nclv', None),
        ('k', '1', 'nclv', None), ('i', '1', 'n', None), ('j', '1', 'm', None)
    ]
    assert assignment_strs(routine.body) == [
        ('a(j, i, k)', 'i + j + k'), ('a(j, i, k)', 'a(j, i, k) - 3')
    ]

    with pragmas_attached(routine, Loop):
        assert is_loki_pragma(loops[0].pragma, starts_with='some-pragma')
        assert is_loki_pragma(loops[1].pragma, starts_with='more-pragma')
        assert is_loki_pragma(loops[2].pragma, starts_with='other-pragma')


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_interchange_project(frontend):
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

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert loop_var_names(routine.body) == ['i', 'j']
    assert loop_bounds(routine.body) == [('i', '1', 'n', None), ('j', 'i', 'm', None)]
    assert assignment_strs(routine.body) == [('a(j, i)', 'i + j')]

    do_loop_interchange(routine, project_bounds=True)

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 2
    assert loop_var_names(routine.body) == ['j', 'i']
    assert loop_bounds(routine.body) == [('j', '1', 'm', None), ('i', '1', 'min(n, j)', None)]
    assert assignment_strs(routine.body) == [('a(j, i)', 'i + j')]


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
def test_transform_loop_fuse_matching(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)
    assert loop_bounds(routine.body) == [('i', '1', 'n', None)]
    assert assignment_strs(routine.body) == [('a(i)', 'i'), ('b(i)', 'n - i + 1')]
    assert pragma_strs(routine.body) == ['fused-loop group(default)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_subranges(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 3
    do_loop_fusion(routine)
    assert loop_bounds(routine.body) == [('i', '1', 'n', None)]
    assert conditional_strs(routine.body) == ['i <= 15', 'i >= 16']
    assert assignment_strs(routine.body) == [
        ('a(:)', '0'), ('b(:)', '0'), ('a(i)', 'a(i) + i'),
        ('b(i)', 'b(i) + n - i + 1'), ('b(i)', 'b(i) + n - i + 1')
    ]
    assert pragma_strs(routine.body) == ['fused-loop group(default)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_groups(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 5
    do_loop_fusion(routine)
    assert loop_bounds(routine.body) == [('i', '1', 'n', None), ('i', '1', 'n', None)]
    assert conditional_strs(routine.body) == ['i >= 2']
    assert assignment_strs(routine.body) == [
        ('c(1)', '1'), ('a(i)', 'i'), ('b(i)', 'n - i + 1'),
        ('c(i)', 'c(i - 1) + 1'), ('a(i)', 'a(i) + 1'), ('b(i)', 'b(i) + 1')
    ]
    assert pragma_strs(routine.body) == ['fused-loop group(g1)', 'fused-loop group(loop-group2)']


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
def test_transform_loop_fuse_alignment(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)
    assert loop_bounds(routine.body) == [('i', '0', 'n', None)]
    assert conditional_strs(routine.body) == ['i >= 1', 'i <= n - 1']
    assert assignment_strs(routine.body) == [('a(i)', 'i'), ('b(i + 1)', 'n - i')]
    assert pragma_strs(routine.body) == ['fused-loop group(1)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_nonmatching_lower(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)

    assert loop_bounds(routine.body) == [('jl', 'min(1, nclv)', 'klev', None)]
    assert conditional_strs(routine.body) == ['jl >= 1', 'jl >= nclv']
    assert assignment_strs(routine.body) == [('a(jl)', 'jl'), ('b(jl)', 'jl - nclv')]
    assert pragma_strs(routine.body) == ['fused-loop group(1)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_nonmatching_lower_annotated(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)

    assert loop_bounds(routine.body) == [('jl', '1', 'klev', None)]
    assert conditional_strs(routine.body) == ['jl >= nclv']
    assert assignment_strs(routine.body) == [('a(jl)', 'jl'), ('b(jl)', 'jl - nclv')]
    assert pragma_strs(routine.body) == ['fused-loop group(1)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_nonmatching_upper(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fusion(routine)

    assert loop_bounds(routine.body) == [('jl', '1', '1 + klev', None)]
    assert conditional_strs(routine.body) == ['jl <= klev']
    assert assignment_strs(routine.body) == [('a(jl)', 'jl'), ('b(jl)', '2*jl')]
    assert pragma_strs(routine.body) == ['fused-loop group(1)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_collapse(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 4
    do_loop_fusion(routine)
    assert loop_bounds(routine.body) == [('jk', '1', 'klev', None), ('jl', '1', 'klon', None)]
    assert assignment_strs(routine.body) == [('a(jl, jk)', 'jk'), ('b(jl, jk)', 'jl + jk')]
    assert pragma_strs(routine.body) == ['fused-loop group(default)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fuse_collapse_nonmatching(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 4
    do_loop_fusion(routine)
    assert loop_bounds(routine.body) == [('jk', '1', '1 + klev', None), ('jl', '1', '1 + klon', None)]
    assert conditional_strs(routine.body) == ['jl <= klon', 'jk <= klev']
    assert assignment_strs(routine.body) == [('a(jl, jk)', 'jk'), ('b(jl, jk)', 'jl + jk')]
    assert pragma_strs(routine.body) == ['fused-loop group(default)']


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
def test_transform_loop_fission_single(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [('j', '1', 'n', None), ('j', '1', 'n', None)]
    assert assignment_strs(routine.body) == [('a(j)', 'j'), ('b(j)', 'n - j')]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_nested(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 1
    assert len(FindNodes(Conditional).visit(routine.body)) == 1
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [('j', '1', 'n + 1', None), ('j', '1', 'n + 1', None)]
    assert conditional_strs(routine.body) == ['j <= n', 'j <= n']
    assert assignment_strs(routine.body) == [('a(j)', 'j'), ('b(j)', 'n - j')]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_nested_promote(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 1
    assert len(FindNodes(Conditional).visit(routine.body)) == 2
    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [('j', '1', 'n + 1', None), ('j', '1', 'n + 1', None)]
    assert conditional_strs(routine.body) == ['j <= n', 'zqxfg(2, j) <= n']
    assert assignment_strs(routine.body) == [
        ('zqxfg(2, j)', 'j'), ('a(j)', 'zqxfg(2, j)'), ('b(j)', 'n - zqxfg(2, j)')
    ]
    assert variable_shape(routine, 'zqxfg') == ('5', '1 + n')


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_collapse(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    assert len(FindNodes(Assignment).visit(routine.body)) == 8
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [
        ('j', '1', 'n + 1', None), ('j', '1', 'n + 1', None),
        ('k', '1', 'n', None), ('j', '1', 'n + 1', None),
        ('k', '1', 'n', None), ('k', '1', 'n', None),
        ('j', '1', 'n + 1', None), ('k', '1', 'n', None)
    ]
    assert assignment_strs(routine.body) == [
        ('tmp(:)', '0'), ('tmp(j)', 'j'), ('tmp2(:, j)', '0'), ('tmp2(k, j)', 'tmp(j) + k'),
        ('a(k, j)', 'tmp2(k, j)'), ('a(k, j)', 'a(k, j) - 1'), ('a(k, j)', '-1 + a(k, j)'), ('tmp(j)', '0')
    ]
    assert variable_shape(routine, 'tmp') == ('1 + n',)
    assert variable_shape(routine, 'tmp2') == ('n', '1 + n')


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_multiple(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [('j', '1', 'n', None), ('j', '1', 'n', None), ('j', '1', 'n', None)]
    assert assignment_strs(routine.body) == [('a(j)', 'j'), ('b(j)', 'n - j'), ('c(j)', 'a(j) + b(j)')]


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
def test_transform_loop_fission_promote_conflicting_lengths(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [
        ('j', '1', 'n', None), ('j', '1', 'n', None),
        ('j', '1', 'n + 1', None), ('j', '1', 'n + 1', None)
    ]
    assert assignment_strs(routine.body) == [
        ('tmp(j)', 'j - 1'), ('a(j)', 'tmp(j) + 1'), ('tmp(j)', 'j - 1'), ('b(j)', 'n - tmp(j)')
    ]
    assert variable_shape(routine, 'tmp') == ('1 + n',)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote_array(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [('jk', '1', 'klev', None), ('jl', '1', 'klon', None), ('jk', '1', 'klev', None)]
    assert assignment_strs(routine.body) == [
        ('zsupsat(:, jk)', '0'), ('zsupsat(jl, jk)', 'jl'), ('a(:, jk)', 'zsupsat(:, jk)')
    ]
    assert variable_shape(routine, 'zsupsat') == ('klon', 'klev')


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_promote_multiple(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [('jk', '1', 'klev', None), ('jl', '1', 'klon', None), ('jk', '1', 'klev', None)]
    assert assignment_strs(routine.body) == [
        ('zsupsat(:, jk)', '0'), ('zsupsat(jl, jk)', 'jl'), ('tmp(jk)', 'jk'), ('a(:, jk)', 'zsupsat(:, jk) + tmp(jk)')
    ]
    assert variable_shape(routine, 'zsupsat') == ('klon', 'klev')
    assert variable_shape(routine, 'tmp') == ('klev',)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_fission_multiple_promote(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 5
    do_loop_fission(routine)
    assert loop_bounds(routine.body) == [
        ('jk', '1', 'klev', None), ('jl', '1', 'klon', None), ('jk', '1', 'klev', None),
        ('jm', '1', 'nclv', None), ('jl', '1', 'klon', None), ('jk', '1', 'klev', None),
        ('jk', '1', 'klev', None), ('jm', '1', 'nclv', None)
    ]
    assert assignment_strs(routine.body) == [
        ('zsupsat(:, jk)', '0'), ('zsupsat(jl, jk)', 'jl'), ('zqxn(jl, jm, jk)', 'jm + jl'),
        ('a(:, jk)', 'zsupsat(:, jk)'), ('b(:, jk, jm)', 'zqxn(:, jm, jk)')
    ]
    assert variable_shape(routine, 'zsupsat') == ('klon', 'klev')
    assert variable_shape(routine, 'zqxn') == ('klon', 'nclv', 'klev')


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
def test_transform_loop_unroll(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_unroll(routine)
    assert not FindNodes(Loop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(routine.body)) == 10
    assert assignment_strs(routine.body) == [('s', f's + {i} + 1') for i in range(1, 11)]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_step(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_unroll(routine)
    assert not FindNodes(Loop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(routine.body)) == 5
    assert assignment_strs(routine.body) == [
        ('s', 's + -2 + 1'), ('s', 's + 0 + 1'), ('s', 's + 2 + 1'), ('s', 's + 4 + 1'), ('s', 's + 6 + 1')
    ]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_non_literal_range(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 1
    do_loop_unroll(routine)
    assert loop_bounds(routine.body) == [('a', '1', 'i', None)]
    assert assignment_strs(routine.body) == [('i', '10'), ('s', 's + a + 1')]
    assert not pragma_strs(routine.body)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_unroll(routine)
    assert not FindNodes(Loop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(routine.body)) == 50
    assert assignment_strs(routine.body)[:3] == [
        ('s', 's + 1 + 1 + 1'), ('s', 's + 1 + 2 + 1'), ('s', 's + 1 + 3 + 1')
    ]
    assert assignment_strs(routine.body)[-3:] == [
        ('s', 's + 10 + 3 + 1'), ('s', 's + 10 + 4 + 1'), ('s', 's + 10 + 5 + 1')
    ]


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested_restricted_depth(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_unroll(routine)
    assert loop_bounds(routine.body) == [('b', '1', '5', None)] * 10
    assert len(FindNodes(Assignment).visit(routine.body)) == 10
    assert assignment_strs(routine.body)[:3] == [
        ('s', 's + 1 + b + 1'), ('s', 's + 2 + b + 1'), ('s', 's + 3 + b + 1')
    ]
    assert not pragma_strs(routine.body)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_loop_unroll_nested_restricted_depth_unrollable(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 2
    do_loop_unroll(routine)
    assert loop_bounds(routine.body) == [('a', '1', 'i', None)]
    assert len(FindNodes(Assignment).visit(routine.body)) == 6
    assert assignment_strs(routine.body) == [
        ('i', '10'), ('s', 's + a + 1 + 1'), ('s', 's + a + 2 + 1'),
        ('s', 's + a + 3 + 1'), ('s', 's + a + 4 + 1'), ('s', 's + a + 5 + 1')
    ]


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
def test_transform_loop_unroll_nested_neighbours(frontend):
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

    assert len(FindNodes(Loop).visit(routine.body)) == 3
    do_loop_unroll(routine)
    assert loop_bounds(routine.body) == [('c', '1', '5', None)] * 10
    assert len(FindNodes(Assignment).visit(routine.body)) == 60
    assert assignment_strs(routine.body)[:6] == [
        ('s', 's + 1 + 1 + 1'), ('s', 's + 1 + 2 + 1'), ('s', 's + 1 + 3 + 1'),
        ('s', 's + 1 + 4 + 1'), ('s', 's + 1 + 5 + 1'), ('s', 's + 1 + c + 1')
    ]


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
