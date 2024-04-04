# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends
from loki import (
    Subroutine, FindNodes, Conditional, Assignment, Loop, Comment, OMNI
)
from loki.transform import dead_code_elimination, remove_marked_regions


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_dead_code_conditional(frontend):
    """
    Test correct elimination of unreachable conditional branches.
    """
    fcode = """
subroutine test_dead_code_conditional(a, b, flag)
  real(kind=8), intent(inout) :: a, b
  logical, intent(in) :: flag

  if (flag) then
    if (1 == 6) then
      a = a + b
    else
      b = b + 2.0
    end if

    if (2 == 2) then
      b = b + a
    else
      a = a + 3.0
    end if

    if (1 == 2) then
      b = b + a
    elseif (3 == 3) then
      a = a + b
    else
      a = a + 6.0
    end if

  end if
end subroutine test_dead_code_conditional
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Please note that nested conditionals (elseif) counts as two
    assert len(FindNodes(Conditional).visit(routine.body)) == 5
    assert len(FindNodes(Assignment).visit(routine.body)) == 7

    dead_code_elimination(routine)

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    assert conditionals[0].condition == 'flag'
    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'b' and assigns[0].rhs == 'b + 2.0'
    assert assigns[1].lhs == 'b' and assigns[1].rhs == 'b + a'
    assert assigns[2].lhs == 'a' and assigns[2].rhs == 'a + b'


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_dead_code_conditional_nested(frontend):
    """
    Test correct elimination of unreachable branches in nested conditionals.
    """
    fcode = """
subroutine test_dead_code_conditional(a, b, flag)
  real(kind=8), intent(inout) :: a, b
  logical, intent(in) :: flag

  if (1 == 2) then
    a = a + 5
  elseif (flag) then
    b = b + 4
  else
    b = a + 3
  end if

  if (a > 2.0) then
    a = a + 5.0
  elseif (2 == 3) then
    a = a + 3.0
  else
    a = a + 1.0
  endif

  if (a > 2.0) then
    a = a + 5.0
  elseif (2 == 3) then
    a = a + 3.0
  elseif (a > 1.0) then
    a = a + 2.0
  else
    a = a + 1.0
  endif
end subroutine test_dead_code_conditional
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Please note that nested conditionals (elseif) counts as two
    assert len(FindNodes(Conditional).visit(routine.body)) == 7
    assert len(FindNodes(Assignment).visit(routine.body)) == 10

    dead_code_elimination(routine)

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 4
    assert conditionals[0].condition == 'flag'
    assert not conditionals[0].has_elseif
    assert conditionals[1].condition == 'a > 2.0'
    assert not conditionals[1].has_elseif
    assert conditionals[2].condition == 'a > 2.0'
    if not frontend == OMNI:  # OMNI does not get elseifs right
        assert conditionals[2].has_elseif
    assert conditionals[3].condition == 'a > 1.0'
    assert not conditionals[3].has_elseif
    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 7
    assert assigns[0].lhs == 'b' and assigns[0].rhs == 'b + 4'
    assert assigns[1].lhs == 'b' and assigns[1].rhs == 'a + 3'
    assert assigns[2].lhs == 'a' and assigns[2].rhs == 'a + 5.0'
    assert assigns[3].lhs == 'a' and assigns[3].rhs == 'a + 1.0'
    assert assigns[4].lhs == 'a' and assigns[4].rhs == 'a + 5.0'
    assert assigns[5].lhs == 'a' and assigns[5].rhs == 'a + 2.0'
    assert assigns[6].lhs == 'a' and assigns[6].rhs == 'a + 1.0'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('mark_with_comment', [True, False])
def test_transform_remove_code_pragma_region(frontend, mark_with_comment):
    """
    Test correct removal of pragma-marked code regions.
    """
    fcode = """
subroutine test_remove_code(a, b, n, flag)
  real(kind=8), intent(inout) :: a, b(n)
  integer, intent(in) :: n
  logical, intent(in) :: flag
  integer :: i

  if (flag) then
    a = a + 1.0
  end if

  !$loki remove
  do i=1, n
    !$loki rick-roll
    a = a + 3.0
    !$loki end rick-roll
  end do
  !$loki end remove

  b(:) = 1.0

  !$acc parallel
  do i=1, n
    b(i) = b(i) + a

    !$loki remove
    a = b(i) + 42.
    !$loki end remove
  end do
end subroutine test_remove_code
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    remove_marked_regions(routine, mark_with_comment=mark_with_comment)

    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'a' and assigns[0].rhs == 'a + 1.0'
    assert assigns[1].lhs == 'b(:)' and assigns[1].rhs == '1.0'
    assert assigns[2].lhs == 'b(i)' and assigns[2].rhs == 'b(i) + a'

    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    assert assigns[2] in loops[0].body

    comments = [
        c for c in FindNodes(Comment).visit(routine.body)
        if '[Loki] Removed content' in c.text
    ]
    assert len(comments) == (2 if mark_with_comment else 0)
