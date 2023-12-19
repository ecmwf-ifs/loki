# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends
from loki import Subroutine, FindNodes, Conditional, Assignment
from loki.transform import dead_code_elimination


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
end subroutine test_dead_code_conditional
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    # Please note that nested conditionals (elseif) counts as two
    assert len(FindNodes(Conditional).visit(routine.body)) == 2
    assert len(FindNodes(Assignment).visit(routine.body)) == 3

    dead_code_elimination(routine)

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 1
    assert conditionals[0].condition == 'flag'
    assigns = FindNodes(Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].lhs == 'b' and assigns[0].rhs == 'b + 4'
    assert assigns[1].lhs == 'b' and assigns[1].rhs == 'a + 3'

    # Ensure that the `has_elseif` attribute has been resolved
    assert not conditionals[0].has_elseif
