import pytest

from conftest import available_frontends
from loki.frontend import OMNI
from loki.ir import Assignment, Associate, CallStatement

from loki.transform import resolve_associates
from loki import (
    BasicType, FindNodes, Subroutine
)


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_associates_simple(frontend):
    """
    Test association resolver on simple cases.
    """
    fcode = """
subroutine transform_associates_simple
  use some_module, only: some_obj
  implicit none

  real :: local_var

  associate (a => some_obj%a)
    local_var = a
  end associate
end subroutine transform_associates_simple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(Associate).visit(routine.body)) == 1
    assert len(FindNodes(Assignment).visit(routine.body)) == 1
    assign = FindNodes(Assignment).visit(routine.body)[0]
    assert assign.rhs == 'a' and 'some_obj' not in assign.rhs
    assert assign.rhs.type.dtype == BasicType.DEFERRED

    # Now apply the association resolver
    resolve_associates(routine)

    assert len(FindNodes(Associate).visit(routine.body)) == 0
    assert len(FindNodes(Assignment).visit(routine.body)) == 1
    assign = FindNodes(Assignment).visit(routine.body)[0]
    assert assign.rhs == 'some_obj%a'
    assert assign.rhs.parent == 'some_obj'
    assert assign.rhs.type.dtype == BasicType.DEFERRED
    assert assign.rhs.scope == routine
    assert assign.rhs.parent.scope == routine


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_associates_nested(frontend):
    """
    Test association resolver with deeply nested associates.
    """
    fcode = """
subroutine transform_associates_nested
  use some_module, only: some_obj
  implicit none

  real :: rick

  associate (never => some_obj%never)
    associate (gonna => never%gonna)
      associate (a => gonna%give%you%up)
        rick = a
      end associate
    end associate
  end associate
end subroutine transform_associates_nested
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(Associate).visit(routine.body)) == 3
    assert len(FindNodes(Assignment).visit(routine.body)) == 1
    assign = FindNodes(Assignment).visit(routine.body)[0]
    assert assign.lhs == 'rick' and assign.rhs == 'a'
    assert assign.rhs.type.dtype == BasicType.DEFERRED

    # Now apply the association resolver
    resolve_associates(routine)

    assert len(FindNodes(Associate).visit(routine.body)) == 0
    assert len(FindNodes(Assignment).visit(routine.body)) == 1
    assign = FindNodes(Assignment).visit(routine.body)[0]
    assert assign.rhs == 'some_obj%never%gonna%give%you%up'


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_associates_array_call(frontend):
    """
    Test a neat corner case where a component of an associated array
    is used as a keyword argument in a subroutine call.
    """
    fcode = """
subroutine transform_associates_simple
  use some_module, only: some_obj
  implicit none

  integer :: i
  real :: local_var

  associate (some_array => some_obj%some_array)

    do i=1, 5
      call another_routine(i, n=some_array(i)%n)
    end do
  end associate
end subroutine transform_associates_simple
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(Associate).visit(routine.body)) == 1
    assert len(FindNodes(CallStatement).visit(routine.body)) == 1
    call = FindNodes(CallStatement).visit(routine.body)[0]
    assert call.kwarguments[0][1] == 'some_array(i)%n'
    assert call.kwarguments[0][1].type.dtype == BasicType.DEFERRED

    # Now apply the association resolver
    resolve_associates(routine)

    assert len(FindNodes(Associate).visit(routine.body)) == 0
    assert len(FindNodes(CallStatement).visit(routine.body)) == 1
    call = FindNodes(CallStatement).visit(routine.body)[0]
    assert call.kwarguments[0][1] == 'some_obj%some_array(i)%n'
    assert call.kwarguments[0][1].scope == routine
    assert call.kwarguments[0][1].type.dtype == BasicType.DEFERRED
