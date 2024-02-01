# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from conftest import available_frontends
from loki.frontend import OMNI
from loki.ir import Assignment, Associate, CallStatement, Conditional

from loki.transform import (
    resolve_associates, transform_sequence_association, SanitiseTransformation
)
from loki import (
    BasicType, FindNodes, Subroutine, Module, fgen
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


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_associates_nested_conditional(frontend):
    """
    Test association resolver when associate is nested into a conditional.
    """
    fcode = """
subroutine transform_associates_nested_conditional
    use some_module, only: some_obj, some_flag
    implicit none

    real :: local_var

    if (some_flag) then
        local_var = 0.
    else
        ! Other nodes before the associate
        ! This one, too

        ! And this one
        associate (a => some_obj%a)
            local_var = a
            ! And a conditional which may inject a tuple nesting in the IR
            if (local_var > 10.) then
                local_var = 10.
            end if
        end associate
        ! And nodes after it

        ! like this
    end if
end subroutine transform_associates_nested_conditional
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(Conditional).visit(routine.body)) == 2
    assert len(FindNodes(Associate).visit(routine.body)) == 1
    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assign = FindNodes(Assignment).visit(routine.body)[1]
    assert assign.rhs == 'a' and 'some_obj' not in assign.rhs
    assert assign.rhs.type.dtype == BasicType.DEFERRED

    # Now apply the association resolver
    resolve_associates(routine)

    assert len(FindNodes(Conditional).visit(routine.body)) == 2
    assert len(FindNodes(Associate).visit(routine.body)) == 0
    assert len(FindNodes(Assignment).visit(routine.body)) == 3
    assign = FindNodes(Assignment).visit(routine.body)[1]
    assert assign.rhs == 'some_obj%a'
    assert assign.rhs.parent == 'some_obj'
    assert assign.rhs.type.dtype == BasicType.DEFERRED
    assert assign.rhs.scope == routine
    assert assign.rhs.parent.scope == routine


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_sequence_assocaition_scalar_notation(frontend):
    fcode = """
module mod_a
    implicit none

    type type_b
        integer :: c
        integer :: d
    end type type_b

    type type_a
        type(type_b) :: b
    end type type_a

contains

    subroutine main()

        type(type_a) :: a
        integer :: k, m, n

        real    :: array(10,10)

        call sub_x(array(1, 1), 1)
        call sub_x(array(2, 2), 2)
        call sub_x(array(m, 1), k)
        call sub_x(array(m-1, 1), k-1)
        call sub_x(array(a%b%c, 1), a%b%d)

    contains

        subroutine sub_x(array, k)

            integer, intent(in) :: k
            real, intent(in)    :: array(k:n)

        end subroutine sub_x

    end subroutine main

end module mod_a
    """.strip()

    module = Module.from_source(fcode, frontend=frontend)
    routine = module['main']

    transform_sequence_association(routine)

    calls = FindNodes(CallStatement).visit(routine.body)

    assert fgen(calls[0]).lower() == 'call sub_x(array(1:10, 1), 1)'
    assert fgen(calls[1]).lower() == 'call sub_x(array(2:10, 2), 2)'
    assert fgen(calls[2]).lower() == 'call sub_x(array(m:10, 1), k)'
    assert fgen(calls[3]).lower() == 'call sub_x(array(m - 1:10, 1), k - 1)'
    assert fgen(calls[4]).lower() == 'call sub_x(array(a%b%c:10, 1), a%b%d)'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('resolve_associate', [True, False])
@pytest.mark.parametrize('resolve_sequence', [True, False])
def test_transformation_sanitise(frontend, resolve_associate, resolve_sequence):
    """
    Test that the selective dispatch of the sanitisations works.
    """

    fcode = """
module test_transformation_sanitise_mod
  implicit none

  type rick
    real :: scalar
  end type rick
contains

  subroutine test_transformation_sanitise(a, dave)
    real, intent(inout) :: a(3)
    type(rick), intent(inout) :: dave

    associate(scalar => dave%scalar)
      scalar = a(1) + a(2)

      call vadd(a(1), 2.0, 3)
    end associate

  contains
    subroutine vadd(x, y, n)
      real, intent(inout) :: x(n)
      real, intent(inout) :: y
      integer, intent(in) :: n

      x = x + 2.0
    end subroutine vadd
  end subroutine test_transformation_sanitise
end module test_transformation_sanitise_mod
"""
    module = Module.from_source(fcode, frontend=frontend)
    routine = module['test_transformation_sanitise']

    assoc = FindNodes(Associate).visit(routine.body)
    assert len(assoc) == 1
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments[0] == 'a(1)'

    trafo = SanitiseTransformation(
        resolve_associate_mappings=resolve_associate,
        resolve_sequence_association=resolve_sequence,
    )
    trafo.apply(routine)

    assoc = FindNodes(Associate).visit(routine.body)
    assert len(assoc) == 0 if resolve_associate else 1

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments[0] == 'a(1:3)' if resolve_sequence else 'a(1)'
