# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import BasicType, Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes

from loki.transformations.sanitise import (
    do_resolve_associates, do_merge_associates,
    ResolveAssociatesTransformer, AssociatesTransformation
)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
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
    local_var = a(:)
  end associate
end subroutine transform_associates_simple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 1
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 1
    assign = FindNodes(ir.Assignment).visit(routine.body)[0]
    assert assign.rhs == 'a(:)' and 'some_obj' not in assign.rhs
    assert assign.rhs.type.dtype == BasicType.DEFERRED

    # Now apply the association resolver
    do_resolve_associates(routine)

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 0
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 1
    assign = FindNodes(ir.Assignment).visit(routine.body)[0]
    assert assign.rhs == 'some_obj%a(:)'
    assert assign.rhs.parent == 'some_obj'
    assert assign.rhs.type.dtype == BasicType.DEFERRED
    assert assign.rhs.scope == routine
    assert assign.rhs.parent.scope == routine


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
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

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 3
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 1
    assign = FindNodes(ir.Assignment).visit(routine.body)[0]
    assert assign.lhs == 'rick' and assign.rhs == 'a'
    assert assign.rhs.type.dtype == BasicType.DEFERRED

    # Now apply the association resolver
    do_resolve_associates(routine)

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 0
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 1
    assign = FindNodes(ir.Assignment).visit(routine.body)[0]
    assert assign.rhs == 'some_obj%never%gonna%give%you%up'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
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
  real, allocatable :: local_arr(:)

  associate (some_array => some_obj%some_array, a => some_obj%a)
    allocate(local_arr(a%n))

    do i=1, 5
      call another_routine(i, n=some_array(i)%n)
    end do
  end associate
end subroutine transform_associates_simple
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 1
    assert len(FindNodes(ir.CallStatement).visit(routine.body)) == 1
    call = FindNodes(ir.CallStatement).visit(routine.body)[0]
    assert call.kwarguments[0][1] == 'some_array(i)%n'
    assert call.kwarguments[0][1].type.dtype == BasicType.DEFERRED
    assert routine.variable_map['local_arr'].type.shape == ('a%n',)

    # Now apply the association resolver
    do_resolve_associates(routine)

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 0
    assert len(FindNodes(ir.CallStatement).visit(routine.body)) == 1
    call = FindNodes(ir.CallStatement).visit(routine.body)[0]
    assert call.kwarguments[0][1] == 'some_obj%some_array(i)%n'
    assert call.kwarguments[0][1].scope == routine
    assert call.kwarguments[0][1].type.dtype == BasicType.DEFERRED

    # Test the special case of shapes derived from allocations
    assert routine.variable_map['local_arr'].type.shape == ('some_obj%a%n',)


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_associates_array_slices(frontend):
    """
    Test the resolution of associated array slices.
    """
    fcode = """
subroutine transform_associates_slices(arr2d, arr3d)
  use some_module, only: some_obj, another_routine
  implicit none
  real, intent(inout) :: arr2d(:,:), arr3d(:,:,:)
  integer :: i, j
  integer, parameter :: idx_a = 2
  integer, parameter :: idx_c = 3

  associate (a => arr2d(:, 1), b=>arr2d(:, idx_a), &
           & c => arr3d(:,:,idx_c) )
    b(:) = 42.0
    do i=1, 5
      a(i) = b(i+2)
      call another_routine(i, a(2:4), b)
      do j=1, 7
        c(i, j) = c(i, j) + b(j)
      end do
    end do
  end associate
end subroutine transform_associates_slices
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 1
    assert len(FindNodes(ir.CallStatement).visit(routine.body)) == 1
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments[1] == 'a(2:4)'
    assert calls[0].arguments[2] == 'b'

    # Now apply the association resolver
    do_resolve_associates(routine)

    assert len(FindNodes(ir.Associate).visit(routine.body)) == 0
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'arr2d(:, idx_a)'
    assert assigns[1].lhs == 'arr2d(i, 1)'
    assert assigns[1].rhs == 'arr2d(i+2, idx_a)'
    assert assigns[2].lhs == 'arr3d(i, j, idx_c)'
    assert assigns[2].rhs == 'arr3d(i, j, idx_c) + arr2d(j, idx_a)'

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 1
    assert calls[0].arguments[1] == 'arr2d(2:4, 1)'
    assert calls[0].arguments[2] == 'arr2d(:, idx_a)'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
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

    assert len(FindNodes(ir.Conditional).visit(routine.body)) == 2
    assert len(FindNodes(ir.Associate).visit(routine.body)) == 1
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 3
    assign = FindNodes(ir.Assignment).visit(routine.body)[1]
    assert assign.rhs == 'a' and 'some_obj' not in assign.rhs
    assert assign.rhs.type.dtype == BasicType.DEFERRED

    # Now apply the association resolver
    do_resolve_associates(routine)

    assert len(FindNodes(ir.Conditional).visit(routine.body)) == 2
    assert len(FindNodes(ir.Associate).visit(routine.body)) == 0
    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 3
    assign = FindNodes(ir.Assignment).visit(routine.body)[1]
    assert assign.rhs == 'some_obj%a'
    assert assign.rhs.parent == 'some_obj'
    assert assign.rhs.type.dtype == BasicType.DEFERRED
    assert assign.rhs.scope == routine
    assert assign.rhs.parent.scope == routine


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_associates_partial_body(frontend):
    """
    Test resolving associated symbols, but only for a part of an
    associate's body.
    """
    fcode = """
subroutine transform_associates_partial
  use some_module, only: some_obj
  implicit none

  integer :: i
  real :: local_var

  associate (a=>some_obj%a, b=>some_obj%b)
    local_var = a(1)

    do i=1, some_obj%n
      a(i) = a(i) + 1.
      b(i) = b(i) + 1.
    end do
  end associate
end subroutine transform_associates_partial
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 3
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    transformer = ResolveAssociatesTransformer(inplace=True)
    transformer.visit(loops[0])

    # Check that associated symbols have been resolved in loop body only
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'local_var'
    assert assigns[0].rhs == 'a(1)'
    assert assigns[1].lhs == 'some_obj%a(i)'
    assert assigns[1].rhs == 'some_obj%a(i) + 1.'
    assert assigns[2].lhs == 'some_obj%b(i)'
    assert assigns[2].rhs == 'some_obj%b(i) + 1.'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_transform_associates_start_depth(frontend):
    """
    Test resolving associated symbols, but only for a part of an
    associate's body.
    """
    fcode = """
subroutine transform_associates_partial
  use some_module, only: some_obj
  implicit none

  integer :: i
  real :: local_var

  associate (a=>some_obj%a, b=>some_obj%b)
  associate (c=>a%b, d=>b%d)
    local_var = a(1)

    do i=1, some_obj%n
      c(i) = c(i) + 1.
      d(i) = d(i) + 1.
    end do
  end associate
  end associate
end subroutine transform_associates_partial
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert len(FindNodes(ir.Assignment).visit(routine.body)) == 3
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1

    # Resolve all expect the outermost associate block
    do_resolve_associates(routine, start_depth=1)

    # Check that associated symbols have been resolved in loop body only
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 1
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 3
    assert assigns[0].lhs == 'local_var'
    assert assigns[0].rhs == 'a(1)'
    assert assigns[1].lhs == 'a%b(i)'
    assert assigns[1].rhs == 'a%b(i) + 1.'
    assert assigns[2].lhs == 'b%d(i)'
    assert assigns[2].rhs == 'b%d(i) + 1.'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_merge_associates_nested(frontend):
    """
    Test association merging for nested mappings.
    """
    fcode = """
subroutine merge_associates_simple(base)
  use some_module, only: some_type
  implicit none

  type(some_type), intent(inout) :: base
  integer :: i
  real :: local_var

  associate(a => base%a)
  associate(b => base%other%symbol)
  associate(d => base%other%symbol%really%deep, &
   &        a => base%a, c => a%more)
    do i=1, 5
      call another_routine(i, n=b(c)%n)

      d(i) = 42.0
    end do
  end associate
  end associate
  end associate
end subroutine merge_associates_simple
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    assocs = FindNodes(ir.Associate).visit(routine.body)
    assert len(assocs) == 3
    assert len(assocs[0].associations) == 1
    assert len(assocs[1].associations) == 1
    assert len(assocs[2].associations) == 3

    # Move associate mapping around
    do_merge_associates(routine, max_parents=2)

    assocs = FindNodes(ir.Associate).visit(routine.body)
    assert len(assocs) == 3
    assert len(assocs[0].associations) == 2
    assert assocs[0].associations[0] == ('base%a', 'a')
    assert assocs[0].associations[1] == ('base%other%symbol', 'b')
    assert len(assocs[1].associations) == 1
    assert assocs[1].associations[0] == ('a%more', 'c')
    assert len(assocs[2].associations) == 1
    assert assocs[2].associations[0] == ('base%other%symbol%really%deep', 'd')

    # Check that body symbols have been rescoped correctly
    call = FindNodes(ir.CallStatement).visit(routine.body)[0]
    b_c_n = call.kwarguments[0][1]  # b(c)%n
    assert b_c_n.scope == assocs[0]


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
))
@pytest.mark.parametrize('merge', [False, True])
@pytest.mark.parametrize('resolve', [False, True])
def test_associates_transformation(frontend, merge, resolve):
    """
    Test association merging paired with partial resolution of inner
    scopes via :any:`AssociatesTransformation`.
    """
    fcode = """
subroutine merge_associates_simple(base)
  use some_module, only: some_type
  implicit none

  type(some_type), intent(inout) :: base
  integer :: i
  real :: local_var

  associate(a => base%a)
  associate(b => base%b)
  associate(c => a%c)
  associate(d => c%d)
    do i=1, 5
      call another_routine(b(i), c%n)

      d(i) = 42.0
    end do
  end associate
  end associate
  end associate
  end associate
end subroutine merge_associates_simple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    AssociatesTransformation(
        resolve_associates=resolve, merge_associates=merge, start_depth=1
    ).apply(routine)

    assocs = FindNodes(ir.Associate).visit(routine.body)
    call = FindNodes(ir.CallStatement).visit(routine.body)[0]
    assign = FindNodes(ir.Assignment).visit(routine.body)[0]

    if not merge and not resolve:
        assert len(assocs) == 4
        assert all(len(a.associations) == 1 for a in assocs)

        assert call.arguments[0] == 'b(i)'
        assert call.arguments[1] == 'c%n'
        assert assign.lhs == 'd(i)'

    if merge and not resolve:
        assert len(assocs) == 4
        assert assocs[0].associations == (('base%a', 'a'), ('base%b', 'b'))
        assert assocs[1].associations == (('a%c', 'c'), )
        assert assocs[2].associations == ()
        assert assocs[3].associations == (('c%d', 'd'), )

        assert call.arguments[0] == 'b(i)'
        assert call.arguments[1] == 'c%n'
        assert assign.lhs == 'd(i)'

    if not merge and resolve:
        assert len(assocs) == 1
        assert assocs[0].associations == (('base%a', 'a'),)

        assert call.arguments[0] == 'base%b(i)'
        assert call.arguments[1] == 'a%c%n'
        assert assign.lhs == 'a%c%d(i)'

    if merge and resolve:
        assert len(assocs) == 1
        assert assocs[0].associations == (('base%a', 'a'), ('base%b', 'b'))

        assert call.arguments[0] == 'b(i)'
        assert call.arguments[1] == 'a%c%n'
        assert assign.lhs == 'a%c%d(i)'


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_resolve_associates_stmt_func(frontend):
    """
    Test scope management for stmt funcs, either as
    :any:`ProcedureSymbol` or :any:`DeferredTypeSymbol`.
    """
    fcode = """
subroutine test_associates_stmt_func(ydcst, a, b)
  use yomcst, only: tcst
  implicit none
  type(tcst), intent(in) :: ydcst
  real(kind=8), intent(inout) :: a, b
#include "some_stmt.func.h"
  real(kind=8) :: not_an_array
  not_an_array ( x, y ) =  x * y

associate(RTT=>YDCST%RTT)
  a = not_an_array(RTT, 1.0) + a
  b = some_stmt_func(RTT, 1.0) + b
end associate
end subroutine test_associates_stmt_func
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    associate = FindNodes(ir.Associate).visit(routine.body)[0]
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert isinstance(assigns[0].rhs.children[0], sym.InlineCall)
    assert assigns[0].rhs.children[0].function.scope == associate
    assert isinstance(assigns[1].rhs.children[0], sym.InlineCall)
    assert assigns[1].rhs.children[0].function.scope == associate

    do_resolve_associates(routine)

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].rhs == 'not_an_array(YDCST%RTT, 1.0) + a'
    assert assigns[1].rhs == 'some_stmt_func(YDCST%RTT, 1.0) + b'
    assert isinstance(assigns[0].rhs.children[0], sym.InlineCall)
    assert assigns[0].rhs.children[0].function.scope == routine
    assert isinstance(assigns[1].rhs.children[0], sym.InlineCall)
    assert assigns[1].rhs.children[0].function.scope == routine

    # Trigger a full clone, which would fail if scopes are missing
    routine.clone()
