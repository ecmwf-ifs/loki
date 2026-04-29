# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for control-flow IR nodes.
"""

import pytest

from loki import Module, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'OMNI fails to read without full module')]))
def test_select_type(frontend, tmp_path):
    fcode = """
module select_type_mod
    use imported_type_mod, only: imported_type
    implicit none
    type, abstract :: base
    end type base
    type, extends(base) :: derived1
        real :: val
    end type derived1
    type, extends(base) :: derived2
        integer :: val
    end type derived2
contains
    subroutine select_type_routine(arg, arg2)
        class(base), intent(inout) :: arg
        class(imported_type), intent(inout) :: arg2
        select type( arg )
            class is(derived1)
                arg%val = 1.0
            class is(derived2)
                arg%val = 1
            class default
                print *, 'error'
        end select
        ! Some comment before the second select
        select type( arg )
            type is(base)
                write(*,*) 'default'
        end select
        select type( arg2 )
            ! inline comment
            type is(imported_type)
                print *, 'imported type'
        end select
    end subroutine select_type_routine
end module select_type_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    tconds = FindNodes(ir.TypeConditional).visit(module['select_type_routine'].body)
    assert len(tconds) == 3

    assert tconds[0].expr == 'arg'
    assert tconds[0].values == (
        ('derived1', True), ('derived2', True)
    )
    assert len(tconds[0].bodies) == 2
    assert len(tconds[0].else_body) == 1

    assert tconds[1].expr == 'arg'
    assert tconds[1].values == (('base', False),)
    assert not tconds[1].else_body

    assert tconds[2].expr == 'arg2'
    assert tconds[2].values == (('imported_type', False),)
    assert not tconds[2].else_body

    comments = FindNodes(ir.Comment).visit(module['select_type_routine'].body)
    assert len(comments) == 2
    assert 'Some comment' in comments[0].text
    assert 'inline comment' in comments[1].text


@pytest.mark.parametrize('frontend', available_frontends())
def test_select_case(frontend):
    """
    Test `SELECT CASE` parsing into multi-conditional IR.
    """
    fcode = """
subroutine select_case_routine(kind, out)
  implicit none
  integer, intent(in) :: kind
  integer, intent(out) :: out

  select case (kind)
  case (1)
    out = 10
  case (2:4)
    out = 20
  case default
    out = 30
  end select
end subroutine select_case_routine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    multi_conds = FindNodes(ir.MultiConditional).visit(routine.body)
    assert len(multi_conds) == 1
    assert multi_conds[0].expr == 'kind'
    assert len(multi_conds[0].values) == 2
    assert len(multi_conds[0].bodies) == 2
    assert len(multi_conds[0].else_body) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_do_while_loop(frontend):
    """
    Test `DO WHILE` parsing.
    """
    fcode = """
subroutine do_while_routine(n, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: out
  integer :: i

  out = 0
  i = 1
  do while (i <= n)
    out = out + i
    i = i + 1
  end do
end subroutine do_while_routine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    whiles = FindNodes(ir.WhileLoop).visit(routine.body)
    assert len(whiles) == 1
    assert whiles[0].condition == 'i <= n'
    assigns = FindNodes(ir.Assignment).visit(whiles[0].body)
    assert len(assigns) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_control_flow_statements(frontend):
    """
    Test Fortran statements that define control flow in loop bodies.
    """
    fcode = """
subroutine control_flow_routine(n, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: out
  integer :: i

  out = 0
  do i=1, n
    if (i == 2) cycle
    if (i == 3) continue
    if (i == 4) go to 42
    if (i == 5) then
42    exit
    end if
    if (i == 6) stop
    out = out + i
  end do
end subroutine control_flow_routine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1
    statements = FindNodes(ir.GenericStmt).visit(loops[0].body)

    assert len(statements) == 5
    assert isinstance(statements[0], ir.CycleStmt)
    assert isinstance(statements[1], ir.ContinueStmt)
    assert isinstance(statements[2], ir.GotoStmt)
    assert statements[2].text == '42'
    assert isinstance(statements[3], ir.ExitStmt)
    assert isinstance(statements[4], ir.StopStmt)


@pytest.mark.parametrize('frontend', available_frontends())
def test_where_construct(frontend):
    """
    Test `WHERE` handling for frontends that lower it to intrinsic/raw bodies.
    """
    fcode = """
subroutine where_routine(mask, a, b)
  implicit none
  logical, intent(in) :: mask(:)
  real, intent(inout) :: a(:), b(:)

  where (mask)
    a = b
  elsewhere
    a = 0.
  end where
end subroutine where_routine
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    masked = FindNodes(ir.MaskedStatement).visit(routine.body)
    assert len(masked) == 1
    assert len(masked[0].conditions) == 1
    assert masked[0].conditions[0] == 'mask'
    assert len(masked[0].bodies) == 1
    assert len(masked[0].default) == 1
