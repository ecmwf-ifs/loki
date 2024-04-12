# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from loki import OMNI, Subroutine, FindNodes, Loop, Conditional, Node, Intrinsic
from loki.build import jit_compile, clean_test
from loki.frontend import available_frontends


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_nest_fixed(here, frontend):
    """
    Test basic loops and reductions with fixed sizes.

    Basic loop nest loop:
        out1(i, j) = in1(i, j) + in2(i, j)

    Basic reduction:
        out2(j) = out2(j) + in1(i, j) * in1(i, j)
    """

    fcode = """
subroutine loop_nest_fixed(in1, in2, out1, out2)

  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), dimension(3, 2), intent(in) :: in1, in2
  real(kind=jprb), intent(inout) :: out1(3, 2), out2(2)
  integer :: i, j

  do j=1, 2
     do i=1, 3
        out1(i, j) = in1(i, j) + in2(i, j)
     end do
  end do

  do j=1, 2
     out2(j) = 0.
     do i=1, 3
        out2(j) = out2(j) + in1(i, j) * in2(i, j)
     end do
  end do
end subroutine loop_nest_fixed
"""
    filepath = here/(f'control_flow_loop_nest_fixed_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='loop_nest_fixed')

    in1 = np.array([[1., 2.], [2., 3.], [3., 4.]], order='F')
    in2 = np.array([[2., 3.], [3., 4.], [4., 5.]], order='F')
    out1 = np.zeros((3, 2), order='F')
    out2 = np.zeros(2, order='F')

    function(in1, in2, out1, out2)
    assert (out1 == [[3, 5], [5, 7], [7, 9]]).all()
    assert (out2 == [20, 38]).all()
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_nest_variable(here, frontend):
    """
    Test basic loops and reductions with passed sizes.

    Basic loop nest loop:
        out1(i, j) = in1(i, j) + in2(i, j)

    Basic reduction:
        out2(j) = out2(j) + in1(i, j) * in1(i, j)
    """

    fcode = """
subroutine loop_nest_variable(dim1, dim2, in1, in2, out1, out2)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: dim1, dim2
  real(kind=jprb), dimension(dim1, dim2), intent(in) :: in1, in2
  real(kind=jprb), intent(inout) :: out1(dim1, dim2), out2(dim2)

  integer :: i, j

  do j=1, dim2
     do i=1, dim1
        out1(i, j) = in1(i, j) + in2(i, j)
     end do
  end do

  do j=1, dim2
     out2(j) = 0.
     do i=1, dim1
        out2(j) = out2(j) + in1(i, j) * in2(i, j)
     end do
  end do
end subroutine loop_nest_variable
"""
    filepath = here/(f'control_flow_loop_nest_variable_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='loop_nest_variable')

    in1 = np.array([[1., 2.], [2., 3.], [3., 4.]], order='F')
    in2 = np.array([[2., 3.], [3., 4.], [4., 5.]], order='F')
    out1 = np.zeros((3, 2), order='F')
    out2 = np.zeros(2, order='F')

    function(3, 2, in1, in2, out1, out2)
    assert (out1 == [[3, 5], [5, 7], [7, 9]]).all()
    assert (out2 == [20, 38]).all()
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_scalar_logical_expr(here, frontend):
    """
    Test a while loop with a logical expression as condition.
    """

    fcode = """
subroutine loop_scalar_logical_expr(outvar)
  integer, intent(out) :: outvar

  outvar = 0
  do while (outvar < 5)
    outvar = outvar + 1
  end do
end subroutine loop_scalar_logical_expr
"""
    filepath = here/(f'control_flow_loop_scalar_logical_expr_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='loop_scalar_logical_expr')

    outvar = function()
    assert outvar == 5
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_unbounded(here, frontend):
    """
    Test unbounded loops.
    """

    fcode = """
subroutine loop_unbounded(out)
  integer, intent(out) :: out

  out = 1
  do
    out = out + 1
    if (out > 5) then
      exit
    endif
  enddo
end subroutine loop_unbounded
"""
    filepath = here/(f'control_flow_loop_unbounded_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='loop_unbounded')

    outvar = function()
    assert outvar == 6
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_loop_labeled_continue(here, frontend):
    """
    Test labeled loops with continue statement.

    Note that this does not get represented 1:1 as we always insert ENDDO
    statements in fgen. But this does not harm the outcome as the resulting
    loop behaviour will still be the same.
    """

    fcode = """
subroutine loop_labeled_continue(out)
  integer, intent(out) :: out
  integer :: j

  out = 1
  do 101 j=1,10
    out = out + 1
101 continue
end subroutine loop_labeled_continue
"""
    filepath = here/(f'control_flow_loop_labeled_continue_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)

    if frontend != OMNI:  # OMNI doesn't read the Loop label...
        assert FindNodes(Loop).visit(routine.ir)[0].loop_label == '101'

    function = jit_compile(routine, filepath=filepath, objname='loop_labeled_continue')

    outvar = function()
    assert outvar == 11
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_conditionals(here, frontend):
    """
    Test the use of inline conditionals.
    """

    fcode = """
subroutine inline_conditionals(in1, in2, out1, out2)
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2

  out1 = in1
  out2 = in2

  if (in1 < 0) out1 = 0
  if (in2 > 5) out2 = 5
end subroutine inline_conditionals
"""
    filepath = here/(f'control_flow_inline_conditionals_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='inline_conditionals')

    in1, in2 = 2, 2
    out1, out2 = function(in1, in2)
    assert out1 == 2 and out2 == 2

    in1, in2 = -2, 10
    out1, out2 = function(in1, in2)
    assert out1 == 0 and out2 == 5
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_multi_body_conditionals(here, frontend):
    fcode = """
subroutine multi_body_conditionals(in1, out1, out2)
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
end subroutine multi_body_conditionals
"""
    filepath = here/(f'control_flow_multi_body_conditionals_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 4
    if frontend != OMNI:
        assert sum(int(cond.has_elseif) for cond in conditionals) == 2

    function = jit_compile(routine, filepath=filepath, objname='multi_body_conditionals')

    out1, out2 = function(5)
    assert out1 == 1 and out2 == 4

    out1, out2 = function(2)
    assert out1 == 1 and out2 == 2

    out1, out2 = function(-1)
    assert out1 == 1 and out2 == 0

    out1, out2 = function(10)
    assert out1 == 5 and out2 == 5
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_goto_stmt(here, frontend):
    fcode = """
subroutine goto_stmt(var)
  implicit none
  integer, intent(out) :: var
  var = 3
  go to 1234
  var = 5
  1234 return
  var = 7
end subroutine goto_stmt
"""
    filepath = here/(f'control_flow_goto_stmt_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='goto_stmt')

    result = function()
    assert result == 3
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_select_case(here, frontend):
    fcode = """
subroutine select_case(cmd, out1)
  implicit none
  integer, intent(in) :: cmd
  integer, intent(out) :: out1

  select case (cmd)
    case (0)
      out1 = 0
    case (1:9)
      out1 = 1
    case (10, 11)
      out1 = 2
    case default
      out1 = -1
  end select
end subroutine select_case
"""
    filepath = here/(f'control_flow_select_case_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='select_case')

    in_out_pairs = {0: 0, 1: 1, 2: 1, 5: 1, 9: 1, 10: 2, 11: 2, 12: -1}
    for cmd, ref in in_out_pairs.items():
        out1 = function(cmd)
        assert out1 == ref
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_select_case_nested(here, frontend):
    fcode = """
subroutine select_case(cmd, out1)
  implicit none
  integer, intent(in) :: cmd
  integer, intent(out) :: out1

  out1 = -1000

  ! comment 1
  select case (cmd)
    ! comment 2
    case (0)
      out1 = 0
    ! comment 3
    case (1:9)
      out1 = 1
      select case (cmd)
        case (2:3)
          out1 = out1 + 100
        case (4:5)
          out1 = out1 + 200
      end select
    ! comment 4
    ! comment 5

    ! comment 6
    case (10, 11)
      out1 = 2
    ! comment 7
    case default
      out1 = -1
  end select
end subroutine select_case
"""
    filepath = here/(f'control_flow_select_case_nested_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='select_case')

    in_out_pairs = {0: 0, 1: 1, 2: 101, 5: 201, 9: 1, 10: 2, 11: 2, 12: -1}
    for cmd, ref in in_out_pairs.items():
        out1 = function(cmd)
        assert out1 == ref
    clean_test(filepath)

    assert routine.to_fortran().count('! comment') == 7


@pytest.mark.parametrize('frontend', available_frontends())
def test_cycle_stmt(here, frontend):
    fcode = """
subroutine cycle_stmt(var)
  implicit none
  integer, intent(out) :: var
  integer :: i

  var = 0
  do i=1,10
    if (var > 5) cycle
    var = var + 1
  end do
end subroutine cycle_stmt
"""
    filepath = here/(f'control_flow_cycle_stmt_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    function = jit_compile(routine, filepath=filepath, objname='cycle_stmt')

    result = function()
    assert result == 6
    clean_test(filepath)


@pytest.mark.parametrize('frontend', available_frontends())
def test_conditional_bodies(frontend):
    """Verify that conditional bodies and else-bodies are tuples of :class:`Node`"""
    fcode = """
subroutine conditional_body(nanana, zzzzz, trololo, tralala, xoxoxoxo, yyyyyy, kidia, kfdia)
integer, intent(inout) :: nanana, zzzzz, trololo, tralala, xoxoxoxo, yyyyyy, kidia, kfdia
integer :: jlon
if (nanana == 1) then
    zzzzz = 1
else
    zzzzz = 4
end if
if (trololo == 1) then
    tralala = 1
else if (trololo == 2) then
    tralala = 2
else if (trololo == 3) then
    tralala = 3
else
    tralala = 4
end if
if (xoxoxoxo == 1) then
    do jlon = kidia, kfdia
        yyyyyy = 1
    enddo
else
    do jlon = kidia, kfdia
        yyyyyy = 4
    enddo
end if
end subroutine conditional_body
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    conditionals = FindNodes(Conditional).visit(routine.ir)
    assert len(conditionals) == 5
    assert all(
        c.body and isinstance(c.body, tuple) and all(isinstance(n, Node) for n in c.body)
        for c in conditionals
    )
    assert all(
        c.else_body and isinstance(c.else_body, tuple) and all(isinstance(n, Node) for n in c.else_body)
        for c in conditionals
    )


@pytest.mark.parametrize('frontend', available_frontends())
def test_conditional_else_body_return(frontend):
    fcode = """
FUNCTION FUNC(PX,KN)
IMPLICIT NONE
INTEGER,INTENT(INOUT) :: KN
REAL,INTENT(IN) :: PX
REAL :: FUNC
INTEGER :: J
REAL :: Z0, Z1, Z2
Z0= 1.0
Z1= PX
IF (KN == 0) THEN
  FUNC= Z0
  RETURN
ELSEIF (KN == 1) THEN
  FUNC= Z1
  RETURN
ELSE
  DO J=2,KN
    Z2= Z0+Z1
    Z0= Z1
    Z1= Z2
  ENDDO
  FUNC= Z2
  RETURN
ENDIF
END FUNCTION FUNC
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 2
    assert isinstance(conditionals[0].body[-1], Intrinsic)
    assert conditionals[0].body[-1].text.upper() == 'RETURN'
    assert conditionals[0].else_body == (conditionals[1],)
    assert isinstance(conditionals[1].body[-1], Intrinsic)
    assert conditionals[1].body[-1].text.upper() == 'RETURN'
    assert isinstance(conditionals[1].else_body[-1], Intrinsic)
    assert conditionals[1].else_body[-1].text.upper() == 'RETURN'
