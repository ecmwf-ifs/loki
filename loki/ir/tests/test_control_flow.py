# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from loki import Subroutine
from loki.backend import fgen
from loki.build import jit_compile, clean_test
from loki.frontend import available_frontends, OMNI, FP
from loki.ir import nodes as ir, FindNodes


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
        assert FindNodes(ir.Loop).visit(routine.ir)[0].loop_label == '101'

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

    conditionals = FindNodes(ir.Conditional).visit(routine.body)
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
    conditionals = FindNodes(ir.Conditional).visit(routine.ir)
    assert len(conditionals) == 5
    assert all(
        c.body and isinstance(c.body, tuple) and all(isinstance(n, ir.Node) for n in c.body)
        for c in conditionals
    )
    assert all(
        c.else_body and isinstance(c.else_body, tuple) and all(isinstance(n, ir.Node) for n in c.else_body)
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
    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conditionals) == 2
    assert isinstance(conditionals[0].body[-1], ir.Intrinsic)
    assert conditionals[0].body[-1].text.upper() == 'RETURN'
    assert conditionals[0].else_body == (conditionals[1],)
    assert isinstance(conditionals[1].body[-1], ir.Intrinsic)
    assert conditionals[1].body[-1].text.upper() == 'RETURN'
    assert isinstance(conditionals[1].else_body[-1], ir.Intrinsic)
    assert conditionals[1].else_body[-1].text.upper() == 'RETURN'


@pytest.mark.parametrize('frontend', [FP])
def test_single_line_forall_stmt(tmp_path, frontend):
    fcode = """
subroutine forall_stmt(n, a)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: n
  real(kind=jprb), dimension(n, n), intent(inout) :: a

  ! Create a diagonal square matrix
  forall (i=1:n)  a(i, i) = 1
end subroutine forall_stmt
    """.strip()
    filepath = tmp_path/f'single_line_forall_stmt_{frontend}.f90'
    routine = Subroutine.from_source(fcode, frontend=frontend)
    fun_forall_stmt = jit_compile(routine, filepath=filepath, objname="forall_stmt")

    # Check generated IR for the Forall statement
    statements = FindNodes(ir.Forall).visit(routine.ir)
    assert len(statements) == 1
    # Check the i=1:n bound
    assert len(statements[0].named_bounds) == 1
    bound_var, bound_range = statements[0].named_bounds[0]
    assert bound_var.name == "i"
    assert bound_range == '1:n'
    # Check the a(i, i) = 1 assignment
    assignments = FindNodes(ir.Assignment).visit(statements[0])
    assert len(assignments) == 1, "Single-line FORALL statement must have only one assignment"
    assert assignments[0].lhs == "a(i, i)"  # Assign to array `a`
    assert assignments[0].rhs == '1'  # Assign 1 on the diagonal

    # Check execution and produced results
    n = 3
    a = np.zeros((n, n), order="F")
    fun_forall_stmt(n, a)
    assert (a == [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]).all()
    n = 5
    a = np.empty((n, n), order="F")
    a.fill(3.0)
    fun_forall_stmt(n, a)
    assert (a == [[1.0, 3.0, 3.0, 3.0, 3.0],
                  [3.0, 1.0, 3.0, 3.0, 3.0],
                  [3.0, 3.0, 1.0, 3.0, 3.0],
                  [3.0, 3.0, 3.0, 1.0, 3.0],
                  [3.0, 3.0, 3.0, 3.0, 1.0]]).all()

    # Check the fgen code generation
    expected_fcode = "FORALL(i = 1:n) a(i, i) = 1"
    assert fgen(statements[0]) == expected_fcode
    assert expected_fcode in routine.to_fortran()


@pytest.mark.parametrize('frontend', [FP])
def test_single_line_forall_masked_stmt(tmp_path, frontend):
    fcode = """
subroutine forall_masked_stmt(n, a, b)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: n
  real(kind=jprb), dimension(n, n), intent(inout) :: a, b

  forall(i = 1:n, j = 1:n, a(i, j) .ne. 0.0) b(i, j) = 1.0 / a(i, j)
end subroutine forall_masked_stmt
    """.strip()
    filepath = tmp_path / (f'single_line_forall_masked_stmt_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    fun_forall_masked_stmt = jit_compile(routine, filepath=filepath, objname="forall_masked_stmt")

    # Check generated IR for the Forall statement
    statements = FindNodes(ir.Forall).visit(routine.ir)
    assert len(statements) == 1
    assert len(statements[0].named_bounds) == 2
    # Check the i=1:n bound
    bound_var, bound_range = statements[0].named_bounds[0]
    assert bound_var == "i"
    assert bound_range == '1:n'
    # Check the j=1:n bound
    bound_var, bound_range = statements[0].named_bounds[1]
    assert bound_var == "j"
    assert bound_range == '1:n'
    # Check the array mask
    mask = statements[0].mask
    assert statements[0].mask == 'a(i, j) != 0.0'
    # Quickly check assignment
    assignments = FindNodes(ir.Assignment).visit(statements[0])
    assert len(assignments) == 1
    assert assignments[0].lhs.name == "b" and len(assignments[0].lhs.dimensions) == 2
    assert assignments[0].rhs == '1.0 / a(i, j)'

    # Check execution and produced results
    n = 3
    a = np.array([[2.0, 0.0, 2.0],
                  [0.0, 4.0, 0.0],
                  [10.0, 10.0, 0.0]], order="F")
    b = np.zeros((n, n), order="F")
    fun_forall_masked_stmt(n, a, b)
    assert (b == [[0.5, 0.0, 0.5], [0, 0.25, 0], [0.1, 0.1, 0]]).all()

    # Check the fgen code generation
    expected_fcode = "FORALL(i = 1:n, j = 1:n, a(i, j) /= 0.0) b(i, j) = 1.0 / a(i, j)"
    assert fgen(statements[0]) == expected_fcode
    assert expected_fcode in routine.to_fortran()


@pytest.mark.parametrize('frontend', [FP])
def test_multi_line_forall_construct(tmp_path, frontend):
    fcode = """
subroutine forall_construct(n, c, d)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: n
  real(kind=jprb), dimension(n, n), intent(inout) :: c, d

  forall(i = 3:n - 2, j = 3:n - 2)
    c(i, j) = c(i, j + 2) + c(i, j - 2) + c(i + 2, j) + c(i - 2, j)
    d(i, j) = c(i, j)
  end forall
end subroutine forall_construct
    """.strip()
    filepath = tmp_path / (f'multi_line_forall_construct_{frontend}.f90')
    routine = Subroutine.from_source(fcode, frontend=frontend)
    fun_forall_construct = jit_compile(routine, filepath=filepath, objname="forall_construct")

    # Check generated IR for the Forall statement
    statements = FindNodes(ir.Forall).visit(routine.ir)
    assert len(statements) == 1
    assert len(statements[0].named_bounds) == 2
    # Check the i=3:(n-2) bound
    bound_var, bound_range = statements[0].named_bounds[0]
    assert bound_var.name == "i"
    assert bound_range == '3:n-2'
    # Check the j=3:(n-2) bound
    bound_var, bound_range = statements[0].named_bounds[1]
    assert bound_var.name == "j"
    assert bound_range == '3:n-2'
    # Check assignments
    assignments = FindNodes(ir.Assignment).visit(statements[0])
    assert len(assignments) == 2
    # Quickly check first assignment
    assert assignments[0].lhs == 'c(i, j)'
    assert assignments[0].rhs == 'c(i, j + 2) + c(i, j - 2) + c(i + 2, j) + c(i - 2, j)'
    # Check the second assignment
    assert assignments[1].lhs == 'd(i, j)'
    assert assignments[1].rhs == 'c(i, j)'

    n = 6
    c = np.zeros((n, n), order="F")
    c.fill(1)
    d = np.zeros((n, n), order="F")
    fun_forall_construct(n, c, d)
    assert (c == [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 4.0, 4.0, 1.0, 1.0],
                  [1.0, 1.0, 4.0, 4.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).all()
    assert (d == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 4.0, 4.0, 0.0, 0.0],
                  [0.0, 0.0, 4.0, 4.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).all()

    # Check the fgen code generation
    regenerated_code = routine.to_fortran().split("\n")
    assert regenerated_code[5].strip() == "FORALL(i = 3:n - 2, j = 3:n - 2)"
    assert regenerated_code[6].strip() == "c(i, j) = c(i, j + 2) + c(i, j - 2) + c(i + 2, j) + c(i - 2, j)"
    assert regenerated_code[7].strip() == "d(i, j) = c(i, j)"
    assert regenerated_code[8].strip() == "END FORALL"
