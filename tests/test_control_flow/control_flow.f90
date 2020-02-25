
subroutine loop_nest_fixed(in1, in2, out1, out2)
  ! test basic loops and reductions with fixed sizes
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


subroutine loop_nest_variable(dim1, dim2, in1, in2, out1, out2)
  ! test basic loops and reductions with passed sizes
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


subroutine loop_scalar_logical_expr(outvar)
  integer, intent(out) :: outvar

  do while (outvar < 5)
    outvar = outvar + 1
  end do
end subroutine loop_scalar_logical_expr


subroutine inline_conditionals(in1, in2, out1, out2)
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2

  out1 = in1
  out2 = in2

  if (in1 < 0) out1 = 0
  if (in2 > 5) out2 = 5
end subroutine inline_conditionals


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


subroutine goto_stmt(var)
  implicit none
  integer, intent(out) :: var
  var = 3
  go to 1234
  var = 5
  1234 return
  var = 7
end subroutine goto_stmt


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
