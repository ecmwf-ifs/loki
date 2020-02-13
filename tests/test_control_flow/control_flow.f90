
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


subroutine goto_stmt(var)
  implicit none
  integer, intent(out) :: var
  var = 3
  go to 1234
  var = 5
  1234 return
  var = 7
end subroutine
