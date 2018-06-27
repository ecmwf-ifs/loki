
SUBROUTINE loop_nest_fixed(in1, in2, out1, out2)
  ! Test basic loops and reductions with fixed sizes
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
  REAL(KIND=JPRB), dimension(3, 2), intent(in) :: in1, in2
  REAL(KIND=JPRB), intent(inout) :: out1(3, 2), out2(2)

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

END SUBROUTINE loop_nest_fixed


SUBROUTINE loop_nest_variable(dim1, dim2, in1, in2, out1, out2)
  ! Test basic loops and reductions with passed sizes
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
  INTEGER, intent(in) :: dim1, dim2
  REAL(KIND=JPRB), dimension(dim1, dim2), intent(in) :: in1, in2
  REAL(KIND=JPRB), intent(inout) :: out1(dim1, dim2), out2(dim2)

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

END SUBROUTINE loop_nest_variable
