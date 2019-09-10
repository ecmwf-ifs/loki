
subroutine routine_axpy(a, x, y)
  implicit none
  ! A simple standard routine that computes x = a * x + y for
  ! scalar arguments
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: a, y
  real(kind=jprb), intent(inout) :: x

  x = a * x + y
end subroutine routine_axpy

subroutine routine_copy(x, y)
  implicit none
  ! A simple routine that copies the value of x to y 
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: x
  real(kind=jprb), intent(out) :: y

  y = x
end subroutine routine_copy

subroutine routine_fixed_loop(scalar, vector, vector_out, tensor)
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: n=6, m=4
  real(kind=real64), intent(in) :: scalar
  real(kind=real64), intent(in) :: tensor(n, m), vector(n) 
  real(kind=real64), intent(out) :: vector_out(n)
  integer :: i, j

  ! For testing, the operation is:
  do i=1, n
     vector_out(i) = vector(i) + tensor(i, 1) + 1.0
  end do

  ! do j=1, m
  !    do i=1, n
  !       tensor_out(i, j) = 10.* j + i
  !    end do
  ! end do
end subroutine routine_fixed_loop

subroutine routine_moving_average(length, data_in, data_out)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: length
  real(kind=real64), intent(in) :: data_in(length)
  real(kind=real64), intent(out) :: data_out(length)
  integer :: i
  real(kind=real64) :: prev, next, divisor

  !$loki dataflow
  do i=1, length
    divisor = 1.0
    if (i > 1) then
      prev = data_in(i-1)
      divisor = divisor + 1.0
    else
      prev = 0
    end if
    if (i < length) then
      next = data_in(i+1)
      divisor = divisor + 1.0
    else
      next = 0
    end if
    data_out(i) = (prev + data_in(i) + next) / divisor
  end do
end subroutine routine_moving_average

subroutine routine_shift(length, scalar, vector_in, vector_out)
  implicit none
  ! A simple standard looking routine to test argument declarations
  ! and generator toolchain
  integer, intent(in) :: length, scalar, vector_in(length)
  integer, intent(inout) :: vector_out(length)
  integer :: i

  do i=1, length
    vector_out(i) = vector_in(i) + scalar
  end do
end subroutine routine_shift
