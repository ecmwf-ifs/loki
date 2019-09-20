
subroutine routine_axpy(n, a, x, y)
  ! A simple standard routine that computes x = a * x + y for
  ! scalar arguments
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: n
  real(kind=jprb), intent(in) :: a, y(n)
  real(kind=jprb), intent(inout) :: x(n)

  x(:) = a * x(:) + y(:)
end subroutine routine_axpy

subroutine routine_shift(length, scalar, vector_in, vector_out)
  ! A simple standard looking routine to test argument declarations
  ! and generator toolchain
  integer, intent(in) :: length, scalar, vector_in(length)
  integer, intent(inout) :: vector_out(length)
  integer :: i

  do i=1, length
    vector_out(i) = vector_in(i) + scalar
  end do
end subroutine routine_shift
