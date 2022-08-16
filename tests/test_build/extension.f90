subroutine extended_fma(a, b, c, sum)
  ! Add number from an imported module
  use base, only: jprb, a_times_b_plus_c

  implicit none
  real(kind=jprb), intent(in) :: a, b, c
  real(kind=jprb), intent(out) :: sum

  sum = a_times_b_plus_c(a, b, c)
end subroutine extended_fma
