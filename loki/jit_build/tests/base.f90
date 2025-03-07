module base

  implicit none

  save

  integer, parameter :: jprb = selected_real_kind(13,300)

  real(kind=jprb) :: a, b
  integer :: i, j

contains

  function a_plus_b()
    ! Test to verify module-level variables work
    real(kind=jprb) :: a_plus_b
    a_plus_b = a + b
  end function a_plus_b

  function a_times_b_plus_c(a, b, c)
    ! Simple test to verify that module functions work
    real(kind=jprb) :: a_times_b_plus_c
    real(kind=jprb), intent(in) :: a, b, c
    a_times_b_plus_c = a * b + c
  end function a_times_b_plus_c

end module base
