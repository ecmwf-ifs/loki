module base

  implicit none

  save

  ! TODO: Using this in this module causes issues in f90wrap,
  ! so we leave it here for other modules to use, but don't
  ! use it in this module itself.
  integer, parameter :: jprb = selected_real_kind(13,300)

  real(kind=8) :: a, b
  integer :: i, j

contains

  function a_plus_b()
    ! Test to verify module-level variables work
    real(kind=8) :: a_plus_b
    a_plus_b = a + b
  end function a_plus_b

  function a_times_b_plus_c(a, b, c)
    ! Simple test to verify that module functions work
    real(kind=8) :: a_times_b_plus_c
    real(kind=8), intent(in) :: a, b, c
    a_times_b_plus_c = a * b + c
  end function a_times_b_plus_c

end module base
