module math_operations
  implicit none

  public :: add, subtract, multiply

contains

  ! Subroutine to add two numbers
  subroutine add(x, y, result)
    real, intent(in) :: x, y
    real, intent(out) :: result
    result = x + y
  end subroutine add

  ! Subroutine to subtract two numbers
  subroutine subtract(x, y, result)
    real, intent(in) :: x, y
    real, intent(out) :: result
    result = x - y
  end subroutine subtract

  ! Subroutine to multiply two numbers
  subroutine multiply(x, y, result)
    real, intent(in) :: x, y
    real, intent(out) :: result
    result = x * y
  end subroutine multiply

end module math_operations
