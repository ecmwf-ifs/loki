
subroutine transpile_simple_loops(n, m, scalar, vector, tensor)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n, m
  real(kind=real64), intent(inout) :: scalar
  real(kind=real64), intent(inout) :: vector(n), tensor(n, m)

  integer :: i, j

  ! For testing, the operation is:
  do i=1, n
     vector(i) = vector(i) + tensor(i, 1) + 1.0
  end do

  do j=1, m
     do i=1, n
        tensor(i, j) = 10.* j + i
     end do
  end do

end subroutine transpile_simple_loops


subroutine transpile_arguments(n, array, array_io, a, b, c, a_io, b_io, c_io)
  use iso_fortran_env, only: real32, real64
  implicit none

  integer, intent(in) :: n
  real(kind=real64), intent(inout) :: array(n)
  real(kind=real64), intent(out) :: array_io(n)

  integer, intent(out) :: a
  real(kind=real32), intent(out) :: b
  real(kind=real64), intent(out) :: c
  integer, intent(inout) :: a_io
  real(kind=real32), intent(inout) :: b_io
  real(kind=real64), intent(inout) :: c_io

  integer :: i

  do i=1, n
     array(i) = 3.
     array_io(i) = array_io(i) + 3.
  end do

  a = 2
  b = 3.2
  c = 4.1

  a_io = a_io + 2
  b_io = b_io + 3.2
  c_io = c_io + 4.1

end subroutine transpile_arguments


subroutine transpile_derived_type(a_struct)
  use transpile_type, only: my_struct
  implicit none
    ! integer, intent(in) :: n
    ! integer, intent(inout) :: an_int
    ! logical, intent(inout) :: bools(n)
    ! Should test float/double as arrays
    ! real(kind=real32), intent(inout) :: a_float(i)
    ! real(kind=real64), intent(inout) :: a_double(i)
  type(my_struct), intent(inout) :: a_struct

    ! an_int = an_int + 1
    ! a_float = a_float + 2.
    ! a_double = a_double + 3.
    ! bools(:) = .FALSE.
  a_struct%a = a_struct%a + 4
  a_struct%b = a_struct%b + 5.
  a_struct%c = a_struct%c + 6.

end subroutine transpile_derived_type


subroutine transpile_module_variables(a, b, c)
  use iso_fortran_env, only: real32, real64
  use transpile_type, only: param1, param2, param3

  integer, intent(out) :: a
  real(kind=real32), intent(out) :: b
  real(kind=real64), intent(out) :: c

  a = 1 + param1
  b = 1. + param2
  c = 1. + param3

end subroutine transpile_module_variables
