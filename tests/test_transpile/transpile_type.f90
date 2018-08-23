module transpile_type
  use iso_fortran_env, only: real32, real64

  implicit none

  save

  integer :: param1
  real(kind=real32) :: param2
  real(kind=real64) :: param3

  type my_struct
     integer :: a
     real(kind=real32) :: b
     real(kind=real64) :: c
  end type my_struct

end module transpile_type
