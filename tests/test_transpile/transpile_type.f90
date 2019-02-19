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

  ! type array_struct
  !    real(kind=real64), allocatable :: scalar
  !    real(kind=real64), allocatable :: vector(:)
  !    real(kind=real64), allocatable :: matrix(:,:)
  ! end type array_struct

contains

  ! subroutine alloc_arrays(struct)
  !   type(array_struct), intent(inout) :: struct
  !   allocate(struct%vector(3))
  !   allocate(struct%matrix(3, 3))
  ! end subroutine alloc_arrays

  ! subroutine free_arrays(struct)
  !   type(array_struct), intent(inout) :: struct
  !   deallocate(struct%vector)
  !   deallocate(struct%matrix)
  ! end subroutine free_arrays

end module transpile_type
