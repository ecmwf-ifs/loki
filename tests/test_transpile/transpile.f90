
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

  a = 2**3
  b = 3.2_real32
  c = 4.1_real64

  a_io = a_io + 2
  b_io = b_io + real(3.2, kind=real32)
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


! subroutine transpile_derived_type_array(a_struct)
!   use transpile_type, only: array_struct
!   implicit none
!      ! real(kind=real64) :: vector(:)
!      ! real(kind=real64) :: matrix(:,:)
!   type(array_struct), intent(inout) :: a_struct
!   integer :: i, j

!   a_struct%scalar = 3.
!   do i=1, 3
!     a_struct%vector(i) = a_struct%scalar + 2.
!   end do
!   do i=1, 3
!     do j=1, 3
!       a_struct%matrix(j,i) = a_struct%vector(i) + 1.
!     end do
!   end do

! end subroutine transpile_derived_type_array


subroutine transpile_associates(a_struct)
  use transpile_type, only: my_struct
  implicit none
  type(my_struct), intent(inout) :: a_struct

  associate(a_struct_a=>a_struct%a, a_struct_b=>a_struct%b,&
   & a_struct_c=>a_struct%c)
  a_struct%a = a_struct_a + 4.
  a_struct_b = a_struct%b + 5.
  a_struct_c = a_struct_a + a_struct%b + a_struct_c
  end associate

end subroutine transpile_associates 


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


subroutine transpile_vectorization(n, m, scalar, v1, v2)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n, m
  real(kind=real64), intent(inout) :: scalar
  real(kind=real64), intent(inout) :: v1(n), v2(n)

  real(kind=real64) :: matrix(n, m)

  integer :: i

  v1(:) = scalar + 1.0
  matrix(:, :) = scalar + 2.
  v2(:) = matrix(:, 2)
  v2(1) = 1.

end subroutine transpile_vectorization


subroutine transpile_intrinsics(v1, v2, v3, v4, vmin, vmax, vabs, vmin_nested, vmax_nested)
  ! Test supported intrinsic functions
  use iso_fortran_env, only: real64
  real(kind=real64), intent(in) :: v1, v2, v3, v4
  real(kind=real64), intent(out) :: vmin, vmax, vabs, vmin_nested, vmax_nested

  vmin = min(v1, v2)
  vmax = max(v1, v2)
  vabs = abs(v1 - v2)
  vmin_nested = min(min(v1, v2), min(v3, v4))
  vmax_nested = max(max(v1, v2), max(v3, v4))

end subroutine transpile_intrinsics


subroutine transpile_loop_indices(n, idx, mask1, mask2, mask3)
  ! Test to ensure loop indexing translates correctly
  use iso_fortran_env, only: real64
  integer, intent(in) :: n, idx
  integer, intent(inout) :: mask1(n), mask2(n)
  real(kind=real64), intent(inout) :: mask3(n)

  integer :: i

  do i=1, n
     if (i < idx) then
        mask1(i) = 1
     end if

     if (i == idx) then
        mask1(i) = 2
     end if

     mask2(i) = i
  end do

  mask3(n) = 3.0

end subroutine transpile_loop_indices


subroutine transpile_logical_statements(v1, v2, v_xor, v_xnor, v_nand, v_neqv, v_val)
  logical, intent(in) :: v1, v2
  logical, intent(out) :: v_xor, v_nand, v_xnor, v_neqv, v_val(2)
  
  v_xor = (v1 .and. .not. v2) .or. (.not. v1 .and. v2)
  v_xnor = v1 .eqv. v2
  v_nand = .not. (v1 .and. v2)
  v_neqv = v1 .neqv. v2
  v_val(1) = .true.
  v_val(2) = .false.

end subroutine transpile_logical_statements
