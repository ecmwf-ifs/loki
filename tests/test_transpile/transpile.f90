
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
