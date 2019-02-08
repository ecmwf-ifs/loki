
subroutine simple_expr(v1, v2, v3, v4, v5, v6)
  ! simple floating point arithmetic
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: v1, v2, v3, v4
  real(kind=jprb), intent(out) :: v5, v6

  v5 = (v1 + v2) * (v3 - v4)
  v6 = (v1 ** v2) - (v3 / v4)

end subroutine simple_expr


subroutine intrinsic_functions(v1, v2, vmin, vmax, vabs, vexp, vsqrt, vlog)
  ! test supported intrinsic functions (inlinefunction)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: v1, v2
  real(kind=jprb), intent(out) :: vmin, vmax, vabs, vexp, vsqrt, vlog

  vmin = min(v1, v2)
  vmax = max(v1, v2)
  vabs = abs(v1 - v2)
  vexp = exp(v1 + v2)
  vsqrt = sqrt(v1 + v2)
  vlog = log(v1 + v2)

end subroutine intrinsic_functions


subroutine logical_expr(t, f, vand_t, vand_f, vor_t, vor_f, vnot_t, vnot_f, vtrue, vfalse, veq, vneq)
  logical, intent(in) :: t, f
  logical, intent(out) :: vand_t, vand_f, vor_t, vor_f, vnot_t, vnot_f, vtrue, vfalse, veq, vneq

  vand_t = t .and. t
  vand_f = t .and. f
  vor_t = t .or. f
  vor_f = f .or. f
  vnot_t = .not. f
  vnot_f = .not. t
  vtrue = .true.
  vfalse = .false.
  veq = 3 == 4
  vneq = 3 /= 4

end subroutine logical_expr


subroutine literal_expr(v1, v2, v3, v4)
  ! simple literal values
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(out) :: v1, v2, v3, v4

  v1 = 66
  v2 = 66.0
  v3 = 2.3
  v4 = 2.4_jprb

end subroutine literal_expr


subroutine cast_expr(v1, v2, v3, v4, v5)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: v1
  real(kind=jprb), intent(in) :: v2, v3
  real(kind=jprb), intent(out) :: v4, v5

  v4 = real(v1, kind=jprb)  ! Test a plain cast
  v5 = real(v1, kind=jprb) * max(v2, v3)  ! Cast as part of expression
end subroutine cast_expr


subroutine logical_array(dim, arr, out)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: dim
  real(kind=jprb), intent(in) :: arr(dim)
  real(kind=jprb), intent(out) :: out(dim)
  logical :: mask(dim)
  integer :: i

  mask(:) = .true.
  mask(1) = .false.
  mask(2) = .false.

  do i=1, dim
    ! Use a logical array and a relational
    ! containing an array in a single expression
    if (mask(i) .and. arr(i) > 1.) then
      out(i) = 3.
    else
      out(i) = 1.
    end if
  end do

end subroutine logical_array


subroutine parenthesis(v1, v2, v3)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), intent(in) :: v1, v2
  real(kind=jprb), intent(out) :: v3

  v3 = (v1**1.23_jprb) * 1.3_jprb + (1_jprb - v2**1.26_jprb)

end subroutine parenthesis


subroutine commutativity(v1, v2, v3)
  integer, parameter :: jprb = selected_real_kind(13,300)
  real(kind=jprb), pointer, intent(in) :: v1(:), v2
  real(kind=jprb), pointer, intent(out) :: v3(:)

  v3(:) = 1._jprb + v2*v1(:) - v2 - v3(:)

end subroutine commutativity
