
SUBROUTINE simple_expr(v1, v2, v3, v4, v5, v6)
  ! Simple floating point arithmetic
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
  REAL(KIND=JPRB), intent(in) :: v1, v2, v3, v4
  REAL(KIND=JPRB), intent(out) :: v5, v6

  v5 = (v1 + v2) * (v3 - v4)
  v6 = (v1 ** v2) - (v3 / v4)

END SUBROUTINE simple_expr


SUBROUTINE intrinsic_functions(v1, v2, vmin, vmax, vabs, vexp, vsqrt, vlog)
  ! Test supported intrinsic functions (InlineFunction)
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
  REAL(KIND=JPRB), intent(in) :: v1, v2
  REAL(KIND=JPRB), intent(out) :: vmin, vmax, vabs, vexp, vsqrt, vlog

  vmin = min(v1, v2)
  vmax = max(v1, v2)
  vabs = abs(v1 - v2)
  vexp = exp(v1 + v2)
  vsqrt = sqrt(v1 + v2)
  vlog = log(v1 + v2)

END SUBROUTINE intrinsic_functions
