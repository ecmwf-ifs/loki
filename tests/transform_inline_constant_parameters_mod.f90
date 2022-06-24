MODULE transform_inline_constant_parameters_mod
  ! TODO: use parameters_mod, only: b
  IMPLICIT NONE
  INTEGER, PARAMETER :: c = 1 + 1
CONTAINS
  SUBROUTINE transform_inline_constant_parameters (v1, v2, v3)
    USE parameters_mod, ONLY: a, b
    INTEGER, INTENT(IN) :: v1
    INTEGER, INTENT(OUT) :: v2, v3
    
    v2 = v1 + b - a
    v3 = c
  END SUBROUTINE transform_inline_constant_parameters
END MODULE transform_inline_constant_parameters_mod
