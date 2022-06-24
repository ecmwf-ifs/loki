MODULE transform_inline_constant_parameters_kind_mod
  IMPLICIT NONE
CONTAINS
  SUBROUTINE transform_inline_constant_parameters_kind (v1)
    USE kind_parameters_mod, ONLY: jprb
    REAL(KIND=jprb), INTENT(OUT) :: v1
    
    v1 = REAL(2, kind=jprb) + 3.
  END SUBROUTINE transform_inline_constant_parameters_kind
END MODULE transform_inline_constant_parameters_kind_mod
