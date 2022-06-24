MODULE transform_inline_constant_parameters_replace_kind_mod
  IMPLICIT NONE
CONTAINS
  SUBROUTINE transform_inline_constant_parameters_replace_kind (v1)
    USE replace_kind_parameters_mod, ONLY: jprb
    REAL(KIND=jprb), INTENT(OUT) :: v1
    REAL(KIND=jprb) :: a = 3._JPRB
    
    v1 = 1._jprb + REAL(2, kind=jprb) + a
  END SUBROUTINE transform_inline_constant_parameters_replace_kind
END MODULE transform_inline_constant_parameters_replace_kind_mod
