SUBROUTINE transform_inline_elemental_functions_ (v1, v2, v3)
  USE iso_fortran_env, ONLY: real64
  REAL(KIND=real64), INTENT(IN) :: v1
  REAL(KIND=real64), INTENT(OUT) :: v2, v3
  
  v2 = temp
  v3 = 600. + temp
END SUBROUTINE transform_inline_elemental_functions_
