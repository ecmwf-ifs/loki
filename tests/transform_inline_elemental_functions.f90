SUBROUTINE transform_inline_elemental_functions (v1, v2, v3)
  USE iso_fortran_env, ONLY: real64
  USE multiply_mod, ONLY: multiply
  REAL(KIND=real64), INTENT(IN) :: v1
  REAL(KIND=real64), INTENT(OUT) :: v2, v3
  
  v2 = multiply(v1, 6._real64)
  v3 = 600. + multiply(6._real64, 11._real64)
END SUBROUTINE transform_inline_elemental_functions
