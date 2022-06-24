MODULE multiply_mod
  USE iso_fortran_env, ONLY: real64
  IMPLICIT NONE
CONTAINS
  ELEMENTAL FUNCTION multiply (a, b)
    REAL(KIND=real64) :: multiply
    REAL(KIND=real64), INTENT(IN) :: a, b
    
    multiply = a*b
  END FUNCTION multiply
END MODULE multiply_mod
