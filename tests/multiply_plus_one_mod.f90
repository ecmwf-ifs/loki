MODULE multiply_plus_one_mod
  USE iso_fortran_env, ONLY: real64
  IMPLICIT NONE
CONTAINS
  ELEMENTAL FUNCTION multiply (a, b)
    REAL(KIND=real64) :: multiply
    REAL(KIND=real64), INTENT(IN) :: a, b
    
    multiply = a*b
  END FUNCTION multiply
  ELEMENTAL FUNCTION plus_one (a)
    REAL(KIND=real64) :: plus_one
    REAL(KIND=real64), INTENT(IN) :: a
    
    plus_one = a + 1._real64
  END FUNCTION plus_one
  ELEMENTAL FUNCTION multiply_plus_one (a, b)
    REAL(KIND=real64) :: multiply_plus_one
    REAL(KIND=real64), INTENT(IN) :: a, b
    
    multiply_plus_one = multiply(plus_one(a), b)
  END FUNCTION multiply_plus_one
END MODULE multiply_plus_one_mod
