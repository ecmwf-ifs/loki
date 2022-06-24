MODULE test_SCA_MOD
CONTAINS
  SUBROUTINE test_SCA (klev, var_in, var_out)
    USE parkind1, ONLY: jpim, jprb
    
    IMPLICIT NONE
    
    !-------------
    !    arguments
    !-------------
    
    INTEGER(KIND=jpim), INTENT(IN) :: klev
    
    
    
    !-------------------
    !    local variables
    !-------------------
    
    INTEGER(KIND=jpim), INTENT(IN) :: jk
    REAL(KIND=jprb), INTENT(IN) :: var_in(klev)
    REAL(KIND=jprb), INTENT(OUT) :: var_out(klev)
    
    DO jk=1,klev
      var_out(jk) = var_in(jk)
    END DO
    
    DO jk=1,klev
      var_out(jk) = 2._JPRB*var_out(jk)
    END DO
    
    DO jk=1,klev
      var_out(jk) = var_out(jk) - var_in(jk)
    END DO
    
  END SUBROUTINE test_SCA
END MODULE test_SCA_MOD
