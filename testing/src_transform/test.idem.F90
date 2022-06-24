!module test_mod
!
!contains
!
MODULE test_IDEM_MOD
CONTAINS
  SUBROUTINE test_IDEM (klon, kidia, kfdia, klev, var_in, var_out)
    USE parkind1, ONLY: jpim, jprb
    
    IMPLICIT NONE
    
    !-------------
    !    arguments
    !-------------
    
    INTEGER(KIND=jpim), INTENT(IN) :: klon, kidia, kfdia
    INTEGER(KIND=jpim), INTENT(IN) :: klev
    
    REAL(KIND=jprb), INTENT(IN) :: var_in(klon, klev)
    REAL(KIND=jprb), INTENT(OUT) :: var_out(klon, klev)
    
    
    !-------------------
    !    local variables
    !-------------------
    
    INTEGER(KIND=jpim), INTENT(IN) :: jk, jl
    
    DO jk=1,klev
      DO jl=kidia,kfdia
        var_out(jl, jk) = var_in(jl, jk)
      END DO
    END DO
    
    
  END SUBROUTINE test_IDEM
END MODULE test_IDEM_MOD
!end module
