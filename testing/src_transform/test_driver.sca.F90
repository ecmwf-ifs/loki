
SUBROUTINE test_driver (klon, kidia, kfdia, klev, var_in, var_out)
  USE test_SCA_MOD, ONLY: test_SCA
  USE parkind1, ONLY: jpim, jprb
  
  IMPLICIT NONE
  
  
  !-------------
  !    arguments
  !-------------
  
  INTEGER(KIND=jpim), INTENT(IN) :: klon, kidia, kfdia
  INTEGER(KIND=jpim), INTENT(IN) :: klev
  
  REAL(KIND=jprb), INTENT(INOUT) :: var_in(klon, klev)
  REAL(KIND=jprb), INTENT(INOUT) :: var_out(klon, klev)
  INTEGER(KIND=JPIM) :: JL
  
  
  !-------------------
  !    local variables
  !-------------------
  
  var_in = 1._JPRB
  var_out = 0._JPRB
  
  
  DO JL=kidia,kfdia
    CALL test_SCA(klev, var_in(JL, :), var_out(JL, :))
  END DO
  
  
END SUBROUTINE test_driver
