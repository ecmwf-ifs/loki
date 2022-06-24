
SUBROUTINE test_driver (klon, kidia, kfdia, klev, var_in, var_out)
  USE parkind1, ONLY: jpim, jprb
  !use test_mod, only : test
  
  IMPLICIT NONE
  
#include "test_driver.intfb.h"
  
  !-------------
  !    arguments
  !-------------
  
  INTEGER(KIND=jpim), INTENT(IN) :: klon, kidia, kfdia
  INTEGER(KIND=jpim), INTENT(IN) :: klev
  
  REAL(KIND=jprb), INTENT(INOUT) :: var_in(klon, klev)
  REAL(KIND=jprb), INTENT(INOUT) :: var_out(klon, klev)
  
  
  !-------------------
  !    local variables
  !-------------------
  
  var_in = 1._JPRB
  var_out = 0._JPRB
  
  
  CALL test_IDEM(klon, kidia, kfdia, klev, var_in, var_out)
  
  
END SUBROUTINE test_driver
