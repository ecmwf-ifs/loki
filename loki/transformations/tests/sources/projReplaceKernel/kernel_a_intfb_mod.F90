MODULE KERNEL_A_INTFB_MOD
IMPLICIT NONE
CONTAINS
  SUBROUTINE kernel_a_intfb(a, b, c)

#include "kernel_a1.intfb.h"

    REAL, INTENT(INOUT)   :: a(:)
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)
    LOGICAL :: flag1, flag2

    CALL kernel_a1(b, c, flag1=flag1, flag2=flag2)
  END SUBROUTINE kernel_a_intfb

END MODULE KERNEL_A_INTFB_MOD
