MODULE KERNEL_A_MOD
IMPLICIT NONE
CONTAINS
  SUBROUTINE kernel_a(a, b, c)
    USE KERNEL_A1_MOD, ONLY: KERNEL_A1

    REAL, INTENT(INOUT)   :: a(:)
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)
    LOGICAL :: flag1, flag2

    CALL kernel_a1(b, c, flag1=flag1, flag2=flag2)
  END SUBROUTINE kernel_a

END MODULE KERNEL_A_MOD
