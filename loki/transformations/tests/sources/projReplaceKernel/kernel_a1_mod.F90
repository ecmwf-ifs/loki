MODULE KERNEL_A1_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_a1(b, c, flag1, flag2)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)
    LOGICAL, INTENT(IN) :: flag1, flag2

  END SUBROUTINE kernel_a1

END MODULE KERNEL_A1_MOD
