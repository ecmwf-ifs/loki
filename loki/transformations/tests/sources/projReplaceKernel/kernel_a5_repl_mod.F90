MODULE KERNEL_A5_REPL_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_a5_repl(b1, c, flag1, flag2)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b1(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)
    LOGICAL, INTENT(IN) :: flag1, flag2

  END SUBROUTINE kernel_a5_repl

END MODULE KERNEL_A5_REPL_MOD
