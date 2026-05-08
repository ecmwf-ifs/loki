MODULE KERNEL_A3_REPL_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_a3_repl(b, c, flag1)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)
    LOGICAL, INTENT(IN) :: flag1

  END SUBROUTINE kernel_a3_repl

END MODULE KERNEL_A3_REPL_MOD
