MODULE KERNEL_A4_REPL_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_a4_repl(b, c, flag1, flag2_renamed)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)
    LOGICAL, INTENT(IN) :: flag1, flag2_renamed

  END SUBROUTINE kernel_a4_repl

END MODULE KERNEL_A4_REPL_MOD
