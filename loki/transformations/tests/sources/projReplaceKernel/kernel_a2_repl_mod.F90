MODULE KERNEL_A2_REPL_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_a2_repl(b)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)

  END SUBROUTINE kernel_a2_repl

END MODULE KERNEL_A2_REPL_MOD
