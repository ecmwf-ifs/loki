MODULE KERNEL_A6_REPL_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_a6_repl(b, c, flag1, flag2, optional_flag)
    REAL, INTENT(INOUT) :: b(:,:)
    REAL, INTENT(INOUT) :: c(:,:)
    LOGICAL, INTENT(IN) :: flag1, flag2
    LOGICAL, INTENT(IN), OPTIONAL :: optional_flag

  END SUBROUTINE kernel_a6_repl

END MODULE KERNEL_A6_REPL_MOD
