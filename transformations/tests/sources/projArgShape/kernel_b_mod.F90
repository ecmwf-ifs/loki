MODULE KERNEL_B_MOD
USE VAR_MODULE_MOD, only: n
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_b(b, c)
    ! USE VAR_MODULE_MOD, only: n
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

  END SUBROUTINE kernel_b
END MODULE KERNEL_B_MOD
