MODULE KERNEL_B_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_b(b, c)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

  END SUBROUTINE kernel_b
END MODULE KERNEL_B_MOD
