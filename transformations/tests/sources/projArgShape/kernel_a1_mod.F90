MODULE KERNEL_A1_MOD
IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_a1(b, c)
    ! Second-level kernel call
    REAL, INTENT(INOUT)   :: b(:,:)
    REAL, INTENT(INOUT)   :: c(:,:)

  END SUBROUTINE kernel_a1

END MODULE KERNEL_A1_MOD
