MODULE DRIVER_MOD
USE KERNEL_A_MOD, ONLY: KERNEL_A
USE KERNEL_B_MOD, ONLY: KERNEL_B
IMPLICIT NONE
CONTAINS
  SUBROUTINE driver(nlon, nlev, a, b, c)
    INTEGER, INTENT(IN)   :: nlon, nlev  ! Dimension sizes
    INTEGER, PARAMETER    :: n = 5
    REAL, INTENT(INOUT)   :: a(nlon)
    REAL, INTENT(INOUT)   :: b(nlon,nlev)
    REAL, INTENT(INOUT)   :: c(nlon,n)

    call kernel_a(a, b, c)

    call kernel_b(b, c)
  END SUBROUTINE driver

END MODULE DRIVER_MOD
