
MODULE driver_mod
    USE KERNEL_MOD, ONLY: KERNEL
    IMPLICIT NONE
CONTAINS
    SUBROUTINE driver(nlon, nz, nb, tot, q, t, z)
        INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
        INTEGER, INTENT(IN)   :: tot
        REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
        REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
        REAL, INTENT(INOUT)   :: z(nlon,nz+1,nb)
        INTEGER :: b, start, iend, ibl, icend

        start = 1
        iend = tot
        do b=1,iend,nlon
          ibl = (b-1)/nlon+1
          icend = MIN(nlon,tot-b+1)
          call kernel(start, icend, nlon, nz, q(:,:,b), t(:,:,b), z(:,:,b))
        end do

        do b=1,iend,nlon
          ibl = (b-1)/nlon+1
          icend = MIN(nlon,tot-b+1)
          call kernel(start, icend, nlon, nz, q(:,:,b), t(:,:,b), z(:,:,b))
          call kernel(start, icend, nlon, nz, q(:,:,b), t(:,:,b), z(:,:,b))
        end do

    END SUBROUTINE driver
END MODULE driver_mod
