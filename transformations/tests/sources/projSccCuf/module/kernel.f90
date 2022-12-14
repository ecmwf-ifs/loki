MODULE KERNEL_MOD
    IMPLICIT NONE
    CONTAINS
    SUBROUTINE kernel(start, end, nlon, nz, q, t, z)
        INTEGER, INTENT(IN) :: start, end  ! Iteration indices
        INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
        REAL, INTENT(INOUT) :: t(nlon,nz)
        REAL, INTENT(INOUT) :: q(nlon,nz)
        REAL, INTENT(INOUT) :: z(nlon,nz)
        REAL    :: local_z(nlon, nz)
        INTEGER :: jl, jk
        REAL :: c

        c = 5.345
        DO jk = 2, nz
          DO jl = start, end
            t(jl, jk) = c * k
            q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
          END DO
        END DO

        DO jk = 2, nz
          DO jl = start, end
            local_z(jl, jk) = 0.0
            z(jl, jk) = local_z(jl, jk)
          END DO
        END DO

        ! DO JL = START, END
        !   Q(JL, NZ) = Q(JL, NZ) * C
        ! END DO
    END SUBROUTINE kernel
END MODULE KERNEL_MOD