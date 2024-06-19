MODULE KERNEL_MOD
    IMPLICIT NONE
    CONTAINS
    SUBROUTINE kernel(start, iend, nlon, nz, q, t, z)
        INTEGER, INTENT(IN) :: start
        INTEGER, INTENT(IN) :: iend  ! Iteration indices
        INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
        REAL, INTENT(INOUT) :: t(nlon,nz)
        REAL, INTENT(INOUT) :: q(nlon,nz)
        REAL, INTENT(INOUT) :: z(nlon,nz)
        REAL    :: local_z(nlon, nz)
        INTEGER :: jl, jk
        REAL :: c

        c = 5.345
        DO jk = 2, nz
          DO jl = start, iend
            call ELEMENTAL_DEVICE(z(jl, jk))
          END DO
        END DO

        call DEVICE(nlon, nz, 2, start, iend, z)

        c = 5.345
        DO jk = 2, nz
          DO jl = start, iend
            t(jl, jk) = c * jk
            q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
          END DO
        END DO

        DO jk = 2, nz
          DO jl = start, iend
            local_z(jl, jk) = 0.0
            z(jl, jk) = local_z(jl, jk)
          END DO
        END DO

    END SUBROUTINE kernel

    PURE ELEMENTAL SUBROUTINE ELEMENTAL_DEVICE(x) ! elemental
      REAL, INTENT(INOUT) :: x
      x = 0.0
    END SUBROUTINE ELEMENTAL_DEVICE

    SUBROUTINE DEVICE(nlon, nz, jk_start, start, iend, x)
        INTEGER, INTENT(IN) :: jk_start, start
        INTEGER, INTENT(IN) :: iend
        INTEGER, INTENT(IN) :: nlon, nz
        REAL, INTENT(INOUT) :: x(nlon, nz)
        REAL    :: local_x(nlon, nz)
        INTEGER :: jk, jl
        DO jk = jk_start, nz
            DO jl = start, iend
                local_x(jl, jk) = 0.0
                x(jl, jk) = local_x(jl, jk)
            END DO
        END DO
    END SUBROUTINE DEVICE

END MODULE KERNEL_MOD
