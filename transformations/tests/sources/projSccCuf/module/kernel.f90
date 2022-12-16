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


        call DEVICE2(nlon, nz, 2, start, end, z)

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

    SUBROUTINE DEVICE1(x) ! elemental
        REAL, INTENT(INOUT) :: x
        x = 0.0
    END SUBROUTINE DEVICE1

    !SUBROUTINE DEVICE2(jk, jl, x)
    !    INTEGER, INTENT(IN) :: jk, jl
    !    REAL, INTENT(INOUT) :: x(:, :)
    !    x(jk, jl) = 0.0
    !END SUBROUTINE DEVICE2

    SUBROUTINE DEVICE2(nlon, nz, jk_start, jl_start, jl_end, x)
        INTEGER, INTENT(IN) :: jk_start, jl_start, jl_end
        INTEGER, INTENT(IN) :: nlon, nz
        REAL, INTENT(INOUT) :: x(nlon, nz)
        REAL    :: local_x(nlon, nz)
        INTEGER :: jk, jl
        DO jk = jk_start, nz
            DO jl = jl_start, jl_end
                local_x(jk, jl) = 0.0
                x(jk, jl) = local_x(jk, jl)
            END DO
        END DO
    END SUBROUTINE DEVICE2

END MODULE KERNEL_MOD