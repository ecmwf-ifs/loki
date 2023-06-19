module kernelE_mod
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1

  implicit none

contains

  ! Two kernel routines, but we process only one!

  subroutine kernelE(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

    call compute_l1(vector)

    call ghost_busters(vector)

  contains

    subroutine ghost_busters(vector)
      real(kind=jprb), intent(inout) :: vector(:)

      vector(:) = 42.0
    end subroutine ghost_busters

  end subroutine kernelE

  subroutine kernelET(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

    call compute_l2(vector)

  end subroutine kernelET

end module kernelE_mod
