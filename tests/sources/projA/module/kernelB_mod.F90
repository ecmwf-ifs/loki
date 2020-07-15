module kernelB_mod
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1
  use ext_driver_mod, only: ext_driver

  implicit none

contains

  subroutine kernelB(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

    call compute_l1(vector)

    call ext_driver(matrix)
  end subroutine kernelB

end module kernelB_mod
