module kernelC_mod
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1
  use proj_c_util_mod, only: routine_one

  implicit none

contains

  subroutine kernelC(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

    call compute_l1(vector)

    call routine_one(matrix)
  end subroutine kernelC

end module kernelC_mod
