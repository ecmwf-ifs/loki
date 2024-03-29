module kernelD_mod
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1
  use proj_c_util_mod

  implicit none

contains

  subroutine kernelD(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

    call compute_l1(vector)

    call routine_one(matrix)
  end subroutine kernelD

end module kernelD_mod
