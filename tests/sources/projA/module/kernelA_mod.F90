module kernelA_mod
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1

  implicit none

contains

  subroutine kernelA(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

#include "another_l1.intfb.h"

    call compute_l1(vector)

    call another_l1(matrix)

  end subroutine kernelA

end module kernelA_mod
