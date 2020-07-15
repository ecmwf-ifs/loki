module kernelA_mod
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1

  implicit none

contains

  subroutine kernelA(arrayA, arrayB)
    real(kind=jprb), intent(inout) :: arrayA(:)
    real(kind=jprb), intent(inout) :: arrayB(:)

#include "another_l1.intfb.h"

    call compute_l1(arrayA)

    call another_l1(arrayB)

  end subroutine kernelA

end module kernelA_mod
