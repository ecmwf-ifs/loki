module KERNELA_MOD
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1

  implicit none

contains

  subroutine KERNELA(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

#include "another_l1.intfb.h"

    call COMPUTE_L1(vector)

    call ANOTHER_L1(matrix)

  end subroutine KERNELA

end module KERNELA_MOD
