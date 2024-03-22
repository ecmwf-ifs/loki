module kernelB_mod
  use header_mod, only: jprb
  use compute_l1_mod, only: compute_l1
#ifdef HAVE_EXT_DRIVER_MODULE
  use ext_driver_mod, only: ext_driver
#endif

  implicit none

contains

  subroutine kernelB(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:)

#ifndef HAVE_EXT_DRIVER_MODULE
#include "ext_driver.intfb.h"
#endif

    call compute_l1(vector)

    call ext_driver(matrix)
  end subroutine kernelB

end module kernelB_mod
