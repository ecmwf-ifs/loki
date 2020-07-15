subroutine another_l1(arrayB)
  use header_mod, only: jprb

  implicit none

  real(kind=jprb), intent(inout) :: arrayB(:)

#include "another_l2.intfb.h"

  call another_l2(arrayB)

end subroutine another_l1
