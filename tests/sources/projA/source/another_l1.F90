subroutine another_l1(matrix)
  use header_mod, only: jprb

  implicit none

  real(kind=jprb), intent(inout) :: matrix(:,:)

#include "another_l2.intfb.h"

  call another_l2(matrix)

end subroutine another_l1
