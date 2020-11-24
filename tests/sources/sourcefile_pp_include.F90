subroutine routine_pp_include(a, b, c)
  implicit none
  real(kind=4), intent(in) :: a, b
  real(kind=4), intent(out) :: c

#include "some_header.h"
  c = add(a, b)
end subroutine routine_pp_include
