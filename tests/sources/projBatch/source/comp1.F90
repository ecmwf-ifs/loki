subroutine comp1 (arg, val)
    use t_mod, only: t
    use header_mod, only: k
    implicit none
    type(t), intent(inout) :: arg
    real(kind=k), intent(inout) :: val(:)
#include "comp2.intfb.h"
    call comp2(arg, val)
end subroutine comp1
