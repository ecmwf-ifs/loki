subroutine comp1 (arg, val)
    use t_mod, only: t
    use header_mod
    implicit none
    type(t), intent(inout) :: arg
    real(kind=k), intent(inout) :: val(:)
#include "comp2.intfb.h"
    call arg%proc()
    call comp2(arg, val)
    call arg%no%way()
end subroutine comp1
