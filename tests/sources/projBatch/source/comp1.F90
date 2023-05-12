subroutine comp1 (arg, val)
    use t_mod, only: t, nt1
    use header_mod
    implicit none
    type(t), intent(inout) :: arg
    real(kind=k), intent(inout) :: val(:)
    integer :: jnt1
#include "comp2.intfb.h"
    call arg%proc()
    call comp2(arg, val)
    call comp2(arg, val)  ! Twice to check we're not duplicating dependencies
    do jnt1=1,nt1
        call arg%no(jnt1)%way(.true.)
    end do
end subroutine comp1
