
subroutine test_driver(klon,kidia,kfdia,klev,var_in,var_out)
use parkind1, only : jpim,jprb

implicit none

#include "test.intfb.h"

!-------------
!    arguments
!-------------

integer(kind=jpim),intent(in) :: klon,kidia,kfdia
integer(kind=jpim),intent(in) :: klev

real(kind=jprb),   intent(inout) :: var_in (klon,klev)
real(kind=jprb),   intent(inout) :: var_out(klon,klev)


!-------------------
!    local variables
!-------------------

var_in  = 1._JPRB
var_out = 0._JPRB


call test(klon,kidia,kfdia,klev,var_in,var_out)


end subroutine test_driver
