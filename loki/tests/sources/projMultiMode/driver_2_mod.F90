module driver_2_mod
  implicit none

contains

   subroutine driver_2()
#include "subroutine_2.intfb.h"
     call subroutine_2()
   end subroutine driver_2
end module driver_2_mod
