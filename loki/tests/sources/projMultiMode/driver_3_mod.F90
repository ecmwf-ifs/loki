module driver_3_mod
  implicit none

contains

   subroutine driver_3()
     #include "subroutine_3.intfb.h"
     call subroutine_3()
   end subroutine driver_3
end module driver_3_mod
