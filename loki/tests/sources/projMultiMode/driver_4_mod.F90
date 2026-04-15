module driver_4_mod
  implicit none

contains

   subroutine driver_4()
     #include "subroutine_3.intfb.h"
     call subroutine_3()
   end subroutine driver_4
end module driver_4_mod
