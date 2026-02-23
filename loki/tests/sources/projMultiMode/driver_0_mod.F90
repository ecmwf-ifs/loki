module driver_0_mod
  implicit none
contains

   subroutine driver_0()
     #include "subroutine_1.intfb.h"
     #include "subroutine_3.intfb.h"
     call subroutine_1()
     call subroutine_3()
     call subroutine_3()
   end subroutine driver_0
end module driver_0_mod
