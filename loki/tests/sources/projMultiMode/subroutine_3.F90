subroutine subroutine_3()
  #include "nested_subroutine_1.intfb.h"
  #include "nested_subroutine_3.intfb.h"

! INTERFACE
!   SUBROUTINE nested_subroutine_3()
!   END SUBROUTINE nested_subroutine_3
! END INTERFACE

  call nested_subroutine_1()
  call nested_subroutine_3()
end subroutine subroutine_3
