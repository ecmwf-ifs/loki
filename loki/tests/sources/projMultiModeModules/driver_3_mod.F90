module driver_3_mod
  implicit none

contains

   subroutine driver_3()
     use subroutine_3_mod, only: subroutine_3
     call subroutine_3()
   end subroutine driver_3
end module driver_3_mod
