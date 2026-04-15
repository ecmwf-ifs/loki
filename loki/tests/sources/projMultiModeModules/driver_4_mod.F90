module driver_4_mod
  implicit none

contains

   subroutine driver_4()
     use subroutine_3_mod, only: subroutine_3
     call subroutine_3()
   end subroutine driver_4
end module driver_4_mod
