module driver_2_mod
  implicit none

contains

   subroutine driver_2()
     use subroutine_2_mod, only: subroutine_2
     call subroutine_2()
   end subroutine driver_2
end module driver_2_mod
