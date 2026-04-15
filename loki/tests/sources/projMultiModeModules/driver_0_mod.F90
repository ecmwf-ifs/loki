module driver_0_mod
  implicit none
contains

   subroutine driver_0()
     use subroutine_1_mod, only: subroutine_1
     use subroutine_3_mod, only: subroutine_3
     call subroutine_1()
     call subroutine_3()
     call subroutine_3()
   end subroutine driver_0
end module driver_0_mod
