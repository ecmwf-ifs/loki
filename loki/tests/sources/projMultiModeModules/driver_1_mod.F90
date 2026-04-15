module driver_1_mod
  implicit none
contains

   subroutine driver_1()
     use subroutine_1_mod, only: subroutine_1
     use subroutine_3_mod, only: subroutine_3
     call subroutine_1()
     call subroutine_3()
     call subroutine_3()
   end subroutine driver_1
end module driver_1_mod
