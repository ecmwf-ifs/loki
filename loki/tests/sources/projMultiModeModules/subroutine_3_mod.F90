module subroutine_3_mod
implicit none
contains
subroutine subroutine_3()
  use nested_subroutine_1_mod, only: nested_subroutine_1
  use nested_subroutine_3_mod, only: nested_subroutine_3

  call nested_subroutine_1()
  call nested_subroutine_3()
end subroutine subroutine_3
end module subroutine_3_mod
