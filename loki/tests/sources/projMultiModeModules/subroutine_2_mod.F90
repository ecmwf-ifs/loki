module subroutine_2_mod
implicit none
contains
subroutine subroutine_2()
  use nested_subroutine_2_mod, only: nested_subroutine_2
  use nested_subroutine_3_mod, only: nested_subroutine_3
  call nested_subroutine_2()
  call nested_subroutine_3()
end subroutine subroutine_2
end module subroutine_2_mod
