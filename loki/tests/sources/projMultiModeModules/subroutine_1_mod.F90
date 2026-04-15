module subroutine_1_mod
implicit none
contains
subroutine subroutine_1()
  use nested_subroutine_1_mod, only: nested_subroutine_1
  call nested_subroutine_1()
end subroutine subroutine_1
end module subroutine_1_mod
