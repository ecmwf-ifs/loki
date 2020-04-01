subroutine routine_okay
implicit none
integer :: a
a = 5
contains
subroutine contained_routine_okay
integer :: b
b = 5
end subroutine contained_routine_okay
end subroutine routine_okay

module mod_okay
implicit none
contains
subroutine contained_mod_routine_okay
integer :: a
a = 5
end subroutine contained_mod_routine_okay
end module mod_okay

subroutine routine_not_okay
! This should report
integer :: a
a = 5
contains
subroutine contained_not_okay_routine_okay
implicit none
integer :: b
b = 5
end subroutine contained_not_okay_routine_okay
end subroutine routine_not_okay

module mod_not_okay
contains
subroutine contained_mod_not_okay_routine_okay
implicit none
integer :: a
a = 5
end subroutine contained_mod_not_okay_routine_okay
end module mod_not_okay

subroutine routine_also_not_okay
! This should report
integer :: a
a = 5
contains
subroutine contained_routine_not_okay
! This should report
integer :: b
b = 5
end subroutine contained_routine_not_okay
end subroutine routine_also_not_okay

module mod_also_not_okay
contains
subroutine contained_mod_routine_not_okay
! This should report
integer :: a
a = 5
contains
subroutine contained_contained_routine_not_okay
! This should report
integer :: b
b = 5
end subroutine contained_contained_routine_not_okay
end subroutine contained_mod_routine_not_okay
end module mod_also_not_okay
