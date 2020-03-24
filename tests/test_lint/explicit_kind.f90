subroutine routine_okay
integer(kind=jpim) :: i, j
real(kind=jprb) :: a, b

i = 1_JPIM + 7_JPIM
j = 2_JPIM
a = 3_JPRB
b = 4.0_JPRB
end subroutine routine_okay

subroutine routine_not_okay
integer :: i, j
real :: a, b

i = 1 + 7
j = 2
a = 3
b = 4.0
end subroutine routine_not_okay
