subroutine routine_okay
integer(kind=jpim) :: i, j
real(kind=jprb) :: a, b

i = 1_JPIM + 7_JPIM
j = 2_JPIM
a = 3._JPRB
b = 4.0_JPRB
end subroutine routine_okay

subroutine routine_not_okay
integer :: i
integer(kind=1) :: j
real :: a
real(kind=8) :: b

i = 1 + 7
j = 2
a = 3e0
b = 4.0 + 5d0 + 6._4
end subroutine routine_not_okay
