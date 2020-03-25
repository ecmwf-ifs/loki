subroutine routine_okay
integer(kind=jpim) :: i, j
real(kind=jprb) :: a(3), b

i = 1_JPIM + 7_JPIM
j = 2_JPIM
a(1:3) = 3._JPRB
b = 4.0_JPRB
end subroutine routine_okay

subroutine routine_not_okay
integer :: i
integer(kind=1) :: j
real :: a(3)
real(kind=8) :: b

i = 1 + 7
j = 2
a(1:3) = 3e0
b = 4.0 + 5d0 + 6._4
end subroutine routine_not_okay
