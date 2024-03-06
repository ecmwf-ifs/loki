subroutine kernel0()
use func_mod, only: some_func
implicit none
  real a
  call kernel1()
  call kernel2()
  call kernel3()
  ! print *, "result: ", some_func()
  a = some_func()
end subroutine kernel0

subroutine kernel1()
use moduleA, only: var0,var1,some_func
implicit none
real :: tmp

tmp = var0 + var1 + some_func()

end subroutine kernel1

subroutine kernel2()
use moduleB, only: var2,var3
implicit none
real :: tmp

tmp = var2 + var3

end subroutine kernel2

subroutine kernel3()
use moduleB, only: var2,var3
use moduleC, only: var4,var5
implicit none

var4 = var2
var5 = var3

end subroutine kernel3

! real function some_func()
! use moduleB, only: var2
! implicit none
! real some_func
! some_func = var2
! end function some_func
