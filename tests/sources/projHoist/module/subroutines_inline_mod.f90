module subroutines_inline_mod
    implicit none
    integer, parameter :: len = 3
contains

    function func1(a)
        integer, intent(in) :: a
        integer :: func1(a)

        func1 = 1
    end function func1

    subroutine kernel1(a, b, c)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        integer, intent(inout) :: c(a, len)
        real :: x(a)
        integer :: y(a, len)
        real :: k1_tmp(a, len)
        y = 11
        c = y
        x = func1(a)
    end subroutine kernel1

    subroutine kernel2(a1, b)
        integer, intent(in) :: a1
        integer, intent(inout) :: b(a1)
        real :: x(a1)
        real :: y, z
        real :: k2_tmp(a1, a1)
        call device1(a1, b, x, k2_tmp)
        call device2(a1, b, x)
    end subroutine kernel2

    subroutine device1(a1, b, x, y)
        integer, intent(in) :: a1
        integer, intent(inout) :: b(a1)
        real, intent(inout) :: x(a1)
        real, intent(inout) :: y(a1, a1)
        real :: z
        integer :: d1_tmp
        call device2(a1, b, x)
        call device2(a1, b, x)
    end subroutine device1

    function init_int(a2) result(TMP)
        integer, intent(in) :: a2
        integer :: tmp(a2)
        integer :: tmp0(a2)

        tmp0 = 42
        tmp = tmp0
    end function init_int

    subroutine device2(a2, b, x)
        integer, intent(in) :: a2
        integer, intent(inout) :: b(a2)
        real, intent(inout) :: x(a2)
        integer z(a2)
        integer :: d2_tmp(len,len)
        z = init_int(a2)
        b = z
    end subroutine device2

end module subroutines_inline_mod

