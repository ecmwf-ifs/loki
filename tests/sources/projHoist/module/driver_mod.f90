module transformation_module_hoist
    implicit none
    integer, parameter :: len = 3
contains

    subroutine driver(a, b, c)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        integer, intent(inout) :: c(a, len)
        real :: x, y
        call kernel1(a, b, c)
        call kernel2(a, b)
        call kernel1(a, b, c)
    end subroutine driver

    subroutine another_driver(a, b, c)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        integer, intent(inout) :: c(a, len)
        real :: x, y
        call kernel1(a, b, c)
    end subroutine another_driver

    subroutine kernel1(a, b, c)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        integer, intent(inout) :: c(a, len)
        real :: x(a)
        integer :: y(a, len)
        real :: k1_tmp(a, len)
        y = 11
        c = y
    end subroutine kernel1

    subroutine kernel2(a, b)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        real :: x(a)
        real :: y, z
        real :: k2_tmp(a, a)
        call device1(a, b, x, k2_tmp)
        call device2(a, b, x)
    end subroutine kernel2

    subroutine device1(a, b, x, y)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        real, intent(inout) :: x(a)
        real, intent(inout) :: y(a, a)
        real :: z
        integer :: d1_tmp
        call device2(a, b, x)
        call device2(a, b, x)
    end subroutine device1

    subroutine device2(a, b, x)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        real, intent(inout) :: x(a)
        integer z(a)
        integer :: d2_tmp(len)
        z = 42
        b = z
    end subroutine device2

end module transformation_module_hoist