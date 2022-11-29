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
        !    kernel1(a, b, c, kernel1_x, kernel1_y, kernel1_k1_tmp)
        call kernel2(a, b)
        !    kernel2(a, b, kernel2_x, kernel2_y, kernel2_z, kernel2_k2_tmp, device1_z, device1_d1_tmp, device2_z, device2_d2_tmp)
        call kernel1(a, b, c)
        !    kernel1(a, b, c, kernel1_x, kernel1_y, kernel1_k1_tmp)
    end subroutine driver

    !          kernel1 (a, b, c, x, y, k1_tmp)
    subroutine kernel1(a, b, c)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        integer, intent(inout) :: c(a, len)
        real :: x(a)
        integer :: y(a, len)
        real :: k1_tmp(a, len)
        y = 11
        c = y
        ! call device2(x)
    end subroutine kernel1

    !          kernel2 (a, b, x, y, z, k2_tmp, device1_z, device1_d1_tmp, device2_z, device2_d2_tmp)
    subroutine kernel2(a, b)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        real :: x(a)
        real :: y, z
        real :: k2_tmp(a, a)
        call device1(a, b, x, k2_tmp)
        !    device1(a, b, x, k2_temp, device1_z, device1_d1_tmp, device2_z, device2_d2_tmp)
    end subroutine kernel2

    !          device1 (a, b, x, y, z, d1_tmp, device2_z, device2_d2_tmp)
    subroutine device1(a, b, x, y)
        integer, intent(in) :: a
        integer, intent(inout) :: b(a)
        real, intent(inout) :: x(a)
        real, intent(inout) :: y(a, a)
        real :: z
        integer :: d1_tmp
        call device2(a, b, x)
        !    device2(a, b, x, device2_z, device2_d2_tmp)
        call device2(a, b, x)
        !    device2(a, b, x, device2_z, device2_d2_tmp)
    end subroutine device1

    !          device2 (a, b, x, z, d2_tmp)
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