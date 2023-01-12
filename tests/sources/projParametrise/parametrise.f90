module parametrise
    implicit none
contains

    subroutine stop_execution(msg)
        character(200), INTENT(IN) :: msg
        PRINT *, msg
        stop 1
    end subroutine stop_execution

    subroutine driver(a, b, c, d)
        integer, intent(inout) :: a, b, d(b)
        integer, intent(inout) :: c(a, b)
        real :: x, y
        call kernel1(a, b, c)
        call kernel2(a, b, d)
        call kernel1(a, b, c)
    end subroutine driver

    subroutine another_driver(a, b, c)
        integer, intent(in) :: a
        integer, intent(in) :: b
        integer, intent(inout) :: c(a, b)
        real :: x(a)
        call kernel1(a, b, c)
    end subroutine another_driver

    subroutine kernel1(a, b, c)
        integer, intent(in) :: a
        integer, intent(in) :: b
        integer, intent(inout) :: c(a, b)
        integer :: local_a
        real :: x(a)
        integer :: y(a, b)
        real :: k1_tmp(a, b)
        y = 11
        c = y
    end subroutine kernel1

    subroutine kernel2(a_new, b, d)
        integer, intent(in) :: a_new
        integer, intent(in) :: b
        integer, intent(inout) :: d(b)
        real :: x(a_new)
        real :: y, z
        real :: k2_tmp(a_new, a_new)
        call device1(a_new, b, d, x, k2_tmp)
    end subroutine kernel2

    subroutine device1(a, b, d, x, y)
        integer, intent(in) :: a
        integer, intent(in) :: b
        integer, intent(inout) :: d(b)
        real, intent(inout) :: x(a)
        real, intent(inout) :: y(a, a)
        real :: z
        integer :: d1_tmp
        call device2(a, b, d, x)
        call device2(a, b, d, x)
    end subroutine device1

    subroutine device2(a, b, d, x)
        integer, intent(in) :: a
        integer, intent(in) :: b
        integer, intent(inout) :: d(b)
        real, intent(inout) :: x(a)
        integer z(b)
        integer :: d2_tmp(b)
        z = 42
        d = z
    end subroutine device2

end module parametrise
