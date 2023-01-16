module transformation_module_hoist
    USE subroutines_mod, only: kernel1, kernel2, device1, device2
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

end module transformation_module_hoist

