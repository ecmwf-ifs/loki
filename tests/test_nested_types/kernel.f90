module kernel

    use types, only: parent_type

    implicit none

    public kernel_routine

    contains

    subroutine kernel_routine(size, pt)
        integer, intent(in) :: size
        type(parent_type), intent(inout) :: pt

        integer :: i

        do i=1,size
            pt%type_member%x(i) = pt%member*pt%type_member%x(i)
        end do

    end subroutine kernel_routine

end module kernel
