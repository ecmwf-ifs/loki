module driver

    use kernel, only: kernel_routine
    use types, only: parent_type

    contains

    subroutine driver_routine()
        type(parent_type) :: pt

        integer :: summed, i, size

        size = 100

        pt%member = 12
        ALLOCATE(pt%type_member%x(size))
        do i=1,size
            pt%type_member%x(i) = 1
        end do

        call kernel_routine(size, pt)

        summed = 0
        do i=1,size
            summed = summed + pt%type_member%x(i)
        end do

        print*, "the sum is", summed

    end subroutine driver_routine

end module driver
