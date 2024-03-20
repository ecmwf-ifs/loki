subroutine wrapper
    implicit none

    integer :: h_start = 1
    integer, parameter :: h_dim = 10
    integer :: h_end 
    integer :: v_start = 1
    integer, parameter :: v_dim = 20
    integer, parameter :: block_size = 30
    integer :: block_index

    real :: arr(h_dim, v_dim, block_dim)

    h_end = h_dim
    do block_index = 1, block_size 
        call depth1(h_start, h_end, h_dim, v_start, v_dim, &
                    arr(:, :, block_index))
    end do
    print *, "Sum of array is ", sum(arr)
end subroutine wrapper
