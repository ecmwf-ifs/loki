subroutine depth2(h_start, h_end, h_dim, v_start, v_dim, array)
    implicit none
    integer, intent(in) :: h_start
    integer, intent(in) :: h_end
    integer, intent(in) :: h_dim
    integer, intent(in) :: v_start
    integer, intent(in) :: v_dim
    real, intent(inout) :: array(h_dim, v_dim)

    ! Non-array local variables.
    integer :: v_index
    integer :: h_index
    real :: val

    ! Temporary arrays.
    real :: tmp2(h_dim, v_dim)

    val = 1.0
    call contained(val)
    tmp2 = 2.0 + val
    tmp2(h_dim, v_dim) = -1.0

    do v_index = v_start, v_dim
        do h_index = h_start, h_end
            array(h_index, v_index) = array(h_index, v_index) + tmp2(h_index, v_index)
        end do
    end do
contains

    subroutine contained(x)
        real, intent(inout) :: x
        x = x * 2.0
    end subroutine contained

end subroutine depth2
