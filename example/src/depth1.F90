subroutine depth1(h_start, h_end, h_dim, v_start, v_dim, array)
    implicit none

    ! Arguments.
    integer, intent(in) :: h_start
    integer, intent(in) :: h_end
    integer, intent(in) :: h_dim
    integer, intent(in) :: v_start
    integer, intent(in) :: v_dim
    real, intent(inout) :: array(h_dim, v_dim)

    ! Non-array local variables.
    integer :: v_index
    integer :: h_index

    ! Temporary arrays.
    real :: tmp1(h_dim, v_dim)

    do v_index = v_start, v_dim
        do h_index = h_start, h_end
            tmp1(h_index, v_index) = exp(log(real(h_index)) + log(real(v_index)) - 1.0)
        end do
    end do

    do v_index = v_start, v_dim
        do h_index = h_start, h_end
            array(h_index, v_index) = exp(tmp1(h_index, v_index) + 0.25)
        end do
    end do

    call depth2(h_start, h_end, h_dim, v_start, v_dim, array)

    do v_index = v_start, v_dim
        do h_index = h_start, h_end
            array(h_index, v_index) = log(tmp1(h_index, v_index)) + array(h_index, v_index) * 2.0
        end do
    end do

end subroutine depth1
