program simple_loop

implicit  none

integer, parameter :: dp = selected_real_kind(15, 307)
integer :: i, n, m
real(kind=dp) :: v_scalar
real(kind=dp) :: v_vector(3), v_tensor(3, 4)

n = 3
m = 4
v_scalar = 2.0
v_vector(:) = 3.0
v_tensor(:, :) = 4.0

call tester(n, m, v_scalar, v_vector, v_tensor)
write(*, *) "Tester::Scalar: ", v_scalar
write(*, *) "Tester::Vector: ", v_vector
do i=1, n
   write(*, *) "Tester::Tensor: ", v_tensor(i, :)
end do

v_scalar = 2.0
v_vector(:) = 3.0
v_tensor(:, :) = 4.0

call wrapper(n, m, v_scalar, v_vector, v_tensor)
write(*, *) "FC::Scalar: ", v_scalar
write(*, *) "FC::Vector: ", v_vector
do i=1, n
   write(*, *) "FC::Tensor: ", v_tensor(i, :)
end do

end program simple_loop


subroutine tester(n, m, v_scalar, v_vector, v_tensor)

  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, intent(in) :: n, m
  real(kind=dp), intent(inout) :: v_scalar
  real(kind=dp), intent(inout) :: v_vector(n), v_tensor(n, m)

  integer :: i, j

  ! For testing, the operation is:
  do i=1, n
     v_vector(i) = v_vector(i) + v_tensor(i, 1) + 1.0
  end do

  do j=1, m
     do i=1, n
        v_tensor(i, j) = 10.*real(j) + real(i)
     end do
  end do

end subroutine tester


subroutine wrapper(n, m, v_scalar, v_vector, v_tensor)

  implicit none

  interface
     subroutine simple_loop_fc(n, m, v_scalar, v_vector, v_tensor) &
          & bind(c, name='simple_loop_c')
       use iso_c_binding, only: c_int, c_double, c_ptr
       implicit none

       integer(kind=c_int), value :: n, m
       ! Not sure if this is hacky or not
       ! Instead we could do type(c_ptr) here and cast with c_loc below
       real(kind=c_double), value :: v_scalar
       real(kind=c_double) :: v_vector(n), v_tensor(n, m)
     end subroutine simple_loop_fc
  end interface

  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, intent(in) :: n, m
  real(kind=dp), intent(inout) :: v_scalar
  real(kind=dp), intent(inout) :: v_vector(n), v_tensor(n, m)

  call simple_loop_fc(n, m, v_scalar, v_vector, v_tensor)

end subroutine wrapper
