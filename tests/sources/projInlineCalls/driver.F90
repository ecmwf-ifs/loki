subroutine driver(n, work)
  use some_module, only: return_one, some_var, add_args, some_type
  use vars_module, only: varA, varB
  implicit none

  interface
     real function double_real(i)
       integer, intent(in) :: i
     end function double_real
  end interface

  integer, intent(in) :: n
  real, intent(out) :: work(n)
  type(some_type) :: var
  integer :: i

  do i=1,n
    work(i) = double_real(i) + return_one()
    work(i) = work(i) + dble(some_var)
    work(i) = work(i) + add_args(i,1) + add_args(i,2)
    call var%do_something(work(i))
  enddo

end subroutine driver
