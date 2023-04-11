subroutine driver(n, work)
  implicit none

  interface
     real function double_real(i)
       integer, intent(in) :: i
     end function double_real
  end interface

  integer, intent(in) :: n
  real, intent(out) :: work(n)
  integer :: i

  do i=1,n
    work(i) = double_real(i)
  enddo

end subroutine driver
