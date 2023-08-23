subroutine kernel_derived_type()
use module_derived_type, only: p,p0,g,p_array
implicit none
integer :: i,j

do i=1,p%n
  p%x(i) = p0%x(i)
  p%y(i) = p0%y(i)
  p%z(i) = p0%z(i)
  do j=1,p%n
    p_array(i)%x(j) = 1.
    p_array(i)%y(j) = 2.
    p_array(i)%z(j) = 3.
  enddo
enddo

end subroutine kernel_derived_type
