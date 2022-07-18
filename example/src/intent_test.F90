
module kernel_mod
  use parkind1, only : jpim,jprb

  implicit none

contains

  subroutine some_kernel(n,var_out,var_in,var_inout,x,l,h)
  
  !-------------
  !    arguments
  !-------------
  
  integer(kind=jpim),intent(in)  :: n,h,l
  
  real(kind=jprb),   intent(in )   ::   var_in   (n)
  real(kind=jprb),   intent(inout) ::   var_out  (n)
  real(kind=jprb),   intent(in )   ::   var_inout(n)

  real(kind=jprb),   intent(in )   ::   x(n)
  
  !-------------------
  !    local variables
  !-------------------
  
  
  end subroutine some_kernel
end module kernel_mod

subroutine intent_test(m,n,var_in,var_out,var_inout)
use parkind1, only : jpim,jprb
use kernel_mod, only: some_kernel

implicit none

!-------------
!    arguments
!-------------

integer(kind=jpim),intent(in) :: m,n

real(kind=jprb),   intent(in ) ::   var_in   (n,n,n)
real(kind=jprb),   intent(out) ::   var_out  (n,n,n)
real(kind=jprb),   intent(inout) :: var_inout(n,n,n)

!-------------------
!    local variables
!-------------------

integer(kind=jpim) :: i,j,k,h,l

real(kind=jprb), allocatable :: x(:),y(:)

associate(vout=>var_out(n,m,m))

allocate(x(n))
allocate(y(n))

do k=1,n
  do j=1,n

    do i=1,n   
      var_out(i,j,k) = 2._jprb
    enddo

    var_out(m:n,j,k) = var_in(m:n,j,k)+var_inout(m:n,j,k)+var_out(m:n,j,k)

    call some_kernel(n,var_out(:,j,k),var_in(:,j,k),var_inout(:,j,k),x,h=10,l=5)

    do i=1,n   
      var_inout(i,j,k) = var_out(i,j,k)
    enddo

  enddo
enddo

deallocate(x)
deallocate(y)

end associate

end subroutine intent_test
