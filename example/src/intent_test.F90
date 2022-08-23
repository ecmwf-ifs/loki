
module kernel_mod
  use parkind1, only : jpim,jprb
  implicit none
contains
  subroutine some_kernel(n,vout,var_out,var_in,var_inout,b,l,h,y)
  
  integer(kind=jpim),intent(in)  :: n,l,b
  integer(kind=jpim),intent(in)  :: h
  real(kind=jprb),   intent(in )   ::   var_in   (n)
  real(kind=jprb),   intent(inout) ::   var_out  (n)
  real(kind=jprb),   intent(inout) ::   var_inout(n)
  real(kind=jprb),   intent(inout) ::   vout(n)
  real(kind=jprb),   intent(inout) ::      y(:)
  
  end subroutine some_kernel
end module kernel_mod

subroutine intent_test(m,n,var_in,var_out,var_inout,tendency_loc)
use parkind1, only : jpim,jprb
use kernel_mod, only: some_kernel
use yoecldp, only : nclv
implicit none

integer(kind=jpim),intent(in) :: m,n
integer(kind=jpim) :: i,j,k,h,l
real(kind=jprb),   intent(in )        :: var_in   (n,n,n)
real(kind=jprb),   target,intent(out) :: var_out  (n,n,n)
real(kind=jprb),   intent(inout)      :: var_inout(n,n,n)
real(kind=jprb), allocatable :: x(:),y(:)
real(kind=jprb), pointer :: vout(n)
type(state_type), intent (out) :: tendency_loc

allocate(x(n))
associate(mtmp=>m)
allocate(y(mtmp))
end associate

associate(mtmp=>n)
do k=1,mtmp
  do j=1,mtmp
    do i=1,mtmp  
      var_out(i,j,k) = 2._jprb
    enddo
 
    associate(mbuf=>mtmp) 
    var_out(m:mbuf,j,k) = var_in(m:mbuf,j,k)+var_inout(m:mbuf,j,k)+var_out(m:mbuf,j,k)
    end associate

    vout=>var_out(:,j,k)

    associate(vin=>mtmp)
    call some_kernel(vin,vout,vout,var_in(:,j,k),var_inout(:,j,k),1,h=vin,l=5,y=y)
    end associate

    nullify(vout)

    associate(vout=>tendency_loc%cld(:,j,k))

    associate(vin=>var_in(:,j,k))
    call some_kernel(mtmp,vout,var_out(:,j,k),vin,var_inout(:,j,k),1,h=mtmp,l=5,y=y)
    end associate

    end associate

    do i=1,mtmp
      var_inout(i,j,k) = var_out(i,j,k)
    enddo
  enddo
enddo
end associate

deallocate(x)
deallocate(y)

end subroutine intent_test
