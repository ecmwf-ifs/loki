module driver_mod

use parkind1, only: JRPB
use kernel_mod, only: kernel_colsum

contains
  subroutine driver_colsum(a, b, n)
    integer, intent(in) :: n
    real(kind=JPRB), intent(inout) :: a(n)
    real(kind=JPRB), intent(inout) :: b(n,n)
    integer :: j
    !$loki driver-loop
    do j=1,n
      call kernel_colsum(a(j), b(:, j), n)
    end do
  end subroutine driver_colsum

end module driver_mod
