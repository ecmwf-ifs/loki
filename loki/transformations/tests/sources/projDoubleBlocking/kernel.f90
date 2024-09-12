module kernel_mod

USE PARKIND1, ONLY: JPRB

implicit none
contains
  ! this routine computes the sum of column j in b and stores the result in a
  subroutine kernel_colsum(a, b, n)
    integer, intent(in) :: n              ! size of a and b
    real(kind=JPRB), intent(out) :: a
    real(kind=JPRB), intent(in) :: b(n,n)
    a = 0
    do i=1,n
      a = a + b(i,j)
    end do
  end subroutine kernel_colsum
end module kernel_mod
