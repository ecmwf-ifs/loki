subroutine loop_fuse(n,var_in,var_out)
use parkind1, only : jpim,jprb
implicit none

integer(kind=jpim),intent(in) :: n
real(kind=jprb),   intent(in ) :: var_in (n,n,n)
real(kind=jprb),   intent(out) :: var_out(n,n,n)
integer(kind=jpim) :: i,j,k

do k=1,n
  do j=1,n
    do i=1,n
      var_out(i,j,k) = var_in(i,j,k)
    enddo
    do i=1,n   
      var_out(i,j,k) = 2._JPRB*var_out(i,j,k)
    enddo
  enddo
enddo

end subroutine loop_fuse

subroutine loop_fuse_v1(n,var_in,var_out)
use parkind1, only : jpim,jprb
implicit none

integer(kind=jpim),intent(in) :: n
real(kind=jprb),   intent(in ) :: var_in (n,n,n)
real(kind=jprb),   intent(out) :: var_out(n,n,n)
integer(kind=jpim) :: i,j,k

do k=1,n
  do j=1,n
    do i=1,n
      var_out(i,j,k) = var_in(i,j,k)
    enddo
    do i=1,n   
      var_out(i,j,k) = 2._JPRB*var_out(i,j,k)
    enddo
  enddo

  call some_kernel(n,var_out(1,1,k))

  do j=1,n
    do i=1,n
      var_out(i,j,k) = var_out(i,j,k) + 1._JPRB
    enddo
    do i=1,n   
      var_out(i,j,k) = 2._JPRB*var_out(i,j,k)
    enddo
  enddo
enddo

end subroutine loop_fuse_v1

subroutine loop_fuse_v2(n,var_in,var_out)
use parkind1, only : jpim,jprb
implicit none

integer(kind=jpim),intent(in) :: n
real(kind=jprb),   intent(in ) :: var_in (n,n,n)
real(kind=jprb),   intent(out) :: var_out(n,n,n)
integer(kind=jpim) :: i,j,k

do k=1,n
  do j=1,n
    do i=1,n
      var_out(i,j,k) = var_in(i,j,k)
    enddo
    do i=1,n   
      var_out(i,j,k) = 2._JPRB*var_out(i,j,k)
    enddo

    call some_kernel(n,var_out(1,j,k))

    do i=1,n
      var_out(i,j,k) = var_out(i,j,k) + 1._JPRB
    enddo
    do i=1,n   
      var_out(i,j,k) = 2._JPRB*var_out(i,j,k)
    enddo
  enddo
enddo

end subroutine loop_fuse_v2

subroutine loop_fuse_pragma(n,var_in,var_out)
use parkind1, only : jpim,jprb
implicit none

integer(kind=jpim),intent(in) :: n
real(kind=jprb),   intent(in ) :: var_in (n,n,n)
real(kind=jprb),   intent(out) :: var_out(n,n,n)
integer(kind=jpim) :: i,j,k

do k=1,n
  do j=1,n

    !$loki loop-fusion group(g1)
    do i=1,n
      var_out(i,j,k) = var_in(i,j,k)
    enddo
    !$loki loop-fusion group(g1)
    do i=1,n   
      var_out(i,j,k) = 2._JPRB*var_out(i,j,k)
    enddo

    call some_kernel(n,var_out(1,j,k))

    !$loki loop-fusion group(g2)
    do i=1,n
      var_out(i,j,k) = var_out(i,j,k) + 1._JPRB
    enddo
    !$loki loop-fusion group(g2)
    do i=1,n   
      var_out(i,j,k) = 2._JPRB*var_out(i,j,k)
    enddo
    
  enddo
enddo

end subroutine loop_fuse_pragma
