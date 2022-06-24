subroutine test(klon,kidia,kfdia,klev,var_in,var_out)
use parkind1, only : jpim,jprb

implicit none

!-------------
!    arguments
!-------------

integer(kind=jpim),intent(in) :: klon,kidia,kfdia
integer(kind=jpim),intent(in) :: klev

real(kind=jprb),   intent(in ) :: var_in (klon,klev)
real(kind=jprb),   intent(out) :: var_out(klon,klev)


!-------------------
!    local variables
!-------------------

integer(kind=jpim),intent(in) :: jk,jl,jh,jj



do jk=1,klev
  do jh=1,10
    write(*,*) 10
  enddo
  do jl=kidia,kfdia
    var_out(jl,jk) = var_in(jl,jk)
    do jj=1,10
      write(*,*) jj
    enddo
  enddo
enddo

do jk=1,klev
  do jl=kidia,kfdia
    var_out(jl,jk) = 2._JPRB*var_out(jl,jk)
  enddo
enddo

do jk=1,klev
  do jl=kidia,kfdia
    var_out(jl,jk) = var_out(jl,jk)-var_in(jl,jk)
  enddo
enddo


end subroutine test
