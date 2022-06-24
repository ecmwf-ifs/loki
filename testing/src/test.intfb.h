
interface
subroutine test(klon,kidia,kfdia,klev,var_in,var_out)
use parkind1, only : jpim,jprb

integer(kind=jpim),intent(in) :: klon,kidia,kfdia
integer(kind=jpim),intent(in) :: klev

real(kind=jprb),   intent(in ) :: var_in (klon,klev)
real(kind=jprb),   intent(out) :: var_out(klon,klev)

end subroutine test
end interface
