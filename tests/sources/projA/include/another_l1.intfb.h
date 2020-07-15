interface

subroutine another_l1(arrayB)
  use header_mod, only: jprb
  implicit none

  real(kind=jprb), intent(inout) :: arrayB(:)
end subroutine another_l1

end interface
