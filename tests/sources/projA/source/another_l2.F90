subroutine another_l2(arrayB)
  use header_mod, only: jprb

  implicit none

  real(kind=jprb), intent(inout) :: arrayB(:)

  arrayB(:) = 77.0

end subroutine another_l2
