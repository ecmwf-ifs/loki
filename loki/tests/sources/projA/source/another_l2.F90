subroutine another_l2(matrix)
  use header_mod, only: jprb

  implicit none

  real(kind=jprb), intent(inout) :: matrix(:,:)

  matrix(:,:) = 77.0

end subroutine another_l2
