interface

subroutine another_l2(matrix)
  use header_mod, only: jprb
  implicit none

  real(kind=jprb), intent(inout) :: matrix(:,:)
end subroutine another_l2

end interface
