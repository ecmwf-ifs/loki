interface

subroutine another_l1(matrix)
  use header_mod, only: jprb
  implicit none

  real(kind=jprb), intent(inout) :: matrix(:,:)
end subroutine another_l1

end interface
