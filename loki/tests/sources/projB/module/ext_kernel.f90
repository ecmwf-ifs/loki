module ext_kernel_mod
  integer, parameter :: jprb = 4

contains

  subroutine ext_kernel(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:, :)
    integer :: i

    do i = 1, size(vector)
      matrix(:, i) = matrix(:, i) + vector(i)
    end do
  end subroutine ext_kernel

end module ext_kernel_mod
