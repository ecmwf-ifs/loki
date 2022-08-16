module proj_c_util_mod
  integer, parameter :: jprb = 4

contains

  subroutine routine_one(matrix)
    real(kind=jprb), intent(inout) :: matrix(:,:)
    integer :: i

    do i = 1, size(matrix, dim=1)
      call routine_two(matrix(i,:))
    end do
  end subroutine routine_one

  subroutine routine_two(vector)
    real(kind=jprb), intent(inout) :: vector(:)

    vector(:) = vector(:) + 2.0
  end subroutine routine_two

end module proj_c_util_mod
