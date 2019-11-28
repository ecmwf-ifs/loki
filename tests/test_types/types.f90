module types

  integer, parameter :: jprb = selected_real_kind(13,300)

  type pragma_type
    !$loki dimension(3,3)
    real(kind=jprb), dimension(:,:), pointer :: matrix
    !$loki dimension(klon,klat,2)
    real(kind=jprb), pointer :: tensor(:, :, :)
  end type pragma_type

contains

  subroutine alloc_pragma_type(item)
    type(pragma_type), intent(inout) :: item
    allocate(item%matrix(5,5))
    allocate(item%tensor(3,4,5))
  end subroutine

  subroutine free_pragma_type(item)
    type(pragma_type), intent(inout) :: item
    deallocate(item%matrix)
    deallocate(item%tensor)
  end subroutine

end module types
