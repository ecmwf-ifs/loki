module compute_l2_mod
  use header_mod, only: jprb

contains

  subroutine compute_l2(vector)
    real(kind=jprb), intent(inout) :: vector(:)

    vector(:) = 66.0

  end subroutine compute_l2

end module compute_l2_mod
