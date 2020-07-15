module compute_l1_mod
  use header_mod, only: jprb
  use compute_l2_mod, only: compute_l2

contains

  subroutine compute_l1(arrayA)
    real(kind=jprb), intent(inout) :: arrayA(:)

    call compute_l2(arrayA)

  end subroutine compute_l1

end module compute_l1_mod
