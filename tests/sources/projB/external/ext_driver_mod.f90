module ext_driver_mod
  use ext_kernel_mod, only: jprb, ext_kernel

contains

  subroutine ext_driver(vector, matrix)
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:, :)

    call ext_kernel(vector, matrix)
  end subroutine ext_driver

end module ext_driver_mod
