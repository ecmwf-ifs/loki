#ifdef HAVE_EXT_DRIVER_MODULE
module ext_driver_mod
  implicit none

contains
#endif

  subroutine ext_driver(vector, matrix)
    use ext_kernel_mod, only: jprb, ext_kernel
    implicit none
    real(kind=jprb), intent(inout) :: vector(:)
    real(kind=jprb), intent(inout) :: matrix(:, :)

    call ext_kernel(vector, matrix)
  end subroutine ext_driver

#ifdef HAVE_EXT_DRIVER_MODULE
end module ext_driver_mod
#endif
