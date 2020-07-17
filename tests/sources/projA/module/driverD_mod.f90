module driverD_mod
  use header_mod, only: jprb, header_type
  use kernelD_mod, only: kernelD

  implicit none

contains

  subroutine driverD()
    type(header_type) :: mystruct

    call kernelD(mystruct%vector, mystruct%matrix)
  end subroutine driverD

end module driverD_mod
