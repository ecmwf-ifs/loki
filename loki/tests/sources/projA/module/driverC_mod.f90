module driverC_mod
  use header_mod, only: jprb, header_type
  use kernelC_mod, only: kernelC

  implicit none

contains

  subroutine driverC()
    type(header_type) :: mystruct

    call kernelC(mystruct%vector, mystruct%matrix)

  end subroutine driverC

end module driverC_mod
