module driverA_mod
  use header_mod, only: jprb, header_type
  use kernelA_mod, only: kernelA

  implicit none

contains

  subroutine driverA()
    type(header_type) :: mystruct

    call kernelA(mystruct%vector, mystruct%matrix)

  end subroutine driverA

end module driverA_mod
