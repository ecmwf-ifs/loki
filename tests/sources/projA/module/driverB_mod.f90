module driverB_mod
  use header_mod, only: jprb, header_type
  use kernelB_mod, only: kernelB

  implicit none

contains

  subroutine driverB()
    type(header_type) :: mystruct

    call kernelB(mystruct%vector, mystruct%matrix)

  end subroutine driverB

end module driverB_mod
