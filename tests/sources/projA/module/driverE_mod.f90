module driverE_mod
  use header_mod, only: jprb, header_type
  use kernelE_mod, only: kernelE, kernelET

  implicit none

contains

  ! Two driver routines, but we process only one!

  subroutine driverE_single()
    type(header_type) :: mystruct

    call kernelE(mystruct%vector, mystruct%matrix)

  end subroutine driverE_single


  subroutine driverE_multiple()
    type(header_type) :: mystruct

    call kernelE(mystruct%vector, mystruct%matrix)

    call kernelET(mystruct%vector, mystruct%matrix)
  end subroutine driverE_multiple

end module driverE_mod
