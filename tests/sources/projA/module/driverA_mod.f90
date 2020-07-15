module driverA_mod
  use header_mod, only: jprb
  use kernelA_mod, only: kernelA

  implicit none

contains

  subroutine driverA()
    real(kind=jprb) :: arrayA(5)
    real(kind=jprb) :: arrayB(3)

    call kernelA(arrayA, arrayB)

  end subroutine driverA

end module driverA_mod
