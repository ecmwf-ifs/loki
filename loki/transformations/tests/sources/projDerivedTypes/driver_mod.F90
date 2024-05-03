module driver_mod

use some_derived_type_mod, only: some_derived_type
use kernel_mod, only: kernel
implicit none

contains
  subroutine driver(z)
        integer, intent(in) :: z
        type(some_derived_type) :: t_io
        type(some_derived_type) :: t_in, t_out
        integer :: m, n
        integer :: i, j

        m = 100
        n = 10

        t_in%a = real(m-1)
        t_in%b = real(n-1)

        call kernel(m, n, t_io%a, t_io%b, t_in, t_out)

  end subroutine driver
end module driver_mod
