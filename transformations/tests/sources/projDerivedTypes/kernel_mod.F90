module kernel_mod

use some_derived_type_mod, only: some_derived_type
implicit none

contains

  subroutine kernel(m, n, P_a, P_b, Q, R)
        integer                , intent(in)    :: m, n
        real, intent(inout)                    :: P_a, P_b
        type(some_derived_type), intent(in)    :: Q
        type(some_derived_type), intent(out)   :: R
        integer :: j, k

        R%a = P_a + Q%a
        R%b = P_b - Q%b
  end subroutine kernel

end module kernel_mod
