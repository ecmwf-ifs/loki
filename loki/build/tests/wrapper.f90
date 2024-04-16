module wrapper
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)

contains

  subroutine mult_add_external(a, b, c, sum)
    implicit none

    interface
       subroutine mult_add_fc(a, b, c, sum) &
            & bind(c, name='mult_add_c')
         use iso_c_binding, only: c_double
         implicit none

         ! Pass values in by value, out by reference
         real(kind=c_double), value :: a, b, c
         real(kind=c_double) :: sum
       end subroutine mult_add_fc
    end interface

    real(kind=jprb), intent(in) :: a, b, c
    real(kind=jprb), intent(out) :: sum

    call mult_add_fc(a, b, c, sum)
  end subroutine mult_add_external

end module wrapper
