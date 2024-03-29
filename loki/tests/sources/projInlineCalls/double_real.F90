real function double_real(i)
  use vars_module, only: varA, varB
  implicit none
  integer, intent(in) :: i

  double_real =  dble(i*2)
end function double_real
