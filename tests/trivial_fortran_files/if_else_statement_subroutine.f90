subroutine check_number(x)
  real, intent(in) :: x

  if (x > 0.0) then
    print *, "The number is positive."
  else
    print *, "The number is non-positive."
  end if

end subroutine check_number