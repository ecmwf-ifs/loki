subroutine nested_if_example(x, y)
  integer, intent(in) :: x, y

  if (x > 0) then
    if (y > 0) then
      print *, "Both x and y are positive."
    else
      print *, "x is positive, but y is not."
    end if
  else
    if (y > 0) then
      print *, "x is not positive, but y is positive."
    else
      print *, "Both x and y are not positive."
    end if
  end if

end subroutine nested_if_example
