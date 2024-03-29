subroutine routine_a
  integer a, i
  a = 1
  i = a + 1

  call routine_b(a, i)
end subroutine routine_a

module some_module
contains
  subroutine module_routine
    integer m
    m = 2

    call routine_b(m, 6)
  end subroutine module_routine

  function module_function(n)
    integer n
    n = 3
  end function module_function
end module some_module

subroutine routine_b(i,j)
  integer, intent(in) :: i, j
  integer b
  b = 4

  call contained_c(i)

  call routine_a()
contains

  subroutine contained_c(i)
    integer, intent(in) :: i
    integer c
    c = 5
  end subroutine contained_c

  subroutine contained_d(i)
    integer, intent(in) :: i
    integer c
    c = 8
  end subroutine contained_d
end subroutine routine_b

function function_d(d)
integer d
d = 6
end function function_d
