subroutine routine_a
integer a
a = 1
end subroutine routine_a

module some_module
contains
subroutine module_routine
integer m
m = 2
end subroutine module_routine
function module_function(n)
integer n
n = 3
end function module_function
end module some_module

subroutine routine_b
integer b
b = 4
contains
subroutine contained_c
integer c
c = 5
end subroutine contained_c
end subroutine routine_b

function function_d(d)
integer d
d = 6
end function function_d
