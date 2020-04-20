subroutine routine_limit_statements()
integer :: a, b, c, d, e

! Non-exec statements
#define some_macro
print *, 'Hello world!'

associate (aa=>a)
    aa = 1
    b = 2
    call some_routine(c, e)
    d = 4
end associate

end subroutine routine_limit_statements
