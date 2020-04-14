subroutine routine_nesting(a, b, c, d, e)
integer, intent(in) :: a, b, c, d, e

if (a > 3) then
    if (b > 2) then
        if (c > 1) then
            print *, 'if-if-if'
        end if
    end if
    select case (d)
        case (0)
            if (e == 0) then
                print *, 'if-case-if'
            endif
        case (1:3)
            if (e == 0) then
                print *, 'if-range-if'
            else
                print *, 'if-range-else'
            endif
        case default
            if (e == 0) then
                print *, 'if-default-if'
            endif
    end select
elseif (a == 3) then
    if (b > 2) then
        if (c > 1) then
            print *, 'elseif-if-if'
        end if
    end if
else
    if (e == 0) print *, 'else-inlineif'
    if (b > 2) then
        if (c > 1) then
            print *, 'else-if-if'
        end if
    end if
end if
end subroutine routine_nesting
