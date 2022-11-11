module typebound_item
    use typebound_header
    implicit none

    type some_type
    contains
        procedure, nopass :: routine => module_routine
        procedure :: some_routine
        procedure, pass :: other_routine
        procedure :: routine1, &
            & routine2 => routine
        ! procedure :: routine1
        ! procedure :: routine2 => routine
    end type some_type
contains
    subroutine module_routine
        integer m
        m = 2
    end subroutine module_routine

    subroutine some_routine(self)
        class(some_type) :: self

        call self%routine
    end subroutine some_routine

    subroutine other_routine(self, m)
        class(some_type), intent(inout) :: self
        integer, intent(in) :: m
        integer :: j

        if (m < 0) call abor1('Error with unbalanced parenthesis)')

        j = m
        call self%routine1
        call self%routine2
    end subroutine other_routine

    subroutine routine(self)
        class(some_type) :: self
        call self%some_routine
    end subroutine routine

    subroutine routine1(self)
        class(some_type) :: self
        call module_routine
    end subroutine routine1
end module typebound_item

subroutine driver
    use typebound_item
    use typebound_header
    use typebound_other, only: other => other_type
    implicit none

    type(some_type), allocatable :: obj(:), obj2(:,:)
    type(header_type) :: header
    type(other) :: other_obj, derived(2)
    real :: x
    integer :: i

    allocate(obj(1))
    allocate(obj2(1,1))
    call obj(1)%other_routine(5)
    call obj2(1,1)%some_routine
    call header%member_routine(1)
    call header%routine(x)
    call header%routine(i)
    call other_obj%member(2, 2)
    call derived(1)% var( 2 ) % member_routine(2)
end subroutine driver
