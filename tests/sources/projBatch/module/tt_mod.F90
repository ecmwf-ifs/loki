module tt_mod
    use header_mod, only: k
    implicit none

    integer, parameter :: nclv = 5

    type tt
        real(kind=k), allocatable :: indirection(:)
        real(kind=k) :: other(nclv)
    contains
        procedure :: proc
    end type tt
contains
    subroutine proc(this)
        class(tt), intent(inout) :: this
    end subroutine proc
end module tt_mod
