module tt_mod
    use header_mod, only: k
    implicit none

    integer, parameter :: nclv = 5

    type tt
        real(kind=k), allocatable :: indirection(:)
        real(kind=k) :: other(nclv)
    contains
        procedure :: proc => tt_proc
    end type tt
contains
    subroutine tt_proc(this)
        class(tt), intent(inout) :: this
    end subroutine tt_proc
end module tt_mod
