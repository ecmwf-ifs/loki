module tt_mod
    use header_mod, only: k
    implicit none

    type tt
        real(kind=k), allocatable :: indirection(:)
    end type tt
end module tt_mod
