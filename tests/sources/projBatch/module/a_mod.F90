module a_mod
    implicit none
contains
    subroutine a(arg)
        use header_mod, only: k
        real(kind=k), intent(inout) :: arg(:)
    end subroutine a
end module a_mod
