module b_mod
    implicit none
contains
    subroutine b(arg)
        use header_mod, only: k
        real(kind=k), intent(inout) :: arg(:)
    end subroutine b
end module b_mod
