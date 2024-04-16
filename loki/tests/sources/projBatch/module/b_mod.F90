module b_mod
    use header_mod, only: k
    implicit none
contains
    subroutine b(arg)
        real(kind=k), intent(inout) :: arg(:)
    end subroutine b
end module b_mod
