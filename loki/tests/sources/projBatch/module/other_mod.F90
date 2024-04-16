module other_mod
    use tt_mod, only: tt
    use b_mod, only: b
    implicit none
contains
    subroutine mod_proc(arg)
        type(tt), intent(inout) :: arg
        call arg%proc()
        call b(arg%indirection)
    end subroutine mod_proc
end module other_mod
