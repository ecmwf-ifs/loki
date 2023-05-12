subroutine comp2 (arg, val)
    use t_mod, only: t
    use header_mod, only: k
    use a_mod, only: a
    use b_mod, only: b
    implicit none
    type(t), intent(inout) :: arg
    real(kind=k), intent(inout) :: val(:)

    call a(t%yay%indirection)
    call b(val)
    call arg%yay%proc()
end subroutine comp2
