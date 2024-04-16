interface
subroutine comp2 (arg, val)
    use t_mod, only: t
    use header_mod, only: k
    implicit none
    type(t), intent(inout) :: arg
    real(kind=k), intent(inout) :: val(:)
end subroutine comp2
end interface
