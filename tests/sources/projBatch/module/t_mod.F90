module t_mod
    use tt_mod, only: tt
    use a_mod, only: a
    implicit none

    integer, parameter :: nt1 = 10

    type t1
    contains
        procedure :: way => my_way
    end type t1

    type t
        type(tt) :: yay
        type(t1) :: no(nt1)
    contains
        procedure :: proc => t_proc
    end type t
contains
    subroutine t_proc(this)
        class(t), intent(inout) :: this
        call a(this%yay%other)
        call this%yay%proc()
    end subroutine t_proc

    recursive subroutine my_way(this, recurse)
        class(t1), intent(inout) :: this
        logical, intent(in) :: recurse
        if (recurse) call this%way(.false.)
    end subroutine my_way
end module t_mod
