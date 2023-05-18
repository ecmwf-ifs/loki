module t_mod
    use tt_mod, only: tt
    implicit none

    type t1
    contains
        procedure :: way => my_way
    end type t1

    type t
        type(tt) :: yay
        type(t1) :: no
    contains
        procedure :: proc => t_proc
    end type t
contains
    subroutine t_proc(this)
        class(t), intent(inout) :: this
    end subroutine t_proc

    subroutine my_way(this)
        class(t1), intent(inout) :: this
    end subroutine my_way
end module t_mod
