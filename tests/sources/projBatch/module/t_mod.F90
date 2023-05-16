module t_mod
    use tt_mod, only: tt
    implicit none

    type t
        type(tt) :: yay
    end type t
end module t_mod
