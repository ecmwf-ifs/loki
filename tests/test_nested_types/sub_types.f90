module sub_types
    implicit none

    public sub_type

    type sub_type
    !$loki dimension(size)
    integer, pointer :: x(:)
    end type sub_type

end module sub_types