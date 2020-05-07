module types
    implicit none

    public parent_type

    type sub_type
        !$loki dimension(size)
        integer, pointer :: x(:)
    end type sub_type

    type parent_type
        integer :: member
        type(sub_type) :: type_member
    end type parent_type

end module
