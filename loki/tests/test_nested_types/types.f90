module types
    use sub_types, only: sub_type

    implicit none

    public parent_type
    
    type parent_type
        integer :: member
        type(sub_type) :: type_member
    end type parent_type

end module