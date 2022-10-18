module typebound_other
    use typebound_header, header => header_type

    implicit none

    type other_type
      type(header) :: var(2)
    contains
      procedure :: member => other_member
    end type other_type

contains

    module subroutine other_member(self, i, m)
        use typebound_header, only: member_routine => header_member_routine
        class(other_type) :: self
        integer, intent(in) :: i, m
        call member_routine(m)
        call self%var(i)%member_routine(m)
    end subroutine other_member

end module typebound_other
