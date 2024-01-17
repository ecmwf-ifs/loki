module typebound_other
    use typebound_header, only: header => header_type

    implicit none

    type other_type
      type(header) :: var(2)
    contains
      procedure :: member => other_member
    end type other_type

    type outer_type
      type(other_type) :: other
    contains
      procedure :: nested_call
    end type outer_type

contains

    module subroutine other_member(self, i, m)
        use typebound_header, only: member_routine => header_member_routine
        class(other_type) :: self
        integer, intent(in) :: i, m
        call member_routine(m)
        call self%var(i)%member_routine(m)
    end subroutine other_member

    subroutine nested_call(self, m)
      class(outer_type) :: self
      integer, intent(in) :: m
      call self%other%var(1)%member_routine(m)
      call self%other%var(2)%member_routine(m)
    end subroutine nested_call

end module typebound_other
