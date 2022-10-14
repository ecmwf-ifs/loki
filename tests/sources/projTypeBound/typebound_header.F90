module typebound_header
    implicit none

    type header_type
    contains
        procedure :: member_routine => header_member_routine
    end type header_type

    type other_type
      type(header_type) :: var(2)
    end type other_type

contains

    subroutine header_member_routine(self, val)
        class(header_type) :: self
        integer, intent(in) :: val
        integer :: j
        j = val
    end subroutine header_member_routine

    SUBROUTINE ABOR1(CDTEXT)
        CHARACTER(LEN=*) CDTEXT
        WRITE(0,*) CDTEXT
        call abort()
    END SUBROUTINE ABOR1
end module typebound_header
