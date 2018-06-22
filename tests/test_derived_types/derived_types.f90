MODULE derived_types

  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)

  TYPE structure
     REAL(KIND=JPRB) :: scalar, vector(3), matrix(3, 3)
     REAL(KIND=JPRB) :: red_herring
  END TYPE structure

CONTAINS

  SUBROUTINE simple_loops(item)
    ! Simple vector/matrix arithmetic with a derived type
    TYPE(structure), intent(inout) :: item
    integer :: i, j, n

    n = 3
    do i=1, n
       item%vector(i) = item%vector(i) + item%scalar
    end do

    do j=1, n
       do i=1, n
          item%matrix(i, j) = item%matrix(i, j) + item%scalar
       end do
    end do

  END SUBROUTINE simple_loops

END MODULE derived_types
