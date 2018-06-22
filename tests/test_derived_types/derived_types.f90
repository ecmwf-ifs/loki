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

  SUBROUTINE array_indexing(item)
    ! Simple vector/matrix arithmetic with a derived type
    TYPE(structure), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.

    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do

  END SUBROUTINE array_indexing

END MODULE derived_types
