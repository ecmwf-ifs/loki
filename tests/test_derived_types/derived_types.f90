MODULE derived_types

  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)

  TYPE explicit
     REAL(KIND=JPRB) :: scalar, vector(3), matrix(3, 3)
     REAL(KIND=JPRB) :: red_herring
  END TYPE explicit

  TYPE deferred
     REAL(KIND=JPRB), allocatable :: scalar, vector(:), matrix(:, :)
     REAL(KIND=JPRB), allocatable :: red_herring
  END TYPE deferred

CONTAINS

  SUBROUTINE alloc_deferred(item)
    TYPE(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  END SUBROUTINE alloc_deferred

  SUBROUTINE free_deferred(item)
    TYPE(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  END SUBROUTINE free_deferred

  SUBROUTINE simple_loops(item)
    ! Simple vector/matrix arithmetic with a derived type
    TYPE(explicit), intent(inout) :: item
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

  SUBROUTINE array_indexing_explicit(item)
    ! Simple vector/matrix arithmetic with a derived type
    TYPE(explicit), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.

    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do

  END SUBROUTINE array_indexing_explicit

  SUBROUTINE array_indexing_deferred(item)
    ! Simple vector/matrix arithmetic with a derived type
    TYPE(deferred), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.

    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do

  END SUBROUTINE array_indexing_deferred

END MODULE derived_types
