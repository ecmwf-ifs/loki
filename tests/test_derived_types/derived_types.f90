module derived_types

  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
     real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
     real(kind=jprb) :: red_herring
  end type explicit

  type deferred
     real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
     real(kind=jprb), allocatable :: red_herring
  end type deferred

  type nested
     real(kind=jprb) :: a_scalar, a_vector(3)
     type(explicit) :: another_item
  end type nested

contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine simple_loops(item)
    ! simple vector/matrix arithmetic with a derived type
    type(explicit), intent(inout) :: item
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

  end subroutine simple_loops

  subroutine array_indexing_explicit(item)
    ! simple vector/matrix arithmetic with a derived type
    type(explicit), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.

    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do

  end subroutine array_indexing_explicit

  subroutine array_indexing_deferred(item)
    ! simple vector/matrix arithmetic with a derived type
    type(deferred), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.

    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do

  end subroutine array_indexing_deferred

  subroutine array_indexing_nested(item)
    ! simple vector/matrix arithmetic with a nested derived type
    type(nested), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%a_vector(:) = 666.
    item%another_item%vector(:) = 999.

    do i=1, 3
       item%another_item%matrix(:, i) = vals(i)
    end do

  end subroutine array_indexing_nested


end module derived_types
