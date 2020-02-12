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

  type case_sensitive
    real(kind=jprb) :: u, v, T
    real(kind=jprb) :: q, A
  end type case_sensitive

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

  subroutine deferred_array(item)
    ! simple vector/matrix with an array of derived types
    type(deferred), intent(inout) :: item
    type(deferred), allocatable :: item2(:)
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i, j

    allocate(item2(4))

    do j=1, 4
      call alloc_deferred(item2(j))

      item2(j)%vector(:) = 666.

      do i=1, 3
        item2(j)%matrix(:, i) = vals(i)
      end do
    end do

    item%vector(:) = 0.
    item%matrix(:,:) = 0.

    do j=1, 4
      item%vector(:) = item%vector(:) + item2(j)%vector(:)

      do i=1, 3
          item%matrix(:,i) = item%matrix(:,i) + item2(j)%matrix(:,i)
      end do

      call free_deferred(item2(j))
    end do

    deallocate(item2)
  end subroutine deferred_array

  subroutine derived_type_caller(item)
    ! simple call to another routine specifying a derived type as argument
    type(explicit), intent(inout) :: item

    item%red_herring = 42.
    call simple_loops(item)

  end subroutine derived_type_caller

  subroutine associates(item)
    ! Use associate to access and modify other items
    type(explicit), intent(inout) :: item
    type(deferred) :: item2

    item%scalar = 17.0

    associate(vector2=>item%matrix(:,1))

        vector2(:) = 3.
        item%matrix(:,3) = vector2(:)

    end associate

    associate(vector=>item%vector)

        item%vector(2) = vector(1)
        vector(3) = item%vector(1) + vector(2)
        vector(1) = 1.

    end associate

    call alloc_deferred(item2)

    associate(vec=>item2%vector(2))

        vec = 1.

    end associate

    call free_deferred(item2)

  end subroutine associates

  subroutine check_case(item)
    type(case_sensitive), intent(inout) :: item

    item%u = 1.0
    item%v = 2.0
    item%t = 3.0
    item%q = -1.0
    item%A = -5.0
  end subroutine check_case

end module derived_types
