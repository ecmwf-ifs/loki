subroutine SimpleDependency(data, n)
  implicit none
  integer, intent(in) :: n
  real(8), dimension(n) :: data
  integer :: i

  ! Loop with a simple loop-carried dependency
  do i = 1, n
    data(i) = data(i) + data(i-1)
  end do

end subroutine SimpleDependency


subroutine NestedDependency(data, n)
  implicit none
  integer, intent(in) :: n
  real(8), dimension(n) :: data
  integer :: i, j

  ! Nested loop with a loop-carried dependency
  do i = 2, n
    do j = 1, i-1
      data(i) = data(i) + data(j)
    end do
  end do

end subroutine NestedDependency


subroutine ConditionalDependency(data, n)
  implicit none
  integer, intent(in) :: n
  real(8), dimension(n) :: data
  integer :: i

  ! Loop with a conditional loop-carried dependency
  do i = 2, n
    if (data(i-1) > 0.0) then
      data(i) = data(i) + data(i-1)
    endif
  end do

end subroutine ConditionalDependency

subroutine NoDependency(data)
  implicit none
  real(8), dimension(20) :: data
  integer :: i

  do i = 1, 10, 1
    data(2*i) = 10;
  end do

  do i = 1, 5, 1
    data(2*i + 1) = 20;
  end do
end subroutine NoDependency