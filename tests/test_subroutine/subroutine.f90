
subroutine routine_simple (x, y, scalar, vector, matrix)
  ! A simple standard looking routine
  ! to test argument declarations.
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_simple


subroutine routine_simple_caching (x, y, scalar, vector, matrix)
  ! A simple standard looking routine to test variable caching.
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, parameter :: jpim = selected_int_kind(9)
  integer, intent(in) :: x, y
  ! The next two share names with `routine_simple`, but have different
  ! dimensions or types, so that we can test variable caching.
  integer(kind=jpim), intent(in) :: scalar
  integer(kind=jpim), intent(inout) :: vector(y), matrix(x, y)
  integer :: i

  do i=1, y
     vector(i) = vector(i) + scalar
     matrix(:, i) = i * vector(i)
  end do
end subroutine routine_simple_caching


subroutine routine_multiline_args &
 ! Test multiline dummy arguments with comments
 & (x, y, scalar, &
 ! Of course, not one...
 ! but two comment lines
 & vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_multiline_args


subroutine routine_local_variables (x, y, maximum)
  ! Test local variables and types
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(out) :: maximum

  integer :: i, j
  real(kind=jprb), dimension(x) :: vector
  real(kind=jprb) :: matrix(x, y)

  do i=1, x
     vector(i) = i * 10.
  end do
  do i=1, x
     do j=1, y
        matrix(i, j) = vector(i) + j * 2.
     end do
  end do
  maximum = matrix(x, y)

end subroutine routine_local_variables


subroutine routine_arguments (x, y, vector, matrix)
  ! Test internal argument handling
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), dimension(x), intent(inout) :: vector
  real(kind=jprb), intent(inout) :: matrix(x, y)

  integer :: i, j
  real(kind=jprb), dimension(x) :: local_vector
  real(kind=jprb) :: local_matrix(x, y)

  do i=1, x
     local_vector(i) = i * 10.
  end do
  do i=1, x
     do j=1, y
        local_matrix(i, j) = local_vector(i) + j * 2.
     end do
  end do

  vector(:) = local_vector(:)
  matrix(:, :) = local_matrix(:, :)

end subroutine routine_arguments


subroutine routine_dim_shapes(v1, v2, v3, v4, v5)
  ! Simple variable assignments with non-trivial sizes and indices
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: v1, v2
  real(kind=jprb), allocatable, intent(out) :: v3(:)
  real(kind=jprb), intent(out) :: v4(v1,v2), v5(1:v1,v2-1)

  allocate(v3(v1))
  v3(v1-v2+1) = 1.
  v4(3:v1,1:v2-3) = 2.
  v5(:,:) = 3.

end subroutine routine_dim_shapes


subroutine routine_typedefs_simple(item)
  ! simple vector/matrix arithmetic with a derived type
  ! imported from an external header module
  use header, only: derived_type
  implicit none

  type(derived_type), intent(inout) :: item
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

end subroutine routine_typedefs_simple


subroutine routine_call_callee(x, y, vector, matrix, another)
  ! Simple routine to be called from another routine
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y), another(x, y)

  vector(:) = 6.66
  matrix(:,:) = 9.99
  another(:,:) = 7.77

end subroutine routine_call_callee


subroutine routine_call_caller(x, y, vector, matrix, item)
  ! Simple routine calling another routine
  use header, only: derived_type
  implicit none

  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  type(derived_type), intent(inout) :: item

  ! To a parser, these arrays look like scalarst!
  call routine_call_callee(x, y, vector, matrix, item%matrix)

end subroutine routine_call_caller


subroutine routine_assign_string()
  ! Simple routine calling some unknown routine with a string argument
  ! There is nothing done with this routine but it tests the ability of
  ! frontends to handle string literals
  implicit none
  character(len=:), allocatable :: s
  
  s = 'Hello world!'

end subroutine routine_assign_string


subroutine routine_call_no_arg()
  implicit none

  call abort
end subroutine routine_call_no_arg


subroutine routine_pp_macros()
#define CONSTANT 123
#define FLAG
  implicit none
  integer :: y, z
#define SOME_MACRO(x) x + 1
  y = 1
#define SOME_OTHER_MACRO (x - 1)
#
#warning 'ABC'
#ifdef FLAG
  z = 3
#endif
end subroutine routine_pp_macros


subroutine routine_empty_spec
write(*,*) 'Hello world!'
end subroutine routine_empty_spec


! TODO: Below are placeholders for more testing
! subroutine routine_imports (...)
!   ! Test submodule and header imports

! end subroutine routine_imports


! subroutine routine_associates (...)
!   ! Test associate mapping

! end subroutine routine_associates


subroutine routine_member_procedures(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar

  localvar = in2

  call member_procedure(in1, out1)
  out2 = member_function(out1)
contains
  subroutine member_procedure(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1

    out1 = 5 * in1 + localvar
  end subroutine member_procedure

  ! This is just a dummy comment to test that no frontend trips
  ! over things that appear in the contains part and that are
  ! neither subroutine or function definitions.

  function member_function(a)
    ! This function is just included to test that functions
    ! are also possible
    implicit none
    integer, intent(in) :: a
    integer :: member_function

    member_function = 3 * a + 2
  end function member_function
end subroutine routine_member_procedures
