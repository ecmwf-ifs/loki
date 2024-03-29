module some_module
implicit none

  integer, parameter :: some_var=1

  interface add_args
    procedure add_two_args
    procedure add_three_args
  end interface add_args

  type some_type
    real :: c = 1.
    contains
      procedure :: do_something => add_const
  end type

contains

  function return_one() result(one)
     real :: one
     one = 1.
  end function return_one

  function add_two_args(i,j) result(res)
     integer, intent(in) :: i,j
     real :: res
     res = dble(i+j)
  end function add_two_args

  function add_three_args(i,j,k) result(res)
     integer, intent(in) :: i,j,k
     real :: res
     res = dble(i+j+k)
  end function add_three_args

  subroutine add_const(self, a)
     class(some_type), intent(in) :: self
     real, intent(inout) :: a

     a = a + self%c
  end subroutine add_const

end module some_module
