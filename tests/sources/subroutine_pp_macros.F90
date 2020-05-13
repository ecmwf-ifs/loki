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
