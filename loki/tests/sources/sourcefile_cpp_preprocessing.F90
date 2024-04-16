subroutine sourcefile_external_preprocessing(a, b)
  real(kind=8), intent(inout) :: a, b
#include "some_header.h"
#ifdef FLAG_SMALL
#define CONSTANT 6
#else
#define CONSTANT 123
#endif

#define ADD_ONE(x) x + 1

  a = ADD_ONE(5)
  b = CONSTANT
end subroutine sourcefile_external_preprocessing
