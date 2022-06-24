module radiation_random_numbers

  implicit none

  public :: rng_type, IRngNative

  enum, bind(c) 
    enumerator IRngNative      ! Built-in Fortran-90 RNG
  end enum

  integer, parameter            :: jpim = selected_int_kind(9)
  integer, parameter            :: jprb = selected_real_kind(13,300)
  integer(kind=jpim), parameter :: NMaxStreams = 512

  type rng_type

    integer(kind=jpim) :: itype = IRngNative
    real(kind=jprb)    :: istate(NMaxStreams) 
    integer(kind=jpim) :: nmaxstreams = NMaxStreams
    integer(kind=jpim) :: iseed = 0

  end type rng_type 

contains

  subroutine rng_default(istate_dim, maxstreams)
    integer, intent(out) :: istate_dim, maxstreams
    type(rng_type) :: rng
    integer :: dim(1)
    rng = rng_type(istate=0._jprb)
    dim = shape(rng%istate)
    istate_dim = dim(1)
    maxstreams = rng%nmaxstreams
  end subroutine rng_default

  subroutine rng_init(istate_dim, maxstreams)
    integer, intent(out) :: istate_dim, maxstreams
    type(rng_type) :: rng
    integer :: dim(1)
    rng = rng_type(nmaxstreams=256, istate=0._jprb)
    dim = shape(rng%istate)
    istate_dim = dim(1)
    maxstreams = rng%nmaxstreams
  end subroutine rng_init

end module radiation_random_numbers