module header
  ! header module to provide external typedefs
  integer, parameter :: jprb = selected_real_kind(13,300)

  type derived_type
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type derived_type

end module header
