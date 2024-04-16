module header_mod

  integer, parameter :: jprb = 4

  type header_type
    real(kind=jprb) :: scalar, vector(:), matrix(3, 3)
  end type header_type


end module header_mod
