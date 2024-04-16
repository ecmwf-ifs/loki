module module_derived_type

   type point
      integer :: n
      real, allocatable :: x(:)
      real, allocatable :: y(:)
      real, allocatable :: z(:)
   end type point

   type grid
      type(point), allocatable :: p(:)
   end type grid

   type(point) :: p, p0
   type(point), allocatable :: p_array(:)
   type(grid) :: g

end module module_derived_type
