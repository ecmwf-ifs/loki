subroutine driver_derived_type()
implicit none

!$loki update_device
!$acc serial
call kernel_derived_type()
!$acc end serial
!$loki update_host

end subroutine driver_derived_type
