subroutine driver()
implicit none

!$loki update_device
!$acc serial
call kernel0()
!$acc end serial
!$loki update_host

end subroutine driver
