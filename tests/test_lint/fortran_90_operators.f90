subroutine test_routine(ia, ib, ic)
integer, intent(in) :: ia, ib, ic

! This should produce 6 problems (one for each operator)
do while (ia .ge. 3 .or. ia .le. -7)
  if (ib .gt. 5 .or. ib .lt. -1) then
    if (ic .eq. 4 .and. ib .ne. -2) then
      print *, 'Foo'
    end if
  end if
end do 

! This should produce no problems
do while (ia >= 3 .or. ia <= -7)
  if (ib > 5 .or. ib < -1) then
    if (ic == 4 .and. ib /= -2) then
      print *, 'Foo'
    end if
  end if
end do 

! This should report 3 problems
do while (ia >= 3 .or. ia .le. -7)
  if (ib .gt. 5 .or. ib < -1) then
    if (ic .eq. 4 .and. ib == -2) then
      print *, 'Foo'
    end if
  end if
end do 
end subroutine test_routine
