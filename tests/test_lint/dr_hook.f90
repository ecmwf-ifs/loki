subroutine routine_okay
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle 

! Comments are non-executable statements

if (lhook) call dr_hook('routine_okay', 0, zhook_handle)

print *, "Foo bar"

if (lhook) call dr_hook('routine_okay', 1, zhook_handle)

! Comments are non-executable statements

contains

subroutine routine_contained_okay
real(kind=jprb) :: zhook_handle 

if (lhook) call dr_hook('routine_okay%routine_contained_okay', 0, zhook_handle)

print *, "Foo bar"

if (lhook) call dr_hook('routine_okay%routine_contained_okay', 1, zhook_handle)
end subroutine routine_contained_okay
end subroutine routine_okay


subroutine routine_not_okay_a
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle

! Error: no conditional IF(LHOOK)
call dr_hook('routine_not_okay_a', 0, zhook_handle)

print *, "Foo bar"

! Error: subroutine name not in string argument
if (lhook) call dr_hook('foobar', 1, zhook_handle)
end subroutine routine_not_okay_a


subroutine routine_not_okay_b
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle

! Error: second argument is not 0 or 1
if (lhook) call dr_hook('routine_not_okay_b', 2, zhook_handle)

print *, "Foo bar"

! Error: third argument is not zhook_handle 
if (lhook) call dr_hook('routine_not_okay_b', 1)
end subroutine routine_not_okay_b


subroutine routine_not_okay_c
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle
real(kind=jprb) :: red_herring

red_herring = 1.0

! Error: Executable statement before call to dr_hook 
if (lhook) call dr_hook('routine_not_okay_c', 2, zhook_handle)

print *, "Foo bar"

! Error: Executable statement after call to dr_hook 
if (lhook) call dr_hook('routine_not_okay_c', 1, zhook_handle)

red_herring = 2.0
end subroutine routine_not_okay_c


subroutine routine_not_okay_d
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle
real(kind=jprb) :: red_herring

! Error: First call to dr_hook is missing 

red_herring = 1.0
print *, "Foo bar"

if (lhook) call dr_hook('routine_not_okay_d', 1, zhook_handle)

end subroutine routine_not_okay_d


subroutine routine_not_okay_e
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle
real(kind=jprb) :: red_herring

if (lhook) call dr_hook('routine_not_okay_e', 0, zhook_handle)

red_herring = 1.0
print *, "Foo bar"

! Error: Last call to dr_hook is missing 

contains

subroutine routine_contained_not_okay
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle
real(kind=jprb) :: red_herring

if (lhook) call dr_hook('routine_not_okay_e%routine_contained_not_okay', 0, zhook_handle)

red_herring = 1.0
print *, "Foo bar"

! Error: String argument is not "<parent routine>%<contained routine>"
if (lhook) call dr_hook('routine_contained_not_okay', 1, zhook_handle)
end subroutine routine_contained_not_okay
end subroutine routine_not_okay_e


module some_mod

contains

subroutine mod_routine_okay
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle

if (lhook) call dr_hook('some_mod:mod_routine_okay', 0, zhook_handle)
print *, "Foo bar"
if (lhook) call dr_hook('some_mod:mod_routine_okay', 1, zhook_handle)

contains

subroutine mod_contained_routine_okay
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle

if (lhook) call dr_hook('some_mod:mod_routine_okay%mod_contained_routine_okay', 0, zhook_handle)
print *, "Foo bar"
if (lhook) call dr_hook('some_mod:mod_routine_okay%mod_contained_routine_okay', 1, zhook_handle)
end subroutine mod_contained_routine_okay
end subroutine mod_routine_okay

subroutine mod_routine_not_okay
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle

! Error: String argument does not contain module name
if (lhook) call dr_hook('mod_routine_okay', 0, zhook_handle)
print *, "Foo bar"
if (lhook) call dr_hook('some_mod:mod_routine_not_okay', 1, zhook_handle)

contains

subroutine mod_contained_routine_not_okay
use yomhook, only: lhook, dr_hook
real(kind=jprb) :: zhook_handle

! Error: String argument does not contain module name
if (lhook) call dr_hook('mod_routine_not_okay%mod_contained_routine_not_okay', 0, zhook_handle)
print *, "Foo bar"
! Error: String argument does not contain parent routine name
! Error: Second argument is not 0 or 1
if (lhook) call dr_hook('some_mod:mod_contained_routine_not_okay', 8, zhook_handle)
end subroutine mod_contained_routine_not_okay
end subroutine mod_routine_not_okay
end module some_mod
