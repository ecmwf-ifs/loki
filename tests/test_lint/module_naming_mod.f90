! This is ok
module module_naming_mod
integer foo
contains
subroutine bar
integer foobar
end subroutine bar
end module module_naming_mod

! This should complain about wrong module and file name
module module_naming
integer baz
end module module_naming
