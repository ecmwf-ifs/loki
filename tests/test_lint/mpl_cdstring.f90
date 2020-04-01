subroutine routine_okay
use mpl_module
call mpl_init(cdstring='routine_okay')
end subroutine routine_okay

subroutine routine_also_okay
use MPL_MODULE
call MPL_INIT(KPROCS=5, CDSTRING='routine_also_okay')
end subroutine routine_also_okay

subroutine routine_not_okay
use mpl_module
call mpl_init
end subroutine routine_not_okay

subroutine routine_also_not_okay
use MPL_INIT
call MPL_INIT(kprocs=5)
end subroutine routine_also_not_okay
