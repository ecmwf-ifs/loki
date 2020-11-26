subroutine routine_pp_directives
  print *,"Compiled ",__FILENAME__," on ",__DATE__
#define __FILENAME__ __FILE__
  print *,"This is ",__FILE__,__VERSION__
  y = __LINE__ * 5 + __LINE__
end subroutine routine_pp_directives
