module func_mod
implicit none
contains

  real function some_func()
    use moduleB, only: var2
    implicit none
    real some_func
    some_func = var2
  end function some_func

end module func_mod
