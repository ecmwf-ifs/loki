subroutine driver
    use kernel1_mod, only: kernel1 => kernel
    use kernel2_mod, only: kernel2 => kernel
    implicit none

    call kernel1
    call kernel2
end subroutine driver
