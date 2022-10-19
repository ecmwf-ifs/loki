module kernel2_mod
contains
    subroutine kernel
        use kernel2_impl
        implicit none
        call kernel_impl
    end subroutine kernel
end module kernel2_mod
