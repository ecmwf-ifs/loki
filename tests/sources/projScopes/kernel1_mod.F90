module kernel1_mod
contains
    subroutine kernel
        use kernel1_impl
        implicit none
        call kernel_impl
    end subroutine kernel
end module kernel1_mod
