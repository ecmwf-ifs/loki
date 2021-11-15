! Only the FLAT and CONTIG option require OepnMP parallelism
! on the inner kernel loops, so put it in selectively.
#if defined(FLAT) || defined(CONTIG)
#define do_parallel !$OMP do schedule(static)
#define end_do_parallel !$OMP end do
#else
#define do_parallel
#define end_do_parallel
#endif

module phys_mod

use, intrinsic :: iso_fortran_env

use omp_lib

implicit none

integer, parameter :: sp = REAL32
integer, parameter :: dp = REAL64
#ifdef FLOAT_SINGLE
integer, parameter :: lp = sp     !! lp : "local" precision
#else
integer, parameter :: lp = dp     !! lp : "local" precision
#endif

integer, parameter :: ip = INT64

real(kind=lp) :: cst1 = 2.5, cst2 = 3.14
integer, parameter :: nspecies = 5

type in_struct
  real(kind=lp), dimension(:,:), pointer :: in1,in2,in3,in4,in5,in6,in7,in8,in9,in10
end type in_struct

contains 


! Overload the array access with a macro called "x" (yes, X!)
#if defined(AOSOA)
#define x(f, a, b) structure%f(a, b)
#else
#define x(f, a, b) f(a, b)
#endif


#if defined(AOSOA)
subroutine phys_kernel_LITE_LOOP(dim1,dim2,i1,i2,structure, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  type(in_struct), intent(inout) :: structure
#else
subroutine phys_kernel_LITE_LOOP(dim1,dim2,i1,i2, in1,in2,in3,in4,in5,in6,in7,in8,in9,in10, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: in1,in2,in3,in4,in5,in6,in7,in8,in9,in10
#endif
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: out1

  integer(kind=ip) :: i,k
  do k=1,dim2
    do_parallel
    do i=i1,i2
      out1(i,k) = (x(in1,i,k) + x(in2,i,k) + x(in3,i,k) + x(in4,i,k) + x(in5,i,k) + &
 &                 x(in6,i,k) + x(in7,i,k) + x(in8,i,k) + x(in9,i,k) + x(in10,i,k)) * 0.1
      x(in1,i,k) = out1(i,k)
    end do
    end_do_parallel
  end do
end subroutine phys_kernel_LITE_LOOP

#if defined(AOSOA)
subroutine phys_kernel_VERT_SEARCH(dim1,dim2,i1,i2,structure, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  type(in_struct), intent(inout) :: structure
#else
subroutine phys_kernel_VERT_SEARCH(dim1,dim2,i1,i2, in1,in2,in3,in4,in5,in6,in7,in8,in9,in10, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: in1,in2,in3,in4,in5,in6,in7,in8,in9,in10
#endif
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: out1

  integer(kind=ip) :: i,k
  real(kind=lp) :: temp(i1:i2)
  integer :: kmax(i1:i2)

  temp = -1.
  kmax = -1
  do k=1,dim2
    do_parallel
    do i=i1,i2
      if (x(in1,i,k) > temp(i)) then
        temp(i) = x(in1,i,k)
        kmax(i) = k
      end if
    end do
    end_do_parallel
  end do

  do_parallel
  do i=i1,i2
    do k=1,kmax(i)
      out1(i,k) = (x(in1,i,k) + x(in2,i,k) + x(in3,i,k) + x(in4,i,k) + x(in5,i,k) + &
 &                 x(in6,i,k) + x(in7,i,k) + x(in8,i,k) + x(in9,i,k) + x(in10,i,k)) * 0.1
      x(in1,i,k) = out1(i,k)
    end do
  end do
  end_do_parallel
  
  do_parallel
  do i=i1,i2
    do k=kmax(i)+1,dim2
      out1(i,k) = (x(in1,i,k) * x(in2,i,k) * x(in3,i,k) * x(in4,i,k) * x(in5,i,k) + &
 &                 x(in6,i,k) * x(in7,i,k) * x(in8,i,k) * x(in9,i,k) * x(in10,i,k)) * 0.3
      x(in1,i,k) = out1(i,k)
    end do
  end do
  end_do_parallel
end subroutine phys_kernel_VERT_SEARCH

#if defined(AOSOA)
subroutine phys_kernel_NASTY_EXPS(dim1,dim2,i1,i2,structure, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  type(in_struct), intent(inout) :: structure
#else
subroutine phys_kernel_NASTY_EXPS(dim1,dim2,i1,i2, in1,in2,in3,in4,in5,in6,in7,in8,in9,in10, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: in1,in2,in3,in4,in5,in6,in7,in8,in9,in10
#endif
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: out1

  integer(kind=ip) :: i,k
  real(kind=lp) :: temp_s1, temp_s2

  do k=1,dim2
    do_parallel
    do i=i1,i2
      temp_s1 = (x(in1,i,k) + x(in2,i,k) + x(in3,i,k) + x(in4,i,k) + x(in5,i,k) + &
 &                 x(in6,i,k) + x(in7,i,k) + x(in8,i,k) + x(in9,i,k) + x(in10,i,k)) * 0.1
      temp_s1 = min( exp( (temp_s1 - cst1) / (temp_s1 - cst2) ), exp(x(in4,i,k)-x(in5,i,k)) )

      temp_s2 = (x(in1,i,k) - x(in2,i,k) * x(in3,i,k) - x(in4,i,k) + (x(in5,i,k) - &
 &                 x(in6,i,k)*0.5) + (x(in7,i,k) - x(in8,i,k)*0.1) - x(in9,i,k) - x(in10,i,k)) * 0.2
      temp_s2 = min( exp( (temp_s2 - cst2) / (temp_s2 - cst1) ), exp(x(in6,i,k)+x(in7,i,k)) )

      if (temp_s1 < temp_s2) then
        out1(i,k) = temp_s1
      else 
        out1(i,k) = temp_s2
      end if

      x(in1,i,k) = out1(i,k)
    end do
    end_do_parallel
  end do
end subroutine phys_kernel_NASTY_EXPS

#if defined(AOSOA)
subroutine phys_kernel_LU_SOLVER(dim1,dim2,i1,i2,structure, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  type(in_struct), intent(inout) :: structure
#else
subroutine phys_kernel_LU_SOLVER(dim1,dim2,i1,i2, in1,in2,in3,in4,in5,in6,in7,in8,in9,in10, out1)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: in1,in2,in3,in4,in5,in6,in7,in8,in9,in10
#endif
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: out1

  integer(kind=ip) :: i,k
  real(kind=lp),dimension(dim1,nspecies,nspecies) :: lu_lhs, lu_rhs_implicit
  integer(kind=ip) :: s1,s2,s3, jm,jn
  real(kind=lp) :: dp(i1:i2), temp_hor1(i1:i2)
  real(kind=lp) :: temp_out(i1:i2,nspecies), out_lev_m_1(i1:i2,nspecies)

  ! initialise for k=1
  out_lev_m_1 = 0.

  do k=1,dim2
 
    do_parallel
    do i=i1,i2
      lu_rhs_implicit(i,:,:) = 0.5
      lu_lhs(i,:,:) = 0.1
    end do
    end_do_parallel

    if (k<dim2) then
      do_parallel
      do i=i1,i2
        dp(i) = x(in2,i,k+1)-x(in2,i,k)
      end do
      end_do_parallel
    else
      do_parallel
      do i=i1,i2
        dp(i) = 1.0
      end do
      end_do_parallel
    end if

    do_parallel
    do i=i1,i2
      temp_hor1(i) = x(in1,i,k)*x(in2,i,k) + x(in5,i,k)/x(in6,i,k)
      temp_hor1(i) = temp_hor1(i) - floor(temp_hor1(i))
    end do
    end_do_parallel

    do_parallel
    do i=i1,i2
      lu_rhs_implicit(i,2,2) = (x(in4,i,k+1) + x(in7,i,k+1)) * dp(i) !! diagonal term, are NCLDQL & NCLDQI consecutive? shouldn't really matter as long as diag is non-zero
      lu_rhs_implicit(i,4,4) = (x(in4,i,k+1) + x(in7,i,k+1)) * dp(i) !! idem 
    end do
    end_do_parallel

    ! modify local value of rhs_impl based on some local input value like temperature
    do_parallel
    do i=i1,i2
      if (x(in3,i,k) < 0.1) then
        lu_rhs_implicit(i,2,4) = lu_rhs_implicit(i,2,4) + x(in7,i,k) 
        lu_rhs_implicit(i,3,4) = lu_rhs_implicit(i,3,4) + x(in9,i,k) 
      end if
    end do
    end_do_parallel

    ! modify value of rhs_impl based on local derived value 
    do_parallel
    do i=i1,i2
      if (temp_hor1(i) > 0.8) then
        lu_rhs_implicit(i,1,4) = lu_rhs_implicit(i,1,4) + x(in6,i,k) 
        lu_rhs_implicit(i,3,4) = lu_rhs_implicit(i,3,4) + x(in8,i,k) 
      end if
    end do
    end_do_parallel

    ! set up lhs properly
    do s2=1,nspecies
      do s1=1,nspecies
        if (s1==s2) then
          do_parallel
          do i=i1,i2
            lu_lhs(i,s1,s1) = lu_lhs(i,s1,s1) + sum(lu_rhs_implicit(i,:,s1))  ! diagonal term 
          end do
          end_do_parallel
        else 
          do_parallel
          do i=i1,i2
            lu_lhs(i,s1,s2) = - lu_rhs_implicit(i,s1,s2) ! off-diagonal
          end do
          end_do_parallel
        end if
      end do ! s1
    end do ! s2

    ! set the rhs hopefully plausibly 
    do s2=1,nspecies 
      do_parallel
      do i=i1,i2
        if (s2==1) then
          temp_out(i,s2)= x(in1,i,k) + x(in9,i,k) ! zexplicit
        elseif (s2==2) then
          temp_out(i,s2)= x(in2,i,k) + x(in9,i,k) ! zexplicit
        elseif (s2==3) then
          temp_out(i,s2)= x(in3,i,k) + x(in9,i,k) ! zexplicit
        elseif (s2==4) then
          temp_out(i,s2)= x(in4,i,k) + x(in9,i,k) ! zexplicit
        elseif (s2==5) then
          temp_out(i,s2)= x(in5,i,k) + x(in9,i,k) ! zexplicit
        end if
      enddo
      end_do_parallel
    enddo

    ! following factorization code taken straight from CLOUDSC
    ! Non pivoting recursive factorization 
    do s2 = 1, nspecies-1
      do s1 = s2+1,nspecies
        do_parallel
        do i=i1,i2
          lu_lhs(i,s1,s2)=lu_lhs(i,s1,s2) / lu_lhs(i,s2,s2)
        enddo
        end_do_parallel
        do s3=s2+1,nspecies
          do_parallel
          do i=i1,i2
            lu_lhs(i,s1,s3)=lu_lhs(i,s1,s3)-lu_lhs(i,s1,s2)*lu_lhs(i,s2,s3)
          enddo ! do i
          end_do_parallel
        enddo ! do s3
      enddo ! do s1
    enddo ! do s2

    ! backsubstitution 
    !  step 1 
    do s2=2,nspecies
      do s1 = 1,s2-1
        do_parallel
        do i=i1,i2
          temp_out(i,s2)=temp_out(i,s2)-lu_lhs(i,s2,s1) * temp_out(i,s1)
        end do !  i
        end_do_parallel
      end do ! s1
    end do ! s2
    !  step 2
    do_parallel
    do i=i1,i2
      temp_out(i,nspecies)=temp_out(i,nspecies)/lu_lhs(i,nspecies,nspecies)
    end do !  i
    end_do_parallel
    do s2=nspecies-1,1,-1
      do s1 = s2+1,nspecies
        do_parallel
        do i=i1,i2
          temp_out(i,s2)=temp_out(i,s2)-lu_lhs(i,s2,s1) * temp_out(i,s1)
        end do !  i
        end_do_parallel
      end do ! s1
      do_parallel
      do i=i1,i2
        temp_out(i,s2)=temp_out(i,s2)/lu_lhs(i,s2,s2)
      end do !  i
      end_do_parallel
    enddo ! s2

    ! extract solution values into output
    do_parallel
    do i=i1,i2
      out1(i,k) = sum(temp_out(i,:))
    end do
    end_do_parallel
    ! save k level values for use at k+1
    do s1=1,nspecies
      do_parallel
      do i=i1,i2
        out_lev_m_1(i,s1) = temp_out(i,s1)
      end do ! i
      end_do_parallel
    end do ! s1

  end do !! do k

end subroutine phys_kernel_LU_SOLVER

subroutine phys_kernel_LU_SOLVER_COMPACT(dim1,dim2,i1,i2, in1,in2,in3,in4,in5,in6,in7,in8,in9,in10, out1)
  ! To satisfy my curiosity, flip the allocation of the matrix to be compact for each grid point.
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: in1,in2,in3,in4,in5,in6,in7,in8,in9,in10
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: out1

  integer(kind=ip) :: i,k
  ! Invert the matrix allocation, so that parallel dim is outermost
  real(kind=lp),dimension(nspecies,nspecies,dim1) :: lu_lhs, lu_rhs_implicit
  integer(kind=ip) :: s1,s2,s3, jm,jn
  real(kind=lp) :: dp(i1:i2), temp_hor1(i1:i2)
  real(kind=lp) :: temp_out(i1:i2,nspecies), out_lev_m_1(i1:i2,nspecies)

  ! initialise for k=1
  out_lev_m_1 = 0.

  do k=1,dim2
 
    do_parallel
    do i=i1,i2
      lu_rhs_implicit(:,:,i) = 0.5
      lu_lhs(:,:,i) = 0.1
    end do
    end_do_parallel

    if (k<dim2) then
      do_parallel
      do i=i1,i2
        dp(i) = in2(i,k+1)-in2(i,k)
      end do
      end_do_parallel
    else
      do_parallel
      do i=i1,i2
        dp(i) = 1.0
      end do
      end_do_parallel
    end if

    do_parallel
    do i=i1,i2
      temp_hor1(i) = in1(i,k)*in2(i,k) + in5(i,k)/in6(i,k)
      temp_hor1(i) = temp_hor1(i) - floor(temp_hor1(i))
    end do
    end_do_parallel

    do_parallel
    do i=i1,i2
      lu_rhs_implicit(2,2,i) = (in4(i,k+1) + in7(i,k+1)) * dp(i) !! diagonal term, are NCLDQL & NCLDQI consecutive? shouldn't really matter as long as diag is non-zero
      lu_rhs_implicit(4,4,i) = (in4(i,k+1) + in7(i,k+1)) * dp(i) !! idem 
    end do
    end_do_parallel

    ! modify local value of rhs_impl based on some local input value like temperature
    do_parallel
    do i=i1,i2
      if (in3(i,k) < 0.1) then
        lu_rhs_implicit(2,4,i) = lu_rhs_implicit(2,4,i) + in7(i,k) 
        lu_rhs_implicit(3,4,i) = lu_rhs_implicit(3,4,i) + in9(i,k) 
      end if
    end do
    end_do_parallel

    ! modify value of rhs_impl based on local derived value 
    do_parallel
    do i=i1,i2
      if (temp_hor1(i) > 0.8) then
        lu_rhs_implicit(1,4,i) = lu_rhs_implicit(1,4,i) + in6(i,k) 
        lu_rhs_implicit(3,4,i) = lu_rhs_implicit(3,4,i) + in8(i,k) 
      end if
    end do
    end_do_parallel

    ! set up lhs properly
    do s2=1,nspecies
      do s1=1,nspecies
        if (s1==s2) then
          do_parallel
          do i=i1,i2
            lu_lhs(s1,s1,i) = lu_lhs(s1,s1,i) + sum(lu_rhs_implicit(:,s1,i))  ! diagonal term 
          end do
          end_do_parallel
        else 
          do_parallel
          do i=i1,i2
            lu_lhs(s1,s2,i) = - lu_rhs_implicit(s1,s2,i) ! off-diagonal
          end do
          end_do_parallel
        end if
      end do ! s1
    end do ! s2

    ! set the rhs hopefully plausibly 
    do s2=1,nspecies 
      do_parallel
      do i=i1,i2
        if (s2==1) then
          temp_out(i,s2)= in1(i,k) + in9(i,k) ! zexplicit
        elseif (s2==2) then
          temp_out(i,s2)= in2(i,k) + in9(i,k) ! zexplicit
        elseif (s2==3) then
          temp_out(i,s2)= in3(i,k) + in9(i,k) ! zexplicit
        elseif (s2==4) then
          temp_out(i,s2)= in4(i,k) + in9(i,k) ! zexplicit
        elseif (s2==5) then
          temp_out(i,s2)= in5(i,k) + in9(i,k) ! zexplicit
        end if
      enddo
      end_do_parallel
    enddo

    ! following factorization code taken straight from CLOUDSC
    ! Non pivoting recursive factorization 
    do s2 = 1, nspecies-1
      do s1 = s2+1,nspecies
        do_parallel
        do i=i1,i2
          lu_lhs(s1,s2,i)=lu_lhs(s1,s2,i) / lu_lhs(s2,s2,i)
        enddo
        end_do_parallel
        do s3=s2+1,nspecies
          do_parallel
          do i=i1,i2
            lu_lhs(s1,s3,i)=lu_lhs(s1,s3,i)-lu_lhs(s1,s2,i)*lu_lhs(s2,s3,i)
          enddo ! do i
          end_do_parallel
        enddo ! do s3
      enddo ! do s1
    enddo ! do s2

    ! backsubstitution 
    !  step 1 
    do s2=2,nspecies
      do s1 = 1,s2-1
        do_parallel
        do i=i1,i2
          temp_out(i,s2)=temp_out(i,s2)-lu_lhs(s2,s1,i) * temp_out(i,s1)
        end do !  i
        end_do_parallel
      end do ! s1
    end do ! s2
    !  step 2
    do_parallel
    do i=i1,i2
      temp_out(i,nspecies)=temp_out(i,nspecies)/lu_lhs(nspecies,nspecies,i)
    end do !  i
    end_do_parallel
    do s2=nspecies-1,1,-1
      do s1 = s2+1,nspecies
        do_parallel
        do i=i1,i2
          temp_out(i,s2)=temp_out(i,s2)-lu_lhs(s2,s1,i) * temp_out(i,s1)
        end do !  i
        end_do_parallel
      end do ! s1
      do_parallel
      do i=i1,i2
        temp_out(i,s2)=temp_out(i,s2)/lu_lhs(s2,s2,i)
      end do !  i
      end_do_parallel
    enddo ! s2

    ! extract solution values into output
    do_parallel
    do i=i1,i2
      out1(i,k) = sum(temp_out(i,:))
    end do
    end_do_parallel
    ! save k level values for use at k+1
    do s1=1,nspecies
      do_parallel
      do i=i1,i2
        out_lev_m_1(i,s1) = temp_out(i,s1)
      end do ! i
      end_do_parallel
    end do ! s1

  end do !! do k

end subroutine phys_kernel_LU_SOLVER_COMPACT

end module phys_mod
