program phys_layout

use omp_lib

use phys_mod, only: ip, lp, &
 &                  phys_kernel_LITE_LOOP, phys_kernel_VERT_SEARCH, phys_kernel_NASTY_EXPS, &
 &                  phys_kernel_LU_SOLVER, phys_kernel_LU_SOLVER_COMPACT

implicit none

!! arrays for passing to subroutine
real(kind=lp),dimension(:,:,:),allocatable :: arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10
real(kind=lp),dimension(:,:,:),allocatable :: out1 

integer(kind=ip) :: nproma, npoints, nlev, nblocks, ntotal, num_main_loops
integer(kind=ip) :: nb, nml, inp, i, i1, i2, k, real_bytes, tid

integer :: i_seed
integer, dimension(:), allocatable :: a_seed

integer :: iargs, lenarg
character(len=20) :: clarg

real(kind=lp) :: time1, time2

! Constants
nlev    = 137
num_main_loops = 8

! Defaults
nproma = 32
ntotal = 32*4096

iargs = command_argument_count()
if (iargs >= 1) then
   call get_command_argument(1, clarg, lenarg)
   read(clarg(1:lenarg),*) nproma

   if (iargs >= 2) then
      call get_command_argument(2, clarg, lenarg)
      read(clarg(1:lenarg),*) ntotal
   endif
endif
npoints = nproma  ! Can be unlocked later...
nblocks = ntotal / nproma + 1
write(*, '(A8,I8,A8,I8,A8,I8,A)') 'NPROMA ', nproma, "TOTAL ", ntotal, "NBLOCKS", nblocks

#ifdef FLOAT32
real_bytes = 4
#else
real_bytes = 8
#endif 

allocate(arr1 (npoints,nlev,nblocks))
allocate(arr2 (npoints,nlev,nblocks))
allocate(arr3 (npoints,nlev,nblocks))
allocate(arr4 (npoints,nlev,nblocks))
allocate(arr5 (npoints,nlev,nblocks))
allocate(arr6 (npoints,nlev,nblocks))
allocate(arr7 (npoints,nlev,nblocks))
allocate(arr8 (npoints,nlev,nblocks))
allocate(arr9 (npoints,nlev,nblocks))
allocate(arr10(npoints,nlev,nblocks))
allocate(out1 (npoints,nlev,nblocks))

! ----- Set up random seed portably -----
call random_seed(size=i_seed)
allocate(a_seed(1:i_seed))
call random_seed(get=a_seed)
a_seed = [ (i,i=1,i_seed) ]
call random_seed(put=a_seed)
deallocate(a_seed)

! Initialize values in parallel region to avoid accidental
! NUMA issues in single-socket mode on dual-socket machines.
!$omp parallel

!$omp master
do inp = 1, ntotal, nproma
  nb = (inp-1) / nproma + 1
  call random_number(arr1(:,:,nb))
  call random_number(arr2(:,:,nb))
  call random_number(arr3(:,:,nb))
  call random_number(arr4(:,:,nb))
  call random_number(arr5(:,:,nb))
  call random_number(arr6(:,:,nb))
  call random_number(arr7(:,:,nb))
  call random_number(arr8(:,:,nb))
  call random_number(arr9(:,:,nb))
  call random_number(arr10(:,:,nb))
end do
!$omp end master

! Pre-processor fudging to inject the relevant kernel call
#ifdef LITE_LOOP
#define PHYS_KERNEL phys_kernel_LITE_LOOP
#elif VERT_SEARCH
#define PHYS_KERNEL phys_kernel_VERT_SEARCH
#elif NASTY_EXPS
#define PHYS_KERNEL phys_kernel_NASTY_EXPS
#elif LU_SOLVER
#define PHYS_KERNEL phys_kernel_LU_SOLVER
#elif LU_SOLVER_COMPACT
#define PHYS_KERNEL phys_kernel_LU_SOLVER_COMPACT
#endif

time1 = omp_get_wtime()

do nml=1,num_main_loops

!$OMP PARALLEL DEFAULT(SHARED), PRIVATE(nb,inp,i1,i2)

! The classic BLOCKED-NPROMA structure from the IFS

!$omp do schedule(static)
do inp = 1, ntotal, nproma
  nb = (inp-1) / nproma + 1
  call PHYS_KERNEL( nproma, nlev, 1_ip, nproma, arr1(:,:,nb), arr2(:,:,nb), arr3(:,:,nb), &
 &                  arr4(:,:,nb), arr5(:,:,nb), &
 &                  arr6(:,:,nb), arr7 (:,:,nb), arr8(:,:,nb), &
 &                  arr9(:,:,nb), arr10(:,:,nb), out1(:,:,nb) )
end do
!$omp end do

!$OMP END PARALLEL

end do ! outer nml loop

time2 = omp_get_wtime()


write(*, '(A,3F12.6)') 'Result check : ', arr1(1,1,1), sum(arr1) / real(npoints*nlev*nblocks), sum(out1) / real(npoints*nlev*nblocks)
write(*, '(A,F8.4)') 'Time for kernel call : ', time2 - time1
write(*, '(A,F12.2,A,F12.2,A)') 'Bandwidth estimate : ', num_main_loops*11*npoints*nlev*nblocks*real_bytes*1.e-6, &
     ' MB transferred: ', num_main_loops*11*npoints*nlev*nblocks*real_bytes*1.e-6 / (time2-time1), ' MB/s '

! TODO: Should really dealloc all this stuff... :(

end program phys_layout
