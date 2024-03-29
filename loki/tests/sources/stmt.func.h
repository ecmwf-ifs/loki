!*
! ---------------------------------------------------

!   Sample of statement functions externalized into a
!   header file similar to how they are included into
!   Fortran source code in the IFS.
!
!   This is an excerpt from fcttre.func.h
!
! ---------------------------------------------------
REAL(KIND=JPRB) :: FOEDELTA
REAL(KIND=JPRB) :: PTARE
FOEDELTA (PTARE) = MAX (0.0_JPRB,SIGN(1.0_JPRB,PTARE-RTT))

REAL(KIND=JPRB) :: FOEEWMO, FOEELIQ, FOEEICE 
FOEEWMO( PTARE ) = R2ES*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
FOEELIQ( PTARE ) = R2ES*EXP(R3LES*(PTARE-RTT)/(PTARE-R4LES))
FOEEICE( PTARE ) = R2ES*EXP(R3IES*(PTARE-RTT)/(PTARE-R4IES))

! ---------------------------------------------------
