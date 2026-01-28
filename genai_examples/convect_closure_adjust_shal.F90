SUBROUTINE CONVECT_CLOSURE_ADJUST_SHAL(KLON, KLEV, PADJ, PUMF, PZUMF, PUER, PZUER, PUDR, PZUDR)
    USE PARKIND1, ONLY: JPRB
    ! ... (other declarations and initializations)

    implicit none

    REAL, INTENT(INOUT), DIMENSION(KLON, KLEV) :: PUMF  ! updraft mass flux (kg/s)
    REAL, INTENT(INOUT), DIMENSION(KLON, KLEV) :: PZUMF  ! initial value of  "
    REAL, INTENT(INOUT), DIMENSION(KLON, KLEV) :: PUER  ! updraft entrainment (kg/s)
    REAL, INTENT(INOUT), DIMENSION(KLON, KLEV) :: PZUER  ! initial value of  "
    REAL, INTENT(INOUT), DIMENSION(KLON, KLEV) :: PUDR  ! updraft detrainment (kg/s)
    REAL, INTENT(INOUT), DIMENSION(KLON, KLEV) :: PZUDR  ! initial value of  "
  !
  !
  !*       0.2   Declarations of local variables :
  !
    INTEGER :: IKB, IKE  ! vert. loop bounds
    INTEGER :: JK  ! vertical loop index
    
    !$acc parallel loop collapse(2) private(JK, i_PUMF_0) present(KLON, KLEV, PADJ, PZUMF, PZUER, PZUDR, PUMF, PUER, PUDR)
    DO JK=1 + JCVEXB + 1, KLEV - JCVEXT
        DO i_PUMF_0=1, KLON
            !$acc loop vector
            PUMF(i_PUMF_0, JK) = PZUMF(i_PUMF_0, JK)*PADJ(i_PUMF_0)
            PUER(i_PUMF_0, JK) = PZUER(i_PUMF_0, JK)*PADJ(i_PUMF_0)
            PUDR(i_PUMF_0, JK) = PZUDR(i_PUMF_0, JK)*PADJ(i_PUMF_0)
        END DO
    END DO
    !$acc end parallel loop
    
    END SUBROUTINE CONVECT_CLOSURE_ADJUST_SHAL