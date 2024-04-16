module sourcefile_cpp_stmt_func_mod

    IMPLICIT NONE

    ! originally declared in parkind1.F90
    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    INTEGER, PARAMETer :: JPRB = SELECTED_REAL_KIND(13,300)

contains

subroutine sourcefile_cpp_stmt_func(KIDIA, KFDIA, KLON, KLEV, ZFOEEW)
    INTEGER(KIND=JPIM),INTENT(IN)    :: KLON, KLEV
    INTEGER(KIND=JPIM),INTENT(IN)    :: KIDIA 
    INTEGER(KIND=JPIM),INTENT(IN)    :: KFDIA 
    REAL(KIND=JPRB)   ,INTENT(OUT)   :: ZFOEEW(KLON,KLEV)

    INTEGER(KIND=JPIM) :: JK, JL

    REAL(KIND=JPRB) :: ZTP1(KLON,KLEV)   
    REAL(KIND=JPRB) :: PAP(KLON,KLEV)
    REAL(KIND=JPRB) :: ZALFA

    ! originally declared in yomcst.F90
    REAL(KIND=JPRB) :: RTT = 1._JPRB

    ! originally declared in yoethf.F90
    REAL(KIND=JPRB) :: R2ES = 2._JPRB
    REAL(KIND=JPRB) :: R3LES = 3._JPRB
    REAL(KIND=JPRB) :: R3IES = 3._JPRB
    REAL(KIND=JPRB) :: R4LES = 4._JPRB
    REAL(KIND=JPRB) :: R4IES = 4._JPRB

#include "stmt.func.h"

    ! initialize with some stupid values
    PAP(:,:) = 8._JPRB
    ZTP1(:,:) = 1._JPRB

    DO JK=1,KLEV
        DO JL=KIDIA,KFDIA
            ZALFA=FOEDELTA(ZTP1(JL,JK))

            ! this should essentially become: min((zalfa * 2 + (1-zalfa) * 2))/8,0.5) === 0.25
            ZFOEEW(JL,JK)=MIN((ZALFA*FOEELIQ(ZTP1(JL,JK))+ &
                &  (1.0_JPRB-ZALFA)*FOEEICE(ZTP1(JL,JK)))/PAP(JL,JK),0.5_JPRB)
        END DO
    END DO
end subroutine sourcefile_cpp_stmt_func

end module sourcefile_cpp_stmt_func_mod