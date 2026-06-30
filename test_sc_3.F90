MODULE TEST_SC_3_MOD
CONTAINS
  SUBROUTINE TEST_SC_3(KIDIA, KFIDIA, Z, KLON, KLEV)

    INTEGER, INTENT(IN) :: KIDIA, KFIDIA, KLON, KLEV, HORIZONTAL
    REAL, INTENT(INOUT) :: Z(KLON, KLEV)
    REAL :: TMP
    INTEGER :: JL, JK, JO

    DO JK=1,KLEV
      DO JL=KIDIA,KFIDIA
        Z(JL, JK) = EXP(0.0)
        Z(JL, JK) = 1.0 + EXP(0.0)
        TMP = MAX(1.0, 1.3) + MIN(-1.5, -1.7)
        CALL SOME_FUNC()
        print *, 'intrinsic but probably nothing to allow ...'
      END DO
    END DO

  END SUBROUTINE TEST_SC_3

  SUBROUTINE SOME_FUNC(Z, KLON, KLEV)
    INTEGER, INTENT(IN) :: KLON, KLEV
    REAL, INTENT(INOUT) :: Z(KLON, KLEV)
    print *, "not an intrinsic function ..."
  END SUBROUTINE SOME_FUNC

END MODULE TEST_SC_3_MOD
