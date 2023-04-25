MODULE TEST_SC_4_MOD
CONTAINS
  SUBROUTINE TEST_SC_4(KIDIA, KFIDIA, ZTP1, KLON, KLEV)

    INTEGER, INTENT(IN) :: KIDIA, KFIDIA, KLON, KLEV
    REAL, INTENT(INOUT) :: ZTP1(KLON, KLEV)
    REAL :: SCALAR_TMP, TMP(KLON)
    INTEGER :: JL, JK, JO, JN, JM

    DO JK=1,KLEV
      TMP(JL) = JL + 1
      DO JL=KIDIA,KFIDIA
        ZTP1(JL, JK) = 0.0
      END DO
      DO JL=KIDIA,KFIDIA
        ZTP1(JL, JK) = 0.2
      END DO
    END DO

    DO JK=1,KLEV-1
      DO JL=KIDIA,KFIDIA
        IF (JL > 1) THEN
          JO = JL - 1
        ELSE
          JO = JL
        END IF
        ZTP1(JO + JL, JK) = 1.0
        ZTP1(TMP(JL), JK) = 2.0
      END DO
    END DO

    DO JK=1,KLEV
      DO JL=KIDIA,KFIDIA
        JO = TMP(JL)
        ZTP1(JO, JK) = 3.0
      END DO
    END DO

    DO JK=1,KLEV-1
      DO JL=KIDIA,KFIDIA
        IF (JL > 1) THEN
          JN = JL - 1
        ELSE
          JN = JL
        END IF
        ZTP1(JN, JK) = 1.0
        ZTP1(TMP(JL), JK) = 4.0
      END DO
    END DO

    DO JK=1,KLEV
      DO JL=KIDIA,KFIDIA
        JO = TMP(JL)
        SCALAR_TMP = ZTP1(JO, JK)
        ZTP1(JL, JK) = ZTP1(JO, JK)
      END DO
    END DO

  END SUBROUTINE TEST_SC_4
END MODULE TEST_SC_4_MOD
