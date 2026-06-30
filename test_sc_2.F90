MODULE TEST_SC_2_MOD
CONTAINS
  SUBROUTINE TEST_SC_2(KIDIA, KFIDIA, Z, KLON, KLEV)

    INTEGER, INTENT(IN) :: KIDIA, KFIDIA, KLON, KLEV
    REAL, INTENT(INOUT) :: Z(KLON, KLEV)
    REAL :: TMP
    INTEGER :: JL, JK, JO

    DO JK=1,KLEV
      DO JL=KIDIA,KFIDIA
        Z(JL, JK) = 0.0
        TMP = Z(JL, JK)
      END DO
      Z(KIDIA:KFIDIA, JK) = 0.2
      Z(:, JK) = 0.5
      TMP = Z(KIDIA:KFIDIA, JK)
    END DO

    DO JK=1,KLEV
      DO JL=KIDIA,KFIDIA
        Z(JL, JK) = 0.0
        TMP = Z(JL, JK)
        DO JO=1,10
          print *, "innermost loop not the horizontal loop"
        END DO
      END DO
    END DO

  END SUBROUTINE TEST_SC_2
END MODULE TEST_SC_2_MOD
