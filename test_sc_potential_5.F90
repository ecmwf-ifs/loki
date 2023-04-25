SUBROUTINE TEST_SC_POTENTIAL_5(A, B, N)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: N
  INTEGER, INTENT(INOUT) :: A(N, N), B(N, N)
  INTEGER :: RES_SUM, JK, JL
  INTEGER :: i_all, i_any, i_count, i_maxval, i_minval, i_product, i_sum
  INTEGER :: i2_sum, i2_maxval

  DO JK=1,N
    DO JL=1,N
      A(JL, JK) = 10
      B(JL, JK) = 20
    END DO
  END DO

  DO JK=1,N
    DO JL=1,N
      A(JL, JK) = A(JL, JK) + 1
      RES_SUM = RES_SUM + A(JL, JK)*B(JL, JK)
    END DO
  END DO

  i_all = all(A>5)
  i_any = any(A>5)
  i_count = count(A>5)
  i_maxval = maxval(A)
  i_minval = minval(A)
  i_product = product(A)
  i_sum = sum(A)

  i2_sum = sum(A(1, :))
  i2_sum = sum(A, dim=2)
  i2_maxval = maxval(A(1, :))

END SUBROUTINE TEST_SC_POTENTIAL_5
