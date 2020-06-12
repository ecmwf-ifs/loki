"""
Test identity of source-to-source translation.

The tests in here do not verify correct representation internally,
they only check whether at the end comes out what went in at the beginning.

"""
import pytest
from loki import Subroutine, OFP, OMNI, FP, fgen


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='This is outright impossible')),
    FP
])
def test_subroutine_conservative(frontend):
    """
    Test that conservative output of fgen reproduces the original source string for
    a simple subroutine.
    This has a few limitations, in particular with respect to the signature of the
    subroutine.
    """
    fcode = """
SUBROUTINE CONSERVATIVE (X, Y, SCALAR, VECTOR, MATRIX)
  IMPLICIT NONE
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)
  INTEGER, INTENT(IN) :: X, Y
  REAL(KIND=JPRB), INTENT(IN) :: SCALAR
  REAL(KIND=JPRB), INTENT(INOUT) :: VECTOR(X)
  REAL(KIND=JPRB), DIMENSION(X, Y), INTENT(OUT) :: MATRIX
  INTEGER :: I, SOME_VERY_LONG_INTEGERS, TO_SEE_IF_LINE_CONTUATION, IS_NOT_ENFORCED_IN_THIS_CASE
  ! Some comment that is very very long and exceeds the line width but should not be wrapped
  DO I=1, X
    VECTOR(I) = VECTOR(I) + SCALAR
!$LOKI SOME NONSENSE PRAGMA WITH VERY LONG TEXT THAT EXCEEDS THE LINE WIDTH LIMIT OF THE OUTPUT
    MATRIX(I, :) = I * VECTOR(I)
  ENDDO
END SUBROUTINE CONSERVATIVE
    """.strip()

    # Parse and re-generate the code
    routine = Subroutine.from_source(fcode, frontend=frontend)
    source = fgen(routine, linewidth=90, conservative=True)
    assert source == fcode


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='This is outright impossible')),
    FP
])
def test_subroutine_simple_fgen(frontend):
    """
    Test that non-conservative output produces the original source string for
    a simple subroutine.
    This has a few limitations, in particular for formatting of expressions.
    """
    fcode = """
SUBROUTINE SIMPLE_FGEN (X, Y, SCALAR, VECTOR, MATRIX)
  IMPLICIT NONE
  INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)
  INTEGER, INTENT(IN) :: X, Y
  REAL(KIND=JPRB), INTENT(IN) :: SCALAR  ! A very long inline comment that should not be wrapped but simply appended
  REAL(KIND=JPRB), INTENT(INOUT) :: VECTOR(X)
  REAL(KIND=JPRB), DIMENSION(X, Y), INTENT(OUT) :: MATRIX
  INTEGER :: I, SOME_VERY_LONG_INTEGERS, TO_SEE_IF_LINE_CONTINUATION,  &
  & WORKS_AS_EXPECTED_IN_ITEM_LISTS
  ! Some comment that is very very long and exceeds the line width but should not be wrapped
  DO I=1,X
    VECTOR(I) = VECTOR(I) + SCALAR
!$LOKI SOME NONSENSE PRAGMA WITH VERY LONG TEXT THAT EXCEEDS THE LINE WIDTH LIMIT OF THE OUTPUT
    MATRIX(I, :) = I*VECTOR(I)
    IF (SOME_VERY_LONG_INTEGERS > X) THEN
      ! Some comment to have more than one line
      ! in the body of the condtional
      IF (TO_SEE_IF_LINE_CONTINUATION > Y) THEN
        PRINT *, 'Intrinsic statement'
      END IF
    END IF
  END DO
END SUBROUTINE SIMPLE_FGEN
    """.strip()

    # Parse and write the code
    routine = Subroutine.from_source(fcode, frontend=frontend)
    source = fgen(routine, linewidth=90)
    assert source == fcode
