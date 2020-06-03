"""
Test identity of source-to-source translation.

The tests in here do not verify correct representation internally,
they only check whether at the end comes out what went in at the beginning.

"""
from pathlib import Path
import pytest
from loki import SourceFile, Subroutine, OFP, OMNI, FP, fgen, read_file


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_subroutine_conservative(here, frontend):
    fcode = """
SUBROUTINE CONSERVATIVE (X, Y, SCALAR, VECTOR, MATRIX)
    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)
    INTEGER, INTENT(IN) :: X, Y
    REAL(KIND=JPRB), INTENT(IN) :: SCALAR
    REAL(KIND=JPRB), INTENT(INOUT) :: VECTOR(X)
    REAL(KIND=JPRB), DIMENSION(X, Y), INTENT(OUT) :: MATRIX
    INTEGER :: I

    DO I=1, X
        VECTOR(I) = VECTOR(I) + SCALAR
        MATRIX(I, :) = I * VECTOR(I)
    ENDDO
END SUBROUTINE CONSERVATIVE
    """.strip()

    # Parse and write the code
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = here/('source_id_subroutine_conservative_%s.f90' % frontend)
    SourceFile(filepath).write(fgen(routine, conservative=True))

    # Check the result
    source = read_file(filepath)
    assert source == fcode
