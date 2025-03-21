# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module
from loki.backend import fgen, FortranStyle
from loki.frontend import available_frontends, OMNI


@pytest.mark.parametrize('frontend', available_frontends(
    skip=[(OMNI, "OMNI enforces its own stylistic quirks!")]
))
def test_fgen_default_style(frontend, tmp_path):
    """ Test the default and IFS-specific Fortran styles of the fgen backend """

    fcode = """
MODULE ONCE_UPON
IMPLICIT NONE

INTEGER, PARAMETER :: ATIME = 8

CONTAINS

  SUBROUTINE THERE_WERE ( N, M, RICK, DAVE, NEVER )
    INTEGER(KIND=4), INTENT(IN) :: N, M
    REAL(KIND=ATIME), INTENT(INOUT) :: RICK, DAVE(N)
    REAL(KIND=ATIME), INTENT(OUT) :: NEVER(N, M)
    INTEGER :: I, J, IJK

    DO I=1, N
      DAVE(I) = DAVE(I) + RICK
    END DO

    IF (RICK > 0.5) THEN
      DO I=1, N
        DO J=1, M
          NEVER(I, J) = RICK + DAVE(I)
        END DO
      ENDDO
    END IF
  END SUBROUTINE
END MODULE
"""
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Test the default Fortran layout
    generated_default = fgen(module)
    assert generated_default == """
MODULE ONCE_UPON
  IMPLICIT NONE

  INTEGER, PARAMETER :: ATIME = 8

  CONTAINS

  SUBROUTINE THERE_WERE (N, M, RICK, DAVE, NEVER)
    INTEGER(KIND=4), INTENT(IN) :: N, M
    REAL(KIND=ATIME), INTENT(INOUT) :: RICK, DAVE(N)
    REAL(KIND=ATIME), INTENT(OUT) :: NEVER(N, M)
    INTEGER :: I, J, IJK

    DO I=1,N
      DAVE(I) = DAVE(I) + RICK
    END DO

    IF (RICK > 0.5) THEN
      DO I=1,N
        DO J=1,M
          NEVER(I, J) = RICK + DAVE(I)
        END DO
      END DO
    END IF
  END SUBROUTINE THERE_WERE
END MODULE ONCE_UPON
""".strip()

    # Test a custom Fortran layout
    custom_style = FortranStyle(
        conditional_indent=4,
        conditional_end_space=False,
        loop_indent=3,
        loop_end_space=False,
        procedure_spec_indent=5,
        procedure_body_indent=1,
        # procedure_contains_indent=2,
        procedure_end_named=False,
        module_spec_indent=3,
        module_contains_indent=1,
        module_end_named=False,
    )
    generated_custom = fgen(module, style=custom_style)
    assert generated_custom == """
MODULE ONCE_UPON
   IMPLICIT NONE

   INTEGER, PARAMETER :: ATIME = 8

 CONTAINS

 SUBROUTINE THERE_WERE (N, M, RICK, DAVE, NEVER)
      INTEGER(KIND=4), INTENT(IN) :: N, M
      REAL(KIND=ATIME), INTENT(INOUT) :: RICK, DAVE(N)
      REAL(KIND=ATIME), INTENT(OUT) :: NEVER(N, M)
      INTEGER :: I, J, IJK

  DO I=1,N
     DAVE(I) = DAVE(I) + RICK
  ENDDO

  IF (RICK > 0.5) THEN
      DO I=1,N
         DO J=1,M
            NEVER(I, J) = RICK + DAVE(I)
         ENDDO
      ENDDO
  ENDIF
 END SUBROUTINE
END MODULE
""".strip()
