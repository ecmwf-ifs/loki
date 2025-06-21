# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import Subroutine, config_override
from loki.backend import fgen
from loki.ir import nodes as ir, FindNodes, SubstituteExpressions
from loki.frontend import FP, SourceStatus


def test_fgen_conservative_routine():
    fcode = """
SUBROUTINE MY_TEST_ROUTINE( N,DAVE)
  USE MY_MOD, ONLY : AKIND, RTYPE
  IMPLICIT NONE
  INTEGER,          INTENT(IN) :: N
  TYPE(RTYPE)                 :: RICK ! CAN'T MAKE ARGUMENT YET!
  REAL(KIND=AKIND), INTENT(INOUT) :: DAVE(N)
  REAL(KIND=AKIND) :: TMP
  INTEGER :: I

  DO I=1, N
    IF (  DAVE(I)    > 0.5) THEN
      ! Who is DAVE = ...
      TMP = RICK%A
      DAVE(I)   =   DAVE( I   )+TMP

         ! BUT ALSO...
         RICK%B = 42.0
       ELSE
          ! ... AND ...
            DAVE(I) = 66.6
    END IF

      ! BECAUSE DAVE WILL ...
      CALL  NEVER_GONNA ( DAVE%YOU_UP   )
  END DO
END SUBROUTINE   MY_TEST_ROUTINE
"""
    with config_override({'frontend-store-source': True}):
        routine = Subroutine.from_source(fcode, frontend=FP)

    # Check the untouched output of a few noes
    s_routine = fgen(routine, conservative=True)
    assert fcode.strip() == s_routine.strip()

    str_spec = fgen(routine.spec, conservative=True)
    exp_spec = '\n'.join(fcode.splitlines()[2:9])
    assert exp_spec.strip() == str_spec.strip()

    str_body = fgen(routine.body, conservative=True)
    exp_body = '\n'.join(fcode.splitlines()[9:26])
    assert exp_body.strip() == str_body.strip()

    str_loop = fgen(FindNodes(ir.Loop).visit(routine.body), conservative=True)
    exp_loop = '\n'.join(fcode.splitlines()[10:26])
    assert exp_loop == str_loop

    # Use `SubstituteExpressions` to replace RICk with BOB, including in spec!
    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    rick = routine.variable_map['RICK']
    bob = routine.Variable(name='BOB', type=rick.type)  # This replicates type info!
    sub_rick = SubstituteExpressions({rick: bob}, invalidate_source=True)

    routine.spec = sub_rick.visit(routine.spec)
    routine.body = sub_rick.visit(routine.body)

    routine.source.status = SourceStatus.INVALID_CHILDREN
    assert routine.spec.source.status == SourceStatus.INVALID_CHILDREN
    assert routine.body.source.status == SourceStatus.INVALID_CHILDREN

    decls = FindNodes(ir.VariableDeclaration).visit(routine.spec)
    assert 'bob' in decls[1].symbols and not decls[1].source.status == SourceStatus.VALID
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert 'bob%a' == assigns[0].rhs and not assigns[0].source.status == SourceStatus.VALID
    assert 'bob%b' == assigns[2].lhs and not assigns[2].source.status == SourceStatus.VALID

    # And now check the actual output formatting...
    expected = fcode.replace('RICK', 'BOB').strip()
    generated = fgen(routine, conservative=True).strip()
    assert generated == expected
