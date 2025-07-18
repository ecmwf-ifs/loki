# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Module, Sourcefile, config_override
from loki.backend import fgen
from loki.ir import nodes as ir, FindNodes, SubstituteExpressions
from loki.frontend import FP, OMNI, REGEX, SourceStatus, available_frontends


def test_fgen_conservative_routine():
    fcode = """
SUBROUTINE MY_TEST_ROUTINE( N,DAVE)
  USE MY_MOD, ONLY : AKIND, RTYPE
  IMPLICIT NONE
  INTEGER,          INTENT(IN) :: N
  tyPe(RTYPE)                 :: RICK ! CAN'T MAKE ARGUMENT YET!
  REAL(KIND=AKIND), INTENT(INOUT) :: DAVE(N)
  REAL(KIND=AKIND) :: TMP
  INTEGER :: I

  DO I=1, N
    IF (  DAVE(I)    > 0.5) THEN
      ! Who is DAVE = ...
      TMP = RICK%A
      DAvE( I)   = RICK%A

         ! BUT ALSO...
         RICK%B = DaVe(  i)
       ELSE
          ! ... AND ...
            DaVE( I ) = 66.6
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


def test_fgen_conservative_module():
    fcode_type = """
MODULE MY_MOD
  INTEGER, PARAMETER :: AKIND = 8
  TYPE RTYPE
    real(kind = 8) ::   DaVe
  END TYPE RTYPE
END MODULE   MY_MOD
"""

    fcode = """
MODULE MY_TEST_MOD
  USE MY_MOD, ONLY: AKIND, RTYPE
  IMPLICIT NONE

  REAL(KIND=AKIND) ::    DaVE(  5 )

  CONTAINS

  SUBROUTINE A_SHORT_ROUTINE( N,DAVE)
    INTEGER,          INTENT(IN) :: N
    REAL(KIND=AKIND), INTENT(INOUT) :: DAVE(N)

    DaVE(:) = DaVE(:) + 2.0
  END SUBROUTINE   A_SHORT_ROUTINE
END MODULE MY_TEST_MOD
"""
    with config_override({'frontend-store-source': True}):
        type_mod = Module.from_source(fcode_type, frontend=FP)
        module = Module.from_source(fcode, frontend=FP)

    routine = module['a_short_routine']

    # Check modules can be re-created string-identically
    s_type_mod = fgen(type_mod, conservative=True)
    assert fcode_type.strip() == s_type_mod.strip()
    s_module = fgen(module, conservative=True)
    assert fcode.strip() == s_module.strip()

    # Type Module: Use `SubstituteExpressions` to replace AKIND with BKIND
    akind = FindNodes(ir.VariableDeclaration).visit(type_mod.spec)[0].symbols[0]
    assert akind == 'AKIND'
    sub_expr = SubstituteExpressions(
        {akind: akind.clone(name='BKIND')}, invalidate_source=True
    )
    type_mod.spec = sub_expr.visit(type_mod.spec)
    type_mod.source.status = SourceStatus.INVALID_CHILDREN
    assert type_mod.spec.source.status == SourceStatus.INVALID_CHILDREN

    # Type Module: Check that substitutions have invalidated relevant nodes
    decls = FindNodes(ir.VariableDeclaration).visit(type_mod.spec)
    assert 'bkind' in decls[0].symbols and not decls[0].source.status == SourceStatus.VALID

    # Type Module: Check the actual output formatting of type module
    type_mod_expected = fcode_type.replace('AKIND', 'BKIND').strip()
    type_mod_generated = fgen(type_mod, conservative=True).strip()
    assert type_mod_generated == type_mod_expected

    # Main Module: Use `SubstituteExpressions` to replace AKIND with BKIND in main module
    akind = FindNodes(ir.Import).visit(module.spec)[0].symbols[0]
    assert akind == 'AKIND'
    sub_expr = SubstituteExpressions(
        {akind: akind.clone(name='BKIND')}, invalidate_source=True
    )
    module.spec = sub_expr.visit(module.spec)
    module.source.status = SourceStatus.INVALID_CHILDREN
    assert module.spec.source.status == SourceStatus.INVALID_CHILDREN
    # TODO: Change routine directly, as Transformer does not recurse into program unit yet
    routine.spec = sub_expr.visit(module['a_short_routine'].spec)
    routine.source.status = SourceStatus.INVALID_CHILDREN
    assert routine.spec.source.status == SourceStatus.INVALID_CHILDREN

    module.contains.source.status = SourceStatus.INVALID_CHILDREN

    # Main Module: Check that substitutions have invalidated relevant nodes
    decls_m = FindNodes(ir.VariableDeclaration).visit(module.spec)
    assert len(decls_m) == 1
    assert decls_m[0].symbols[0] == 'dave(5)'
    assert decls_m[0].symbols[0].type.kind == 'bkind'
    assert decls_m[0].source.status == SourceStatus.INVALID_NODE

    # Main Module: Check the actual output formatting
    module_expected = fcode.replace('AKIND', 'BKIND').strip()
    module_generated = fgen(module, conservative=True).strip()
    assert module_generated == module_expected


@pytest.mark.parametrize('frontend', available_frontends(
    include_regex=True, skip=[(OMNI, 'OMNI is not string-conservative')]
))
def test_fgen_conservative_sourcefile(frontend):
    """ Test outer program unit conservation via `ir.Section` and REGEX frontend """

    fcode = """
subroutine some_routine
implicit none
end subroutine some_routine

subroutine OTHER_ROUTINE
implicit none
call some_routine
end subroutine OTHER_ROUTINE
"""
    with config_override({'frontend-store-source': True}):
        sourcefile = Sourcefile.from_source(fcode, frontend=frontend)

    assert sourcefile.source
    assert sourcefile.ir.source

    routines = sourcefile.routines
    assert len(routines) == 2
    assert routines[0].source
    if frontend == REGEX:
        sourcefile.routines[0].make_complete()
        sourcefile.routines[1].make_complete()
    assert routines[0].spec.source
    assert routines[0].body.source
    assert routines[1].source
    assert routines[1].spec.source
    assert routines[1].body.source

    # Modify the subroutine objects only
    routines[0].name = routines[0].name.upper()
    routines[0].source.status = SourceStatus.INVALID_NODE

    routines[1].name = routines[1].name.lower()
    routines[1].source.status = SourceStatus.INVALID_NODE

    # Ensure only header/footer are changed
    assert routines[0].to_fortran(conservative=True) == """
SUBROUTINE SOME_ROUTINE ()
implicit none

END SUBROUTINE SOME_ROUTINE
""".strip()

    assert routines[1].to_fortran(conservative=True) == """
SUBROUTINE other_routine ()
implicit none
call some_routine
END SUBROUTINE other_routine
""".strip()


def test_fgen_conservative_rebuild():
    """ Test that triggers a near complete re-build """

    fcode = """
MODULE MY_TEST_MOD
  use type_mod, only: akind, ikind, rtype
  ! use func_mod, only: my_func
  implicit none

  REAL(KIND=AKIND) ::    DaVE(  5 )

  CONTAINS

  SUBROUTINE A_SHORT_ROUTINE( N, DAVE)
    INTEGER,          INTENT(IN) :: N
    REAL(KIND=AKIND), INTENT(INOUT) :: DAVE(N)
    integer( kind =ikind) :: i

    DaVE( : ) = DaVE(:) + 2.0
    do    i=1, n
      if   (  DaVe(i) ==    0.0)  then
        dave (i) = 3.0
      end  if
    enddo

    CALL My_Func(n, daVE(1:n))
  END SUBROUTINE   A_SHORT_ROUTINE
END MODULE MY_TEST_MOD
"""
    with config_override({'frontend-store-source': True}):
        module = Module.from_source(fcode, frontend=FP)
    routine = module['a_short_routine']

    # Change nearly every line to trigger full re-build
    smap = module.imported_symbol_map
    vmap = module.variable_map
    subs_module = SubstituteExpressions(
        {
            smap['ikind']: smap['ikind'].clone(name='ikinder'),
            smap['akind']: smap['akind'].clone(name='akinder'),
            vmap['dave'].symbol: vmap['dave'].symbol.clone(name='rick')
        }, invalidate_source=True
    )
    module.spec = subs_module.visit(module.spec)
    module.contains = subs_module.visit(module.contains)
    assert module.spec.source.status == SourceStatus.INVALID_CHILDREN
    assert module.contains.source.status == SourceStatus.INVALID_CHILDREN
    module.name = 'A_NEW_MOD'
    module.source.status = SourceStatus.INVALID_NODE

    vmap = routine.variable_map
    subs_routine = SubstituteExpressions(
        {
            vmap['dave'].symbol: vmap['dave'].symbol.clone(name='rick'),
            vmap['n']: vmap['n'].clone(name='m'),
            vmap['i']: vmap['i'].clone(name='j'),
        }, invalidate_source=True
    )
    routine.spec = subs_routine.visit(routine.spec)
    routine.body = subs_routine.visit(routine.body)
    assert routine.spec.source.status == SourceStatus.INVALID_CHILDREN
    assert routine.body.source.status == SourceStatus.INVALID_CHILDREN
    routine.name = 'A_CHANGED_ROUTINE'
    routine.source.status = SourceStatus.INVALID_NODE

    # Check that changes have indeed invalidated nodes
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].lhs == 'rick(:)' and assigns[0].rhs == 'rick(:) + 2.0'
    assert assigns[0].source.status == SourceStatus.INVALID_NODE
    assert assigns[1].lhs == 'rick(j)' and assigns[1].rhs == '3.0'
    assert assigns[1].source.status == SourceStatus.INVALID_NODE

    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1
    assert loops[0].variable == 'j' and loops[0].bounds == '1:m'
    assert loops[0].source.status == SourceStatus.INVALID_NODE

    conds = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds) == 1
    assert conds[0].condition == 'rick(j) == 0.0'
    assert conds[0].source.status == SourceStatus.INVALID_NODE

    # Check that fully generated and conservative agree
    routine_expected = fgen(module, conservative=False).strip()
    routine_generated = fgen(module, conservative=True).strip()
    assert routine_generated == routine_expected
