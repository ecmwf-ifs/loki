# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest

from loki.frontend import available_frontends, OMNI
from loki import Sourcefile, FindNodes, CallStatement, fgen, Conditional

from transformations.parallel_routine_dispatch import ParallelRoutineDispatchTransformation


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_dr_hook(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 3

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    calls = [call for call in FindNodes(CallStatement).visit(routine.body) if call.name.name=='DR_HOOK']
    assert len(calls) == 8

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_decl_local_arrays(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])
    var_lst=["YL_ZRDG_CVGQ", "ZRDG_CVGQ", "YL_ZRDG_MU0LU", "ZRDG_MU0LU", "YL_ZRDG_MU0M", "ZRDG_MU0M", "YL_ZRDG_MU0N", "ZRDG_MU0N", "YL_ZRDG_MU0", "ZRDG_MU0"]
    dcls = [dcl for dcl in routine.declarations if dcl.symbols[0].name in var_lst]
    str_dcls = ""
    for dcl in dcls:
        str_dcls += fgen(dcl)+"\n"
    assert str_dcls == """CLASS(FIELD_3RB), POINTER :: YL_ZRDG_CVGQ => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_CVGQ(:, :, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0LU => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0LU(:, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0M => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0M(:, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0N => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0N(:, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0 => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0(:, :) => NULL()
"""

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_decl_field_create_delete(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    var_lst = ["YL_ZRDG_CVGQ", "ZRDG_CVGQ", "YL_ZRDG_MU0LU", "ZRDG_MU0LU", "YL_ZRDG_MU0M", "ZRDG_MU0M", "YL_ZRDG_MU0N", "ZRDG_MU0N", "YL_ZRDG_MU0", "ZRDG_MU0"]
    field_create = ["CALL FIELD_NEW(YL_ZRDG_CVGQ, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KFLEVG, YDCPG_OPTS%KGPBLKS /), LBOUNDS=(/ 0, 1 /),  &\n& PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0N, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KGPBLKS /), PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0LU, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KGPBLKS /), PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KGPBLKS /), PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0M, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KGPBLKS /), PERSISTENT=.true.)"
                ]

    calls = [call for call in FindNodes(CallStatement).visit(routine.body) if call.name.name=="FIELD_NEW"]
    assert len(calls) == 5
    for call in calls:
        assert fgen(call) in field_create
    
    field_delete = ["IF (ASSOCIATED(YL_ZRDG_CVGQ)) CALL FIELD_DELETE(YL_ZRDG_CVGQ)",
                "IF (ASSOCIATED(YL_ZRDG_MU0LU)) CALL FIELD_DELETE(YL_ZRDG_MU0LU)",
                "IF (ASSOCIATED(YL_ZRDG_MU0M)) CALL FIELD_DELETE(YL_ZRDG_MU0M)",
                "IF (ASSOCIATED(YL_ZRDG_MU0)) CALL FIELD_DELETE(YL_ZRDG_MU0)",
                "IF (ASSOCIATED(YL_ZRDG_MU0N)) CALL FIELD_DELETE(YL_ZRDG_MU0N)"
                ]

    conds = [cond for cond in FindNodes(Conditional).visit(routine.body)]
    conditional = []
    for cond in conds:
        for call in FindNodes(CallStatement).visit(cond):
                if call.name.name=="FIELD_DELETE":
                    conditional.append(cond)

    assert len(conditional) == 5
    for cond in conditional:
        assert fgen(cond) in field_delete
    breakpoint()