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

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_derived_dcl(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    dcls = [fgen(dcl) for dcl in routine.spec.body[-13:-1]]
    
    test_dcls=["REAL(KIND=JPRB), POINTER :: Z_YDVARS_U_T0(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_Q_DM(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_GEOMETRY_GELAM_T0(:, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_CVGQ_T0(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_Q_DL(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_V_T0(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_GEOMETRY_GEMU_T0(:, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_Q_T0(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDCPG_PHY0_XYB_RDELP(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_CVGQ_DM(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDCPG_DYN0_CTY_EVEL(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_CVGQ_DL(:, :, :)"]
    for dcl in dcls:
        assert dcl in test_dcls

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_derived_var(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    
    test_map = {
        "YDVARS%GEOMETRY%GEMU%T0" : ["YDVARS%GEOMETRY%GEMU%FT0", "Z_YDVARS_GEOMETRY_GEMU_T0"],
        "YDVARS%GEOMETRY%GELAM%T0" : ["YDVARS%GEOMETRY%GELAM%FT0", "Z_YDVARS_GEOMETRY_GELAM_T0"],
        "YDVARS%U%T0" : ["YDVARS%U%FT0", "Z_YDVARS_U_T0"],
        "YDVARS%V%T0" : ["YDVARS%V%FT0", "Z_YDVARS_V_T0"],
        "YDVARS%Q%T0" : ["YDVARS%Q%FT0", "Z_YDVARS_Q_T0"],
        "YDVARS%Q%DM" : ["YDVARS%Q%FDM", "Z_YDVARS_Q_DM"],
        "YDVARS%Q%DL" : ["YDVARS%Q%FDL", "Z_YDVARS_Q_DL"],
        "YDVARS%CVGQ%T0" : ["YDVARS%CVGQ%FT0", "Z_YDVARS_CVGQ_T0"],
        "YDVARS%CVGQ%DM" : ["YDVARS%CVGQ%FDM", "Z_YDVARS_CVGQ_DM"],
        "YDVARS%CVGQ%DL" : ["YDVARS%CVGQ%FDL", "Z_YDVARS_CVGQ_DL"],
        "YDCPG_PHY0%XYB%RDELP" : ["YDCPG_PHY0%XYB%F_RDELP", "Z_YDCPG_PHY0_XYB_RDELP"],
        "YDCPG_DYN0%CTY%EVEL" : ["YDCPG_DYN0%CTY%F_EVEL", "Z_YDCPG_DYN0_CTY_EVEL"], 
        "YDMF_PHYS_SURF%GSD_VF%PZ0F" : ["YDMF_PHYS_SURF%GSD_VF%F_Z0F", "Z_YDMF_PHYS_SURF_GSD_VF_PZ0F"]
    }
    for var_name in transformation.routine_map_derived:
        value = transformation.routine_map_derived[var_name]
        field_ptr = value[0]
        ptr = value[1]

        assert test_map[var_name][0] == field_ptr.name
        assert test_map[var_name][1] == ptr.name

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_get_data(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    get_data = transformation.get_data
    
    test_get_data = {}
#    test_get_data["OpenMP"] = """
#IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)
#ZRDG_CVGQ => GET_HOST_DATA_RDWR(YL_ZRDG_CVGQ)
#ZRDG_MU0 => GET_HOST_DATA_RDWR(YL_ZRDG_MU0)
#ZRDG_MU0LU => GET_HOST_DATA_RDWR(YL_ZRDG_MU0LU)
#ZRDG_MU0M => GET_HOST_DATA_RDWR(YL_ZRDG_MU0M)
#ZRDG_MU0N => GET_HOST_DATA_RDWR(YL_ZRDG_MU0N)
#Z_YDCPG_DYN0_CTY_EVEL => GET_HOST_DATA_RDONLY(YDCPG_DYN0%CTY%F_EVEL)
#Z_YDCPG_PHY0_XYB_RDELP => GET_HOST_DATA_RDONLY(YDCPG_PHY0%XYB%F_RDELP)
#Z_YDVARS_CVGQ_DL => GET_HOST_DATA_RDONLY(YDVARS%CVGQ%FDL)
#Z_YDVARS_CVGQ_DM => GET_HOST_DATA_RDONLY(YDVARS%CVGQ%FDM)
#Z_YDVARS_CVGQ_T0 => GET_HOST_DATA_RDONLY(YDVARS%CVGQ%FT0)
#Z_YDVARS_GEOMETRY_GELAM_T0 => GET_HOST_DATA_RDONLY(YDVARS%GEOMETRY%GELAM%FT0)
#Z_YDVARS_GEOMETRY_GEMU_T0 => GET_HOST_DATA_RDONLY(YDVARS%GEOMETRY%GEMU%FT0)
#Z_YDVARS_Q_DL => GET_HOST_DATA_RDONLY(YDVARS%Q%FDL)
#Z_YDVARS_Q_DM => GET_HOST_DATA_RDONLY(YDVARS%Q%FDM)
#Z_YDVARS_Q_T0 => GET_HOST_DATA_RDONLY(YDVARS%Q%FT0)
#Z_YDVARS_U_T0 => GET_HOST_DATA_RDONLY(YDVARS%U%FT0)
#Z_YDVARS_V_T0 => GET_HOST_DATA_RDONLY(YDVARS%V%FT0)
#Z_YDMF_PHYS_SURF_GSD_VF_PZ0F => GET_HOST_DATA_RDONLY(YDMF_PHYS_SURF%GSD_VF%F_Z0F)
#IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
#"""
    test_get_data["OpenMP"] = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)
ZRDG_CVGQ => GET_HOST_DATA_RDWR(YL_ZRDG_CVGQ)
ZRDG_MU0 => GET_HOST_DATA_RDWR(YL_ZRDG_MU0)
ZRDG_MU0LU => GET_HOST_DATA_RDWR(YL_ZRDG_MU0LU)
ZRDG_MU0M => GET_HOST_DATA_RDWR(YL_ZRDG_MU0M)
ZRDG_MU0N => GET_HOST_DATA_RDWR(YL_ZRDG_MU0N)
Z_YDCPG_DYN0_CTY_EVEL => GET_HOST_DATA_RDWR(YDCPG_DYN0%CTY%F_EVEL)
Z_YDCPG_PHY0_XYB_RDELP => GET_HOST_DATA_RDWR(YDCPG_PHY0%XYB%F_RDELP)
Z_YDVARS_CVGQ_DL => GET_HOST_DATA_RDWR(YDVARS%CVGQ%FDL)
Z_YDVARS_CVGQ_DM => GET_HOST_DATA_RDWR(YDVARS%CVGQ%FDM)
Z_YDVARS_CVGQ_T0 => GET_HOST_DATA_RDWR(YDVARS%CVGQ%FT0)
Z_YDVARS_GEOMETRY_GELAM_T0 => GET_HOST_DATA_RDWR(YDVARS%GEOMETRY%GELAM%FT0)
Z_YDVARS_GEOMETRY_GEMU_T0 => GET_HOST_DATA_RDWR(YDVARS%GEOMETRY%GEMU%FT0)
Z_YDVARS_Q_DL => GET_HOST_DATA_RDWR(YDVARS%Q%FDL)
Z_YDVARS_Q_DM => GET_HOST_DATA_RDWR(YDVARS%Q%FDM)
Z_YDVARS_Q_T0 => GET_HOST_DATA_RDWR(YDVARS%Q%FT0)
Z_YDVARS_U_T0 => GET_HOST_DATA_RDWR(YDVARS%U%FT0)
Z_YDVARS_V_T0 => GET_HOST_DATA_RDWR(YDVARS%V%FT0)
Z_YDMF_PHYS_SURF_GSD_VF_PZ0F => GET_HOST_DATA_RDWR(YDMF_PHYS_SURF%GSD_VF%F_Z0F)
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
"""
    test_get_data["OpenMPSingleColumn"] = test_get_data["OpenMP"]

#    test_get_data["OpenACCSingleColumn"] = """
#IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)                                                                       
#ZRDG_CVGQ => GET_DEVICE_DATA_RDWR(YL_ZRDG_CVGQ)                                                                                                                
#ZRDG_MU0 => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0)                                                                                                                  
#ZRDG_MU0LU => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0LU)                                                                                                              
#ZRDG_MU0M => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0M)                                                                                                                
#ZRDG_MU0N => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0N)                                                                                                                
#Z_YDCPG_DYN0_CTY_EVEL => GET_DEVICE_DATA_RDONLY(YDCPG_DYN0%CTY%F_EVEL)                                                                                         
#Z_YDCPG_PHY0_XYB_RDELP => GET_DEVICE_DATA_RDONLY(YDCPG_PHY0%XYB%F_RDELP)                                                                                       
#Z_YDVARS_CVGQ_DL => GET_DEVICE_DATA_RDONLY(YDVARS%CVGQ%FDL)                                                                                                    
#Z_YDVARS_CVGQ_DM => GET_DEVICE_DATA_RDONLY(YDVARS%CVGQ%FDM)                                                                                                    
#Z_YDVARS_CVGQ_T0 => GET_DEVICE_DATA_RDONLY(YDVARS%CVGQ%FT0)                                                                                                    
#Z_YDVARS_GEOMETRY_GELAM_T0 => GET_DEVICE_DATA_RDONLY(YDVARS%GEOMETRY%GELAM%FT0)                                                                                
#Z_YDVARS_GEOMETRY_GEMU_T0 => GET_DEVICE_DATA_RDONLY(YDVARS%GEOMETRY%GEMU%FT0)                                                                                  
#Z_YDVARS_Q_DL => GET_DEVICE_DATA_RDONLY(YDVARS%Q%FDL)                                                                                                          
#Z_YDVARS_Q_DM => GET_DEVICE_DATA_RDONLY(YDVARS%Q%FDM)                                                                                                          
#Z_YDVARS_Q_T0 => GET_DEVICE_DATA_RDONLY(YDVARS%Q%FT0)                                                                                                          
#Z_YDVARS_U_T0 => GET_DEVICE_DATA_RDONLY(YDVARS%U%FT0)                                                                                                          
#Z_YDVARS_V_T0 => GET_DEVICE_DATA_RDONLY (YDVARS%V%FT0)                                                                                                          
#Z_YDMF_PHYS_SURF_GSD_VF_PZ0F => GET_DEVICE_DATA_RDONLY(YDMF_PHYS_SURF%GSD_VF%F_Z0F)                                                                          |276 REAL(KIND=JPRB)     :: ZPFL_FPLSH (YDCPG_OPTS%KLON, 0:YDCPG_OPTS%KFLEVG)                                                                                           
#IF(LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
#"""

    test_get_data["OpenACCSingleColumn"] = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)                                                                       
ZRDG_CVGQ => GET_DEVICE_DATA_RDWR(YL_ZRDG_CVGQ)                                                                                                                
ZRDG_MU0 => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0)                                                                                                                  
ZRDG_MU0LU => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0LU)                                                                                                              
ZRDG_MU0M => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0M)                                                                                                                
ZRDG_MU0N => GET_DEVICE_DATA_RDWR(YL_ZRDG_MU0N)                                                                                                                
Z_YDCPG_DYN0_CTY_EVEL => GET_DEVICE_DATA_RDWR(YDCPG_DYN0%CTY%F_EVEL)                                                                                         
Z_YDCPG_PHY0_XYB_RDELP => GET_DEVICE_DATA_RDWR(YDCPG_PHY0%XYB%F_RDELP)                                                                                       
Z_YDVARS_CVGQ_DL => GET_DEVICE_DATA_RDWR(YDVARS%CVGQ%FDL)                                                                                                    
Z_YDVARS_CVGQ_DM => GET_DEVICE_DATA_RDWR(YDVARS%CVGQ%FDM)                                                                                                    
Z_YDVARS_CVGQ_T0 => GET_DEVICE_DATA_RDWR(YDVARS%CVGQ%FT0)                                                                                                    
Z_YDVARS_GEOMETRY_GELAM_T0 => GET_DEVICE_DATA_RDWR(YDVARS%GEOMETRY%GELAM%FT0)                                                                                
Z_YDVARS_GEOMETRY_GEMU_T0 => GET_DEVICE_DATA_RDWR(YDVARS%GEOMETRY%GEMU%FT0)                                                                                  
Z_YDVARS_Q_DL => GET_DEVICE_DATA_RDWR(YDVARS%Q%FDL)                                                                                                          
Z_YDVARS_Q_DM => GET_DEVICE_DATA_RDWR(YDVARS%Q%FDM)                                                                                                          
Z_YDVARS_Q_T0 => GET_DEVICE_DATA_RDWR(YDVARS%Q%FT0)                                                                                                          
Z_YDVARS_U_T0 => GET_DEVICE_DATA_RDWR(YDVARS%U%FT0)                                                                                                          
Z_YDVARS_V_T0 => GET_DEVICE_DATA_RDWR(YDVARS%V%FT0)                                                                                                          
Z_YDMF_PHYS_SURF_GSD_VF_PZ0F => GET_DEVICE_DATA_RDWR(YDMF_PHYS_SURF%GSD_VF%F_Z0F)                                                                          |276 REAL(KIND=JPRB)     :: ZPFL_FPLSH (YDCPG_OPTS%KLON, 0:YDCPG_OPTS%KFLEVG)                                                                                           
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
"""

    for target in get_data:
        for node in get_data[target]:
            assert fgen(node) in test_get_data[target]

###@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
###def test_parallel_routine_dispatch_synchost(here, frontend):
###
###    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
###    routine = source['dispatch_routine']
###
###    transformation = ParallelRoutineDispatchTransformation()
###    transformation.apply(source['dispatch_routine'])
###
###    get_data = transformation.get_data
###    
###    test_get_data = """
###IF (LHOOK) CALL DR_HOOK ('DISPATCH_ROUTINE:CPPHINP:GET_DATA',0,ZHOOK_HANDLE_FIELD_API)
###Z_YDMF_PHYS_SURF_GSD_VV_PZ0H => GET_HOST_DATA_RDWR (YDMF_PHYS_SURF%GSD_VV%F_Z0H)
###IF (LHOOK) CALL DR_HOOK ('DISPATCH_ROUTINE:CPPHINP:GET_DATA',1,ZHOOK_HANDLE_FIELD_API)
###"""
###
###    for node in get_data:
###        assert fgen(node) in test_get_data
###

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_synchost(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    synchost = transformation.synchost[0]
    
    test_synchost = """IF (LSYNCHOST('DISPATCH_ROUTINE:CPPHINP')) THEN
 IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:SYNCHOST', 0, ZHOOK_HANDLE_FIELD_API)
 ZRDG_CVGQ => GET_HOST_DATA_RDWR(YL_ZRDG_CVGQ)
 ZRDG_MU0 => GET_HOST_DATA_RDWR(YL_ZRDG_MU0)
 ZRDG_MU0LU => GET_HOST_DATA_RDWR(YL_ZRDG_MU0LU)
 ZRDG_MU0M => GET_HOST_DATA_RDWR(YL_ZRDG_MU0M)
 ZRDG_MU0N => GET_HOST_DATA_RDWR(YL_ZRDG_MU0N)
 Z_YDCPG_DYN0_CTY_EVEL => GET_HOST_DATA_RDWR(YDCPG_DYN0%CTY%F_EVEL)
 Z_YDCPG_PHY0_XYB_RDELP => GET_HOST_DATA_RDWR(YDCPG_PHY0%XYB%F_RDELP)
 Z_YDVARS_CVGQ_DL => GET_HOST_DATA_RDWR(YDVARS%CVGQ%FDL)
 Z_YDVARS_CVGQ_DM => GET_HOST_DATA_RDWR(YDVARS%CVGQ%FDM)
 Z_YDVARS_CVGQ_T0 => GET_HOST_DATA_RDWR(YDVARS%CVGQ%FT0)
 Z_YDVARS_GEOMETRY_GELAM_T0 => GET_HOST_DATA_RDWR(YDVARS%GEOMETRY%GELAM%FT0)
 Z_YDVARS_GEOMETRY_GEMU_T0 => GET_HOST_DATA_RDWR(YDVARS%GEOMETRY%GEMU%FT0)
 Z_YDVARS_Q_DL => GET_HOST_DATA_RDWR(YDVARS%Q%FDL)
 Z_YDVARS_Q_DM => GET_HOST_DATA_RDWR(YDVARS%Q%FDM)
 Z_YDVARS_Q_T0 => GET_HOST_DATA_RDWR(YDVARS%Q%FT0)
 Z_YDVARS_U_T0 => GET_HOST_DATA_RDWR(YDVARS%U%FT0)
 Z_YDVARS_V_T0 => GET_HOST_DATA_RDWR(YDVARS%V%FT0)
 Z_YDMF_PHYS_SURF_GSD_VF_PZ0F => GET_HOST_DATA_RDWR(YDMF_PHYS_SURF%GSD_VF%F_Z0F)
 IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:SYNCHOST', 1, ZHOOK_HANDLE_FIELD_API)
 ENDIF
"""
    assert fgen(synchost.condition) in test_synchost
    for node in synchost.body:
        assert fgen(node) in test_synchost

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_nullify(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    nullify = transformation.nullify
 
    test_nullify = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:NULLIFY', 0, ZHOOK_HANDLE_FIELD_API)
ZRDG_CVGQ => NULL()
ZRDG_MU0 => NULL()
ZRDG_MU0LU => NULL()
ZRDG_MU0M => NULL()
ZRDG_MU0N => NULL()
Z_YDCPG_DYN0_CTY_EVEL => NULL()
Z_YDCPG_PHY0_XYB_RDELP => NULL()
Z_YDVARS_CVGQ_DL => NULL()
Z_YDVARS_CVGQ_DM => NULL()
Z_YDVARS_CVGQ_T0 => NULL()
Z_YDVARS_GEOMETRY_GELAM_T0 => NULL()
Z_YDVARS_GEOMETRY_GEMU_T0 => NULL()
Z_YDVARS_Q_DL => NULL()
Z_YDVARS_Q_DM => NULL()
Z_YDVARS_Q_T0 => NULL()
Z_YDVARS_U_T0 => NULL()
Z_YDVARS_V_T0 => NULL()
Z_YDMF_PHYS_SURF_GSD_VF_PZ0F => NULL() 
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:NULLIFY', 1, ZHOOK_HANDLE_FIELD_API)
"""

    for node in nullify:
        assert fgen(node) in test_nullify


@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_compute_openmp(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    map_compute = transformation.compute
    compute_openmp = map_compute['OpenMP']
 
    test_compute= """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:COMPUTE', 0, ZHOOK_HANDLE_COMPUTE)

CALL YLCPG_BNDS%INIT(YDCPG_OPTS)
!$OMP PARALLEL DO PRIVATE JBLK FIRSTPRIVATE( YLCPG_BNDS )

DO JBLK=1,YDCPG_OPTS%KGPBLKS
    CALL YLCPG_BNDS%UPDATE(JBLK)

    CALL CPPHINP (YDGEOMETRY, YDMODEL, YLCPG_BNDS%KIDIA, YLCPG_BNDS%KFDIA, Z_YDVARS_GEOMETRY_GEMU_T0&
  &(:, JBLK), Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK), Z_YDVARS_U_T0(:,:, JBLK), Z_YDVARS_V_T0(:&
  &,:, JBLK), Z_YDVARS_Q_T0(:,:, JBLK), Z_YDVARS_Q_DL(:,:, JBLK), Z_YDVARS_Q_DM(:,:, JBLK), Z_YDVARS_CVGQ_DL     &
  &(:,:, JBLK), Z_YDVARS_CVGQ_DM(:,:, JBLK), Z_YDCPG_PHY0_XYB_RDELP(:,:, JBLK), Z_YDCPG_DYN0_CTY_EVEL&
  &(:,:, JBLK), Z_YDVARS_CVGQ_T0(:,:, JBLK), ZRDG_MU0(:, JBLK), ZRDG_MU0LU(:, JBLK), ZRDG_MU0M&
  &(:, JBLK), ZRDG_MU0N(:, JBLK), ZRDG_CVGQ(:,:, JBLK), Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, JBLK))
ENDDO

IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:COMPUTE', 1, ZHOOK_HANDLE_COMPUTE)
"""

    test_call_var = ["YDGEOMETRY", "YDMODEL", "YLCPG_BNDS%KIDIA", "YLCPG_BNDS%KFDIA", 
        "Z_YDVARS_GEOMETRY_GEMU_T0(:, JBLK)", "Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK)", "Z_YDVARS_U_T0(:, :, JBLK)", 
        "Z_YDVARS_V_T0(:, :, JBLK)", "Z_YDVARS_Q_T0(:, :, JBLK)", "Z_YDVARS_Q_DL(:, :, JBLK)", 
        "Z_YDVARS_Q_DM(:, :, JBLK)", "Z_YDVARS_CVGQ_DL(:, :, JBLK)", "Z_YDVARS_CVGQ_DM(:, :, JBLK)", 
        "Z_YDCPG_PHY0_XYB_RDELP(:, :, JBLK)", "Z_YDCPG_DYN0_CTY_EVEL(:, :, JBLK)", 
        "Z_YDVARS_CVGQ_T0(:, :, JBLK)", "ZRDG_MU0(:, JBLK)", "ZRDG_MU0LU(:, JBLK)", 
        "ZRDG_MU0M(:, JBLK)", "ZRDG_MU0N(:, JBLK)", "ZRDG_CVGQ(:, :, JBLK)", 
        "Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, JBLK)"
    ]

    for node in compute_openmp[:3]:
        assert fgen(node) in test_compute
    loop = compute_openmp[3]
    assert fgen(loop.bounds) ==  '1,YDCPG_OPTS%KGPBLKS'
    assert fgen(loop.variable) ==  'JBLK'
    assert fgen(loop.body[0]) == 'CALL YLCPG_BNDS%UPDATE(JBLK)'    
    call = loop.body[1]
    assert fgen(call.name) == 'CPPHINP'
    for arg in call.arguments:
        assert fgen(arg) in test_call_var

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_compute_openmpscc(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    map_compute = transformation.compute
    compute_openmpscc = map_compute['OpenMPSingleColumn']

    test_compute= """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:COMPUTE', 0, ZHOOK_HANDLE_COMPUTE)

!$OMP PARALLEL DO PRIVATE( JBLK, JLON, YLCPG_BNDS, YLSTACK )

DO JBLK = 1, YDCPG_OPTS%KGPBLKS

    DO JLON = 1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)
      YLCPG_BNDS%KIDIA = JLON
      YLCPG_BNDS%KFDIA = JLON
      YLSTACK%L = stack_l(YSTACK, JBLK, YDCPG_OPTS%KGPBLKS)
      YLSTACK%U = stack_u(YSTACK, JBLK, YDCPG_OPTS%KGPBLKS)

      CALL CPPHINP_OPENACC (YDGEOMETRY, YDMODEL, YLCPG_BNDS%KIDIA, YLCPG_BNDS%KFDIA, Z_YDVARS_GEOMETRY_GEMU_T0&
    &(:, JBLK), Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK), Z_YDVARS_U_T0(:,:, JBLK), Z_YDVARS_V_T0(:,:, JBLK&
    &), Z_YDVARS_Q_T0(:,:, JBLK), Z_YDVARS_Q_DL(:,:, JBLK), Z_YDVARS_Q_DM(:,:, JBLK), Z_YDVARS_CVGQ_DL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_DM(:,:, JBLK), Z_YDCPG_PHY0_XYB_RDELP(:,:, JBLK), Z_YDCPG_DYN0_CTY_EVEL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_T0(:,:, JBLK), ZRDG_MU0(:, JBLK), ZRDG_MU0LU(:, JBLK), ZRDG_MU0M(:&
    &, JBLK), ZRDG_MU0N(:, JBLK), ZRDG_CVGQ(:,:, JBLK), Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, JBLK), YDSTACK=YLSTACK)
    ENDDO

ENDDO

IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:COMPUTE', 1, ZHOOK_HANDLE_COMPUTE)
"""

    test_call_var = ["YDGEOMETRY", "YDMODEL", "YLCPG_BNDS%KIDIA", "YLCPG_BNDS%KFDIA", 
        "Z_YDVARS_GEOMETRY_GEMU_T0(:, JBLK)", "Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK)", "Z_YDVARS_U_T0(:, :, JBLK)", 
        "Z_YDVARS_V_T0(:, :, JBLK)", "Z_YDVARS_Q_T0(:, :, JBLK)", "Z_YDVARS_Q_DL(:, :, JBLK)", 
        "Z_YDVARS_Q_DM(:, :, JBLK)", "Z_YDVARS_CVGQ_DL(:, :, JBLK)", "Z_YDVARS_CVGQ_DM(:, :, JBLK)", 
        "Z_YDCPG_PHY0_XYB_RDELP(:, :, JBLK)", "Z_YDCPG_DYN0_CTY_EVEL(:, :, JBLK)", 
        "Z_YDVARS_CVGQ_T0(:, :, JBLK)", "ZRDG_MU0(:, JBLK)", "ZRDG_MU0LU(:, JBLK)", 
        "ZRDG_MU0M(:, JBLK)", "ZRDG_MU0N(:, JBLK)", "ZRDG_CVGQ(:, :, JBLK)", 
        "Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, JBLK)", "YDSTACK=YLSTACK"
    ]

    assert fgen(compute_openmpscc[0]) in test_compute
    assert fgen(compute_openmpscc[1]) in test_compute
    assert fgen(compute_openmpscc[3]) in test_compute
    loop_jblk = compute_openmpscc[2]
    assert fgen(loop_jblk.bounds) ==  '1,YDCPG_OPTS%KGPBLKS'
    assert fgen(loop_jblk.variable) ==  'JBLK'
    loop_jblk_body = loop_jblk.body
    loop_jlon = loop_jblk_body[1]
    assert fgen(loop_jlon.bounds) ==  '1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)'
    assert fgen(loop_jlon.variable) ==  'JLON'
    loop_jlon_body = loop_jlon.body
    for node in loop_jlon_body[:4]:
        assert fgen(node) in test_compute
    call = loop_jlon_body[4]
    assert call.name == 'CPPHINP_OPENACC'
    for arg in call.arguments:
        assert fgen(arg) in test_call_var


@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_compute_openaccscc(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    map_compute = transformation.compute
    compute_openaccscc = map_compute['OpenACCSingleColumn']

    test_compute = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:COMPUTE', 0, ZHOOK_HANDLE_COMPUTE)


  !$ACC PARALLEL LOOP GANG &
  !$ACC&PRESENT (YDCPG_OPTS, YDGEOMETRY, YDMODEL, YSTACK, ZRDG_CVGQ, ZRDG_MU0, ZRDG_MU0LU, &
  !$ACC&         ZRDG_MU0M, ZRDG_MU0N, Z_YDCPG_DYN0_CTY_EVEL, Z_YDCPG_PHY0_XYB_RDELP, &
  !$ACC&         Z_YDVARS_CVGQ_DL, Z_YDVARS_CVGQ_DM, Z_YDVARS_CVGQ_T0, Z_YDVARS_GEOMETRY_GELAM_T0, &
  !$ACC&         Z_YDVARS_GEOMETRY_GEMU_T0, Z_YDVARS_Q_DL, Z_YDVARS_Q_DM, &
  !$ACC&         Z_YDVARS_Q_T0, Z_YDVARS_U_T0, Z_YDVARS_V_T0) &
  !$ACC&PRIVATE (JBLK) &
  !$ACC&VECTOR_LENGTH (YDCPG_OPTS%KLON) 

  DO JBLK = 1, YDCPG_OPTS%KGPBLKS



  !$ACC LOOP VECTOR &
  !$ACC&PRIVATE (JLON, YLCPG_BNDS, YLSTACK) 



    DO JLON = 1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)
      YLCPG_BNDS%KIDIA = JLON
      YLCPG_BNDS%KFDIA = JLON
      YLSTACK%L = stack_l(YSTACK, JBLK, YDCPG_OPTS%KGPBLKS)
      YLSTACK%U = stack_u(YSTACK, JBLK, YDCPG_OPTS%KGPBLKS)

      CALL CPPHINP_OPENACC (YDGEOMETRY, YDMODEL, YLCPG_BNDS%KIDIA, YLCPG_BNDS%KFDIA, Z_YDVARS_GEOMETRY_GEMU_T0&
    &(:, JBLK), Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK), Z_YDVARS_U_T0(:,:, JBLK), Z_YDVARS_V_T0(:,:, JBLK&
    &), Z_YDVARS_Q_T0(:,:, JBLK), Z_YDVARS_Q_DL(:,:, JBLK), Z_YDVARS_Q_DM(:,:, JBLK), Z_YDVARS_CVGQ_DL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_DM(:,:, JBLK), Z_YDCPG_PHY0_XYB_RDELP(:,:, JBLK), Z_YDCPG_DYN0_CTY_EVEL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_T0(:,:, JBLK), ZRDG_MU0(:, JBLK), ZRDG_MU0LU(:, JBLK), ZRDG_MU0M(:&
    &, JBLK), ZRDG_MU0N(:, JBLK), ZRDG_CVGQ(:,:, JBLK),YDSTACK=YLSTACK)
    ENDDO

ENDDO

IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE:CPPHINP:COMPUTE', 1, ZHOOK_HANDLE_COMPUTE)
"""

    test_call_var = ["YDGEOMETRY", "YDMODEL", "YLCPG_BNDS%KIDIA", "YLCPG_BNDS%KFDIA", 
        "Z_YDVARS_GEOMETRY_GEMU_T0(:, JBLK)", "Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK)", "Z_YDVARS_U_T0(:, :, JBLK)", 
        "Z_YDVARS_V_T0(:, :, JBLK)", "Z_YDVARS_Q_T0(:, :, JBLK)", "Z_YDVARS_Q_DL(:, :, JBLK)", 
        "Z_YDVARS_Q_DM(:, :, JBLK)", "Z_YDVARS_CVGQ_DL(:, :, JBLK)", "Z_YDVARS_CVGQ_DM(:, :, JBLK)", 
        "Z_YDCPG_PHY0_XYB_RDELP(:, :, JBLK)", "Z_YDCPG_DYN0_CTY_EVEL(:, :, JBLK)", 
        "Z_YDVARS_CVGQ_T0(:, :, JBLK)", "ZRDG_MU0(:, JBLK)", "ZRDG_MU0LU(:, JBLK)", 
        "ZRDG_MU0M(:, JBLK)", "ZRDG_MU0N(:, JBLK)", "ZRDG_CVGQ(:, :, JBLK)", 
        "Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, JBLK)", "YDSTACK=YLSTACK"
    ]

    assert fgen(compute_openaccscc[0]) in test_compute
    assert fgen(compute_openaccscc[3]) in test_compute
    #TODO test on first ACC pragma
    loop_jblk = compute_openaccscc[2]
    assert fgen(loop_jblk.bounds) ==  '1,YDCPG_OPTS%KGPBLKS'
    assert fgen(loop_jblk.variable) ==  'JBLK'
    loop_jblk_body = loop_jblk.body
    assert fgen(loop_jblk_body[0]) == "!$ACC LOOP VECTOR PRIVATE( JLON, YLCPG_BNDS, YLSTACK )"
    loop_jlon = loop_jblk_body[1]
    assert fgen(loop_jlon.bounds) ==  '1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)'
    assert fgen(loop_jlon.variable) ==  'JLON'
    loop_jlon_body = loop_jlon.body
    for node in loop_jlon_body[:4]:
        assert fgen(node) in test_compute
    call = loop_jlon_body[4]
    assert call.name == 'CPPHINP_OPENACC'
    for arg in call.arguments:
        assert fgen(arg) in test_call_var


    #TODO : test_imports
    #TODO : test_variables

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_variables(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    variables = transformation.dcls

    test_variables = '''TYPE(CPG_BNDS_TYPE), INTENT(IN) :: YLCPG_BNDS
TYPE(STACK) :: YLSTACK
INTEGER(KIND=JPIM) :: JBLK
REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_FIELD_API
REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_PARALLEL
REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_COMPUTE'''

    assert fgen(variables) == test_variables

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_imports(here, frontend):
    #TODO : add imports to _parallel routines

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    imports = transformation.imports

    test_imports = """
USE ACPY_MOD
USE STACK_MOD
USE YOMPARALLELMETHOD
USE FIELD_ACCESS_MODULE
USE FIELD_FACTORY_MODULE
USE FIELD_MODULE
#include "stack.h"
"""
    for imp in imports:
        assert fgen(imp) in test_imports

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_new_callee_imports(here, frontend):
    #TODO : add imports to _parallel routines

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    routine = source['dispatch_routine']

    transformation = ParallelRoutineDispatchTransformation()
    transformation.apply(source['dispatch_routine'])

    imports = transformation.callee_imports

    assert fgen(imports) == '#include "cpphinp_openacc.intfb.h"'