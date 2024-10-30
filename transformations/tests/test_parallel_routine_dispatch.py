# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest

from loki.frontend import available_frontends, OMNI
from loki import Sourcefile, FindNodes, CallStatement, fgen, Conditional, ProcedureItem
from loki import Loop

from transformations.parallel_routine_dispatch import ParallelRoutineDispatchTransformation

import os


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_dr_hook(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 4

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"
    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    calls = [call for call in FindNodes(CallStatement).visit(routine.body) if call.name.name=='DR_HOOK']
    assert len(calls) == 32

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_decl_local_arrays(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']



    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)
    var_lst=["YL_ZRDG_CVGQ",
        "ZRDG_CVGQ",
        "YL_ZRDG_MU0LU",
        "ZRDG_MU0LU",
        "YL_ZRDG_MU0M",
        "ZRDG_MU0M",
        "YL_ZRDG_MU0N",
        "ZRDG_MU0N",
        "YL_ZRDG_MU0",
        "ZRDG_MU0",
        "YL_ZPFL_FPLCH",
        "ZPFL_FPLCH",
        "YL_ZPFL_FPLSH",
        "ZPFL_FPLSH",
        "YL_TOTO",
        "TOTO"]
    dcls = [dcl for dcl in routine.declarations if dcl.symbols[0].name in var_lst]
    str_dcls = ""
    for dcl in dcls:
        str_dcls += fgen(dcl)+"\n"
    dcls_test = """CLASS(FIELD_3RB), POINTER :: YL_ZRDG_CVGQ => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_CVGQ(:, :, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0LU => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0LU(:, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0M => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0M(:, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0N => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0N(:, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_ZRDG_MU0 => NULL()
REAL(KIND=JPRB), POINTER :: ZRDG_MU0(:, :) => NULL()
CLASS(FIELD_3RB), POINTER :: YL_ZPFL_FPLCH => NULL()
REAL(KIND=JPRB), POINTER :: ZPFL_FPLCH(:, :, :) => NULL()
CLASS(FIELD_3RB), POINTER :: YL_ZPFL_FPLSH => NULL()
REAL(KIND=JPRB), POINTER :: ZPFL_FPLSH(:, :, :) => NULL()
CLASS(FIELD_2RB), POINTER :: YL_TOTO => NULL()
REAL(KIND=JPRB), POINTER :: TOTO(:, :) => NULL()
"""

    assert str_dcls == dcls_test

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_decl_field_create_delete(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    var_lst = ["YL_ZRDG_CVGQ", "ZRDG_CVGQ", "YL_ZRDG_MU0LU", "ZRDG_MU0LU", "YL_ZRDG_MU0M", "ZRDG_MU0M", "YL_ZRDG_MU0N", "ZRDG_MU0N", "YL_ZRDG_MU0", "ZRDG_MU0"]
    field_create = ["CALL FIELD_NEW(YL_ZRDG_CVGQ, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KFLEVG, YDCPG_OPTS%JBLKMAX /),  &\n& LBOUNDS=(/ 1, 1, YDCPG_OPTS%JBLKMIN /), PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0N, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%JBLKMAX /), LBOUNDS=(/ 1, YDCPG_OPTS%JBLKMIN /),  &\n& PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0LU, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%JBLKMAX /), LBOUNDS=(/ 1, YDCPG_OPTS%JBLKMIN /),  &\n& PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%JBLKMAX /), LBOUNDS=(/ 1, YDCPG_OPTS%JBLKMIN /),  &\n& PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZRDG_MU0M, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%JBLKMAX /), LBOUNDS=(/ 1, YDCPG_OPTS%JBLKMIN /),  &\n& PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZPFL_FPLSH, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KFLEVG, YDCPG_OPTS%JBLKMAX /),  &\n& LBOUNDS=(/ 1, 0, YDCPG_OPTS%JBLKMIN /), PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_ZPFL_FPLCH, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%KFLEVG, YDCPG_OPTS%JBLKMAX /),  &\n& LBOUNDS=(/ 1, 0, YDCPG_OPTS%JBLKMIN /), PERSISTENT=.true.)",
                "CALL FIELD_NEW(YL_TOTO, UBOUNDS=(/ YDCPG_OPTS%KLON, YDCPG_OPTS%JBLKMAX /), LBOUNDS=(/ 1, YDCPG_OPTS%JBLKMIN /), PERSISTENT=.true. &\n& )"
                ]

    calls = [call for call in FindNodes(CallStatement).visit(routine.body) if call.name.name=="FIELD_NEW"]
    assert len(calls) == 8
    for call in calls:
        assert fgen(call) in field_create
    
    field_delete = ["IF (ASSOCIATED(YL_ZRDG_CVGQ)) CALL FIELD_DELETE(YL_ZRDG_CVGQ)",
                "IF (ASSOCIATED(YL_ZRDG_MU0LU)) CALL FIELD_DELETE(YL_ZRDG_MU0LU)",
                "IF (ASSOCIATED(YL_ZRDG_MU0M)) CALL FIELD_DELETE(YL_ZRDG_MU0M)",
                "IF (ASSOCIATED(YL_ZRDG_MU0)) CALL FIELD_DELETE(YL_ZRDG_MU0)",
                "IF (ASSOCIATED(YL_ZRDG_MU0N)) CALL FIELD_DELETE(YL_ZRDG_MU0N)",
                "IF (ASSOCIATED(YL_ZPFL_FPLSH)) CALL FIELD_DELETE(YL_ZPFL_FPLSH)",
                "IF (ASSOCIATED(YL_ZPFL_FPLCH)) CALL FIELD_DELETE(YL_ZPFL_FPLCH)",
                "IF (ASSOCIATED(YL_TOTO)) CALL FIELD_DELETE(YL_TOTO)"
                ]

    conds = [cond for cond in FindNodes(Conditional).visit(routine.body)]
    conditional = []
    for cond in conds:
        for call in FindNodes(CallStatement).visit(cond):
                if call.name.name=="FIELD_DELETE":
                    conditional.append(cond)

    assert len(conditional) == 8
    for cond in conditional:
        assert fgen(cond) in field_delete

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_derived_dcl(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

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
"REAL(KIND=JPRB), POINTER :: Z_YDVARS_CVGQ_DL(:, :, :)",
"REAL(KIND=JPRB), POINTER :: Z_YDMF_PHYS_OUT_CT(:, :)"]
    for dcl in dcls:
        assert dcl in test_dcls

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_derived_var(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    
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
        "YDMF_PHYS_SURF%GSD_VF%PZ0F" : ["YDMF_PHYS_SURF%GSD_VF%F_Z0F", "Z_YDMF_PHYS_SURF_GSD_VF_PZ0F"],
        "YDMF_PHYS%OUT%CT": ["YDMF_PHYS%OUT%F_CT", "Z_YDMF_PHYS_OUT_CT"]
    }
    routine_map_derived = item.trafo_data['create_parallel']['map_routine']['map_derived']
    for var_name in routine_map_derived:
        value = routine_map_derived[var_name]
        field_ptr = value[0]
        ptr = value[1]

        assert test_map[var_name][0] == field_ptr.name
        assert test_map[var_name][1] == ptr.name

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_get_data(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    get_data = item.trafo_data['create_parallel']['map_routine']['map_region']['get_data']
    
    test_get_data = {}
#    test_get_data["OpenMP"] = """
#IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)
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
#IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
#"""
    test_get_data["OpenMP"] = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)
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
TOTO => GET_HOST_DATA_RDWR(YL_TOTO)
Z_YDMF_PHYS_OUT_CT => GET_HOST_DATA_RDWR(YDMF_PHYS%OUT%F_CT)
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
"""
    test_get_data["OpenMPSingleColumn"] = test_get_data["OpenMP"]

#    test_get_data["OpenACCSingleColumn"] = """
#IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)                                                                       
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
#IF(LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
#"""

    test_get_data["OpenACCSingleColumn"] = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 0, ZHOOK_HANDLE_FIELD_API)                                                                       
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
TOTO => GET_DEVICE_DATA_RDWR(YL_TOTO)
Z_YDMF_PHYS_OUT_CT => GET_DEVICE_DATA_RDWR(YDMF_PHYS%OUT%F_CT)
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA', 1, ZHOOK_HANDLE_FIELD_API)
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
    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

###    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
###    transformation.apply(source['dispatch_routine'], item=item)
###
###    get_data = transformation.get_data
###    
###    test_get_data = """
###IF (LHOOK) CALL DR_HOOK ('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA',0,ZHOOK_HANDLE_FIELD_API)
###Z_YDMF_PHYS_SURF_GSD_VV_PZ0H => GET_HOST_DATA_RDWR (YDMF_PHYS_SURF%GSD_VV%F_Z0H)
###IF (LHOOK) CALL DR_HOOK ('DISPATCH_ROUTINE_PARALLEL:CPPHINP:GET_DATA',1,ZHOOK_HANDLE_FIELD_API)
###"""
###
###    for node in get_data:
###        assert fgen(node) in test_get_data
###

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_synchost(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    synchost = item.trafo_data['create_parallel']['map_routine']['map_region']['synchost']
    
    test_synchost = """IF (LSYNCHOST('DISPATCH_ROUTINE_PARALLEL:CPPHINP')) THEN
 IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:SYNCHOST', 0, ZHOOK_HANDLE_FIELD_API)
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
 TOTO => GET_HOST_DATA_RDWR(YL_TOTO)
 Z_YDMF_PHYS_OUT_CT => GET_HOST_DATA_RDWR(YDMF_PHYS%OUT%F_CT)
 IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:SYNCHOST', 1, ZHOOK_HANDLE_FIELD_API)
 ENDIF
"""
    assert fgen(synchost.condition) in test_synchost
    for node in synchost.body:
        assert fgen(node) in test_synchost

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_nullify(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    nullify = item.trafo_data['create_parallel']['map_routine']['map_region']['nullify']
 
    test_nullify = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:NULLIFY', 0, ZHOOK_HANDLE_FIELD_API)
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
TOTO => NULL()
Z_YDMF_PHYS_OUT_CT => NULL()
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:NULLIFY', 1, ZHOOK_HANDLE_FIELD_API)
"""

    for node in nullify:
        assert fgen(node) in test_nullify


@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_compute_openmp(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    map_compute = item.trafo_data['create_parallel']['map_routine']['map_region']['compute']
    compute_openmp = map_compute['OpenMP']
 
    test_compute= """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:COMPUTE', 0, ZHOOK_HANDLE_COMPUTE)

CALL YLCPG_BNDS%INIT(YDCPG_OPTS)
!$OMP PARALLEL DO PRIVATE( JBLK, JLON ) FIRSTPRIVATE( YLCPG_BNDS )


DO JBLK=YDCPG_OPTS%JBLKMIN,YDCPG_OPTS%JBLKMAX
    CALL YLCPG_BNDS%UPDATE(JBLK)

    CALL CPPHINP (YDGEOMETRY, YDMODEL, YLCPG_BNDS%KIDIA, YLCPG_BNDS%KFDIA, Z_YDVARS_GEOMETRY_GEMU_T0&
  &(:, JBLK), Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK), Z_YDVARS_U_T0(:,:, JBLK), Z_YDVARS_V_T0(:&
  &,:, JBLK), Z_YDVARS_Q_T0(:,:, JBLK), Z_YDVARS_Q_DL(:,:, JBLK), Z_YDVARS_Q_DM(:,:, JBLK), Z_YDVARS_CVGQ_DL     &
  &(:,:, JBLK), Z_YDVARS_CVGQ_DM(:,:, JBLK), Z_YDCPG_PHY0_XYB_RDELP(:,:, JBLK), Z_YDCPG_DYN0_CTY_EVEL&
  &(:,:, JBLK), Z_YDVARS_CVGQ_T0(:,:, JBLK), ZRDG_MU0(:, JBLK), ZRDG_MU0LU(:, JBLK), ZRDG_MU0M&
  &(:, JBLK), ZRDG_MU0N(:, JBLK), ZRDG_CVGQ(:,:, JBLK), Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, JBLK))
ENDDO

IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:COMPUTE', 1, ZHOOK_HANDLE_COMPUTE)
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
    assert fgen(loop.bounds) ==  'YDCPG_OPTS%JBLKMIN,YDCPG_OPTS%JBLKMAX'
    assert fgen(loop.variable) ==  'JBLK'
    assert fgen(loop.body[0]) == 'CALL YLCPG_BNDS%UPDATE(JBLK)'    
    call = loop.body[2]
    assert fgen(call.name) == 'CPPHINP'
    for arg in call.arguments:
        assert fgen(arg) in test_call_var

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_compute_openmpscc(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    map_compute = item.trafo_data['create_parallel']['map_routine']['map_region']['compute']
    compute_openmpscc = map_compute['OpenMPSingleColumn']

    test_compute= """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:COMPUTE', 0, ZHOOK_HANDLE_COMPUTE)

!$OMP PARALLEL DO PRIVATE( JBLK, JLON, YLCPG_BNDS, YLSTACK )

DO JBLK = YDCPG_OPTS%JBLKMIN, YDCPG_OPTS%JBLKMAX

    DO JLON = 1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)
      YLCPG_BNDS%KIDIA = JLON
      YLCPG_BNDS%KFDIA = JLON

      YLSTACK%L8 = stack_l8(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)
      YLSTACK%U8 = stack_u8(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)
      YLSTACK%L4 = stack_l4(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)
      YLSTACK%U4 = stack_u4(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)

      CALL CPPHINP_OPENACC (YDGEOMETRY, YDMODEL, YLCPG_BNDS%KIDIA, YLCPG_BNDS%KFDIA, Z_YDVARS_GEOMETRY_GEMU_T0&
    &(:, JBLK), Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK), Z_YDVARS_U_T0(:,:, JBLK), Z_YDVARS_V_T0(:,:, JBLK&
    &), Z_YDVARS_Q_T0(:,:, JBLK), Z_YDVARS_Q_DL(:,:, JBLK), Z_YDVARS_Q_DM(:,:, JBLK), Z_YDVARS_CVGQ_DL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_DM(:,:, JBLK), Z_YDCPG_PHY0_XYB_RDELP(:,:, JBLK), Z_YDCPG_DYN0_CTY_EVEL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_T0(:,:, JBLK), ZRDG_MU0(:, JBLK), ZRDG_MU0LU(:, JBLK), ZRDG_MU0M(:&
    &, JBLK), ZRDG_MU0N(:, JBLK), ZRDG_CVGQ(:,:, JBLK), Z_YDMF_PHYS_SURF_GSD_VF_PZ0F(:, JBLK), YDSTACK=YLSTACK)
    ENDDO

ENDDO

IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:COMPUTE', 1, ZHOOK_HANDLE_COMPUTE)
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
    assert fgen(loop_jblk.bounds) ==  'YDCPG_OPTS%JBLKMIN,YDCPG_OPTS%JBLKMAX'
    assert fgen(loop_jblk.variable) ==  'JBLK'
    loop_jblk_body = loop_jblk.body
    loop_jlon = loop_jblk_body[0]
    assert fgen(loop_jlon.bounds) ==  '1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)'
    assert fgen(loop_jlon.variable) ==  'JLON'
    loop_jlon_body = loop_jlon.body
    for node in loop_jlon_body[:4]:
        assert fgen(node) in test_compute
    call = loop_jlon_body[7]
    assert call.name == 'CPPHINP_OPENACC'
    for arg in call.arguments:
        assert fgen(arg) in test_call_var


@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_compute_openaccscc(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    map_compute = item.trafo_data['create_parallel']['map_routine']['map_region']['compute']
    compute_openaccscc = map_compute['OpenACCSingleColumn']

    test_compute = """
IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:COMPUTE', 0, ZHOOK_HANDLE_COMPUTE)


  !$ACC PARALLEL LOOP GANG &
  !$ACC&PRESENT (YDCPG_OPTS, YDMODEL, YSTACK, ZRDG_CVGQ, ZRDG_MU0, ZRDG_MU0LU, TOTO, &
  !$ACC&         ZRDG_MU0M, ZRDG_MU0N, Z_YDCPG_DYN0_CTY_EVEL, Z_YDCPG_PHY0_XYB_RDELP, &
  !$ACC&         Z_YDVARS_CVGQ_DL, Z_YDVARS_CVGQ_DM, Z_YDVARS_CVGQ_T0, Z_YDVARS_GEOMETRY_GELAM_T0, &
  !$ACC&         Z_YDVARS_GEOMETRY_GEMU_T0, Z_YDVARS_Q_DL, Z_YDVARS_Q_DM, &
  !$ACC&         Z_YDVARS_Q_T0, Z_YDVARS_U_T0, Z_YDVARS_V_T0) &
  !$ACC&PRIVATE (JBLK) &
  !$ACC&VECTOR_LENGTH (YDCPG_OPTS%KLON) 

  DO JBLK = YDCPG_OPTS%JBLKMIN,YDCPG_OPTS%JBLKMAX 



  !$ACC LOOP VECTOR &
  !$ACC&PRIVATE (JLON, YLCPG_BNDS, YLSTACK) 



    DO JLON = 1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)
      YLCPG_BNDS%KIDIA = JLON
      YLCPG_BNDS%KFDIA = JLON
      YLSTACK%L8 = stack_l8(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)
      YLSTACK%U8 = stack_u8(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)
      YLSTACK%L4 = stack_l4(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)
      YLSTACK%U4 = stack_u4(YSTACK, JBLK - YDCPG_OPTS%JBLKMIN + 1, YDCPG_OPTS%KGPBLKS)

      CALL CPPHINP_OPENACC (YDGEOMETRY, YDMODEL, YLCPG_BNDS%KIDIA, YLCPG_BNDS%KFDIA, Z_YDVARS_GEOMETRY_GEMU_T0&
    &(:, JBLK), Z_YDVARS_GEOMETRY_GELAM_T0(:, JBLK), Z_YDVARS_U_T0(:,:, JBLK), Z_YDVARS_V_T0(:,:, JBLK&
    &), Z_YDVARS_Q_T0(:,:, JBLK), Z_YDVARS_Q_DL(:,:, JBLK), Z_YDVARS_Q_DM(:,:, JBLK), Z_YDVARS_CVGQ_DL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_DM(:,:, JBLK), Z_YDCPG_PHY0_XYB_RDELP(:,:, JBLK), Z_YDCPG_DYN0_CTY_EVEL&
    &(:,:, JBLK), Z_YDVARS_CVGQ_T0(:,:, JBLK), ZRDG_MU0(:, JBLK), ZRDG_MU0LU(:, JBLK), ZRDG_MU0M(:&
    &, JBLK), ZRDG_MU0N(:, JBLK), ZRDG_CVGQ(:,:, JBLK),YDSTACK=YLSTACK)
    ENDDO

ENDDO

IF (LHOOK) CALL DR_HOOK('DISPATCH_ROUTINE_PARALLEL:CPPHINP:COMPUTE', 1, ZHOOK_HANDLE_COMPUTE)
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
    assert fgen(loop_jblk.bounds) ==  'YDCPG_OPTS%JBLKMIN,YDCPG_OPTS%JBLKMAX'
    assert fgen(loop_jblk.variable) ==  'JBLK'
    loop_jblk_body = loop_jblk.body
    assert fgen(loop_jblk_body[0]) == "!$ACC LOOP VECTOR PRIVATE( JLON, YLCPG_BNDS, YLSTACK )"
    loop_jlon = loop_jblk_body[1]
    assert fgen(loop_jlon.bounds) ==  '1,MIN(YDCPG_OPTS%KLON, YDCPG_OPTS%KGPCOMP - (JBLK - 1)*YDCPG_OPTS%KLON)'
    assert fgen(loop_jlon.variable) ==  'JLON'
    loop_jlon_body = loop_jlon.body
    for node in loop_jlon_body[:4]:
        assert fgen(node) in test_compute
    call = loop_jlon_body[7]
    assert call.name == 'CPPHINP_OPENACC'
    for arg in call.arguments:
        assert fgen(arg) in test_call_var


    #TODO : test_imports
    #TODO : test_variables

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_variables(here, frontend):

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    variables = item.trafo_data['create_parallel']['map_routine']['variable_declarations']

    routine_str = fgen(routine)

    test_variables = ["TYPE(CPG_BNDS_TYPE) :: YLCPG_BNDS", 
"TYPE(STACK) :: YLSTACK", 
"INTEGER(KIND=JPIM) :: JBLK", 
"REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_FIELD_API", 
"REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_PARALLEL", 
"REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_COMPUTE"]

    for var in test_variables : 
        assert var in routine_str

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_imports(here, frontend):
    #TODO : add imports to _parallel routines

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    imports = item.trafo_data['create_parallel']['map_routine']['imports']
    imports = [fgen(imp) for imp in imports]

    test_imports = [
"USE ACPY_MOD", 
"USE STACK_MOD", 
"USE YOMPARALLELMETHOD", 
"USE FIELD_ACCESS_MODULE", 
"USE FIELD_FACTORY_MODULE", 
"USE FIELD_MODULE", 
'#include "stack.h"']

    assert len(test_imports) == len(imports)
    for imp in test_imports:
        assert fgen(imp) in imports

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_imports(here, frontend):
    #TODO : add imports to _parallel routines

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)

    routine_str = fgen(routine)

    assert '#include "cpphinp_openacc.intfb.h"' in routine_str
    assert '#include "cpphinp.intfb.h"' in routine_str
    assert '#include "mf_phys_fpl_part1.intfb.h"' not in routine_str
    assert '#include "mf_phys_fpl_part1_parallel.intfb.h"' in routine_str

@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
def test_parallel_routine_dispatch_lparallel(here, frontend):
    #TODO : add imports to _parallel routines

    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
    routine = source['dispatch_routine']

    is_intent = False 
    horizontal = [
            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
            "KPROMA", "YDDIM%NPROMA", "NPROMA"
    ]
    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"

    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
    transformation.apply(source['dispatch_routine'], item=item)


    routine_str = fgen(routine)
    assert "LPARALLELMETHOD" in fgen(routine_str)
    assert "'OPENMP'" in fgen(routine_str)
    assert "'OPENMPSINGLECOLUMN'" in fgen(routine_str)
    assert "'OPENACCSINGLECOLUMN'" in fgen(routine_str)


#@pytest.mark.parametrize('frontend', available_frontends(skip=[OMNI]))
#def test_parallel_routine_dispatch_lparallel(here, frontend):
#
#    source = Sourcefile.from_file(here/'sources/projParallelRoutineDispatch/dispatch_routine.F90', frontend=frontend)
#    item = ProcedureItem(name='parallel_routine_dispatch', source=source)
#    routine = source['dispatch_routine']
#
#    is_intent = False 
#    horizontal = [
#            "KLON", "YDCPG_OPTS%KLON", "YDGEOMETRY%YRDIM%NPROMA",
#            "KPROMA", "YDDIM%NPROMA", "NPROMA"
#    ]
#    path_map_index = os.getcwd()+"/transformations/transformations/field_index.pkl"
#
#    transformation = ParallelRoutineDispatchTransformation(is_intent, horizontal, path_map_index)
#    transformation.apply(source['dispatch_routine'], item=item)
#
#    loops = [loop for loop in FindNodes(Loop).visit(routine) if loop.variable.name=="JLON"]
#
#    str_loops_body = [fgen(loop.body) for loop in loops]
##    str_loops_spec = [fgen(loop.spec) for loop in loops]
#    #str_loops = [fgen(loop) for loop in loops]
#
#    test_loop="Z_YDMF_PHYS_OUT_CT(JLON, JBLK) = TOTO(JLON, JBLK)"
#
#    print(fgen(routine.body))
#    assert test_loop in str_loops_body
        
