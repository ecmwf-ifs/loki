# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import shutil
import pytest

from loki import (
    Scheduler, SFilter, ProcedureItem, SchedulerConfig, FindNodes,
    CallStatement, gettempdir, OMNI, Import
)

from conftest import available_frontends
from loki.transformations import DrHookTransformation


@pytest.fixture(scope='module', name='config')
def fixture_config():
    """
    Write default configuration as a temporary file and return
    the file path
    """
    default_config = {
        'default': {
            'role': 'kernel', 'expand': True, 'strict': False, 'disable': ['dr_hook', 'abor1']
        },
        'routines': {
            'rick_astley': {'role': 'driver'},
        }
    }
    return default_config


@pytest.fixture(scope='module', name='srcdir')
def fixture_srcdir():
    """
    Create a src directory in the temp directory
    """
    srcdir = gettempdir()/'test_dr_hook'
    if srcdir.exists():
        shutil.rmtree(srcdir)
    srcdir.mkdir()
    yield srcdir
    shutil.rmtree(srcdir)


@pytest.fixture(scope='module', name='source')
def fixture_source(srcdir):
    """
    Write some source files to use in the test
    """
    fcode_driver = """
subroutine rick_astley
    use parkind1, only: jprb
    use yomhook, only : lhook, dr_hook
    use rick_rolled, only : never_gonna_give
    implicit none

    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('rick_astley',0,zhook_handle)
    call never_gonna_give()
    if (lhook) call dr_hook('rick_astley',1,zhook_handle)
end subroutine
    """.strip()

    fcode_kernel = """
module rick_rolled
contains
subroutine never_gonna_give
    use parkind1, only: jprb
    use yomhook, only : lhook, dr_hook
    implicit none

    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('never_gonna_give',0,zhook_handle)

    CALL ABOR1('[SUBROUTINE CALL]')

    print *, 'never gonna let you down'

    if (dave) call abor1('[INLINE CONDITIONAL]')

    call never_gonna_run_around()

    WRITE(NULOUT,*) "[WRITE INTRINSIC]"
    if (.not. dave) WRITE(NULOUT, *) "[WRITE INTRINSIC]"

    if (lhook) call dr_hook('never_gonna_give',1,zhook_handle)

contains

subroutine never_gonna_run_around

    implicit none

    if (lhook) call dr_hook('never_gonna_run_around',0,zhook_handle)

    if (dave) call abor1('[INLINE CONDITIONAL]')
    WRITE(NULOUT,*) "[WRITE INTRINSIC]"
    if (.not. dave) WRITE(NULOUT, *) "[WRITE INTRINSIC]"

    if (lhook) call dr_hook('never_gonna_run_around',1,zhook_handle)

end subroutine never_gonna_run_around

end subroutine
subroutine i_hope_you_havent_let_me_down
    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('i_hope_you_havent_let_me_down',0,zhook_handle)

    if (lhook) call dr_hook('i_hope_you_havent_let_me_down',1,zhook_handle)
end subroutine i_hope_you_havent_let_me_down
end module rick_rolled
    """.strip()

    (srcdir/'rick_astley.F90').write_text(fcode_driver)
    (srcdir/'never_gonna_give.F90').write_text(fcode_kernel)

    yield srcdir

    (srcdir/'rick_astley.F90').unlink()
    (srcdir/'never_gonna_give.F90').unlink()


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'Incomplete source tree impossible with OMNI')]
))
def test_dr_hook_transformation(frontend, config, source):
    """Test DrHook transformation for a renamed Subroutine"""
    scheduler_config = SchedulerConfig.from_dict(config)
    scheduler = Scheduler(paths=source, config=scheduler_config, frontend=frontend)
    scheduler.process(transformation=DrHookTransformation(mode='you_up'))

    for item in SFilter(scheduler.sgraph, item_filter=ProcedureItem):
        drhook_calls = [
            call for call in FindNodes(CallStatement).visit(item.ir.ir)
            if call.name == 'dr_hook'
        ]
        assert len(drhook_calls) == 2
        drhook_imports = [
            imp for imp in FindNodes(Import).visit(item.ir.ir)
            if imp.module == 'yomhook'
        ]
        assert len(drhook_imports) == 1
        assert 'zhook_handle' in item.ir.variables
        if item.role == 'driver':
            assert all(
                str(call.arguments[0]).lower().strip("'") == item.local_name.lower()
                for call in drhook_calls
            )
        elif item.role == 'kernel':
            assert all(
                str(call.arguments[0]).lower().strip("'") == f'{item.local_name.lower()}_you_up'
                for call in drhook_calls
            )


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'Incomplete source tree impossible with OMNI')]
))
def test_dr_hook_transformation_remove(frontend, config, source):
    """Test DrHook transformation in remove mode"""
    scheduler_config = SchedulerConfig.from_dict(config)
    scheduler = Scheduler(paths=source, config=scheduler_config, frontend=frontend)
    scheduler.process(transformation=DrHookTransformation(mode='you_up', remove=True))

    for item in SFilter(scheduler.sgraph, item_filter=ProcedureItem):
        drhook_calls = [
            call for call in FindNodes(CallStatement).visit(item.ir.ir)
            if call.name == 'dr_hook'
        ]
        drhook_imports = [
            imp for imp in FindNodes(Import).visit(item.ir.ir)
            if imp.module == 'yomhook'
        ]
        for r in item.ir.members:
            drhook_calls += [
                call for call in FindNodes(CallStatement).visit(r.ir)
                if call.name == 'dr_hook'
            ]
            drhook_imports += [
                imp for imp in FindNodes(Import).visit(item.ir.ir)
                if imp.module == 'yomhook'
            ]
        if item.role == 'driver':
            assert len(drhook_calls) == 2
            assert len(drhook_imports) == 1
            assert 'zhook_handle' in item.ir.variables
            assert all(
                str(call.arguments[0]).lower().strip("'") == item.local_name.lower()
                for call in drhook_calls
            )
        elif item.role == 'kernel':
            assert not drhook_calls
            assert not drhook_imports
            assert 'zhook_handle' not in item.ir.variables
