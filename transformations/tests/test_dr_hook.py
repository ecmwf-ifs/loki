import shutil
import pytest

from loki import (
    Scheduler, SchedulerConfig, FindNodes, CallStatement, gettempdir, OMNI
)

from conftest import available_frontends
from transformations import DrHookTransformation


@pytest.fixture(scope='module', name='config')
def fixture_config():
    """
    Write default configuration as a temporary file and return
    the file path
    """
    default_config = {
        'default': {'role': 'kernel', 'expand': True, 'strict': True, 'disable': ['dr_hook']},
        'routine': [
            {'name': 'rick_astley', 'role': 'driver'},
        ]
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
    implicit none

    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('rick_astley',0,zhook_handle)
    call never_gonna_give()
    if (lhook) call dr_hook('rick_astley',1,zhook_handle)
end subroutine
    """.strip()

    fcode_kernel = """
subroutine never_gonna_give
    use parkind1, only: jprb
    use yomhook, only : lhook, dr_hook
    implicit none

    real(kind=jprb) :: zhook_handle
    if (lhook) call dr_hook('never_gonna_give',0,zhook_handle)
    print *, 'never gonna let you down'
    if (lhook) call dr_hook('never_gonna_give',1,zhook_handle)
end subroutine
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

    for item in scheduler.items:
        drhook_calls = [
            call for call in FindNodes(CallStatement).visit(item.routine.ir)
            if call.name == 'dr_hook'
        ]
        assert len(drhook_calls) == 2
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

    for item in scheduler.items:
        drhook_calls = [
            call for call in FindNodes(CallStatement).visit(item.routine.ir)
            if call.name == 'dr_hook'
        ]
        if item.role == 'driver':
            assert len(drhook_calls) == 2
            assert all(
                str(call.arguments[0]).lower().strip("'") == item.local_name.lower()
                for call in drhook_calls
            )
        elif item.role == 'kernel':
            assert not drhook_calls
