from pathlib import Path
from shutil import rmtree
import pytest

from conftest import available_frontends
from loki import (
    gettempdir, OMNI, OFP, Sourcefile, CallStatement, Import,
    FindNodes, FindInlineCalls, Intrinsic, Scheduler, SchedulerConfig
)
from loki.transform import DependencyTransformation


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent

@pytest.fixture(scope='function', name='tempdir')
def fixture_tempdir(request):
    basedir = gettempdir()/request.function.__name__
    basedir.mkdir(exist_ok=True)
    yield basedir
    if basedir.exists():
        rmtree(basedir)


@pytest.fixture(scope='function', name='config')
def fixture_config():
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routine': [{
            'name': 'driver',
            'role': 'driver',
        }]
    }


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('use_scheduler', [False, True])
def test_dependency_transformation_globalvar_imports(frontend, use_scheduler, tempdir, config):
    """
    Test that global variable imports are not renamed as a
    call statement would be.
    """

    kernel_fcode = """
MODULE kernel_mod
    INTEGER :: some_const
CONTAINS
    SUBROUTINE kernel(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    a = 1
    b = 2
    c = 3
  END SUBROUTINE kernel
END MODULE kernel_mod
    """.strip()

    driver_fcode = """
SUBROUTINE driver(a, b, c)
    USE kernel_mod, only: kernel
    USE kernel_mod, only: some_const
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
END SUBROUTINE driver
    """.strip()

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')

    if use_scheduler:
        (tempdir/'kernel_mod.F90').write_text(kernel_fcode)
        (tempdir/'driver.F90').write_text(driver_fcode)
        scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)
        scheduler.process(transformation, use_file_graph=True)

        kernel = scheduler['kernel_mod#kernel'].source
        driver = scheduler['#driver'].source

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel')
        driver['driver'].apply(transformation, role='driver', targets=('kernel', 'some_const'))

    # Check that the global variable declaration remains unchanged
    assert kernel.modules[0].variables[0].name == 'some_const'

    # Check that calls and matching import have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 2
    assert isinstance(imports[0], Import)
    assert driver['driver'].spec.body[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in driver['driver'].spec.body[0].symbols]

    # Check that global variable import remains unchanged
    assert isinstance(imports[1], Import)
    assert driver['driver'].spec.body[1].module == 'kernel_mod'
    assert 'some_const' in [str(s) for s in driver['driver'].spec.body[1].symbols]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('use_scheduler', [False, True])
def test_dependency_transformation_globalvar_imports_driver_mod(frontend, use_scheduler, tempdir, config):
    """
    Test that global variable imports are not renamed as a
    call statement would be.
    """

    kernel_fcode = """
MODULE kernel_mod
    INTEGER :: some_const
CONTAINS
    SUBROUTINE kernel(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    a = 1
    b = 2
    c = 3
  END SUBROUTINE kernel
END MODULE kernel_mod
    """.strip()

    driver_fcode = """
MODULE DRIVER_MOD
    USE kernel_mod, only: kernel
    USE kernel_mod, only: some_const
CONTAINS
SUBROUTINE driver(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
END SUBROUTINE driver
END MODULE DRIVER_MOD
    """.strip()

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')

    if use_scheduler:
        (tempdir/'kernel_mod.F90').write_text(kernel_fcode)
        (tempdir/'driver_mod.F90').write_text(driver_fcode)
        scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)
        scheduler.process(transformation, use_file_graph=True)

        kernel = scheduler['kernel_mod#kernel'].source
        driver = scheduler['driver_mod#driver'].source

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel')
        driver.apply(transformation, role='driver', targets=('kernel', 'some_const'))

    # Check that the global variable declaration remains unchanged
    assert kernel.modules[0].variables[0].name == 'some_const'

    # Check that calls and matching import have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver_mod'].spec)
    assert len(imports) == 2
    assert isinstance(imports[0], Import)
    assert driver['driver_mod'].spec.body[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in driver['driver_mod'].spec.body[0].symbols]

    # Check that global variable import remains unchanged
    assert isinstance(imports[1], Import)
    assert driver['driver_mod'].spec.body[1].module == 'kernel_mod'
    assert 'some_const' in [str(s) for s in driver['driver_mod'].spec.body[1].symbols]


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'C-imports need pre-processing for OMNI')]))
def test_dependency_transformation_header_includes(here, frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines via c-header includes.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

#include "kernel.intfb.h"

  CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
""", frontend=frontend)

    # Ensure header file does not exist a-priori
    header_file = here/'kernel_test.intfb.h'
    if header_file.exists():
        header_file.unlink()

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='strict', include_path=here)
    kernel['kernel'].apply(transformation, role='kernel')
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the subroutine name in the kernel source has changed
    assert len(kernel.modules) == 0
    assert len(kernel.subroutines) == 1
    assert kernel.subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]

    # Check that the driver name has not changed
    assert len(kernel.modules) == 0
    assert len(kernel.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that the import has been updated
    assert '#include "kernel.intfb.h"' not in driver.to_fortran()
    assert '#include "kernel_test.intfb.h"' in driver.to_fortran()

    # Check that header file was generated and clean up
    assert header_file.exists()
    header_file.unlink()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'C-imports need pre-processing for OMNI')]))
@pytest.mark.parametrize('use_scheduler', [False, True])
def test_dependency_transformation_module_wrap(frontend, use_scheduler, tempdir, config):
    """
    Test injection of suffixed kernels into unchanged driver
    routines automatic module wrapping of the kernel.
    """

    driver_fcode = """
SUBROUTINE driver(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

#include "kernel.intfb.h"

  CALL kernel(a, b ,c)
END SUBROUTINE driver
    """.strip()

    kernel_fcode = """
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
    """.strip()

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')

    if use_scheduler:
        (tempdir/'kernel.F90').write_text(kernel_fcode)
        (tempdir/'driver.F90').write_text(driver_fcode)
        scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)
        scheduler.process(transformation, use_file_graph=True)

        kernel = scheduler['#kernel'].source
        driver = scheduler['#driver'].source

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel')
        driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    assert imports[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in imports[0].symbols]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('use_scheduler', [False, True])
def test_dependency_transformation_replace_interface(frontend, use_scheduler, tempdir, config):
    """
    Test injection of suffixed kernels defined in interface block
    into unchanged driver routines automatic module wrapping of the kernel.
    """

    driver_fcode = """
SUBROUTINE driver(a, b, c)
  IMPLICIT NONE
  INTERFACE
    SUBROUTINE KERNEL(a, b, c)
      INTEGER, INTENT(INOUT) :: a, b, c
    END SUBROUTINE KERNEL
  END INTERFACE

  INTEGER, INTENT(INOUT) :: a, b, c

  CALL kernel(a, b ,c)
END SUBROUTINE driver
    """.strip()

    kernel_fcode = """
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
    """.strip()

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')

    if use_scheduler:
        (tempdir/'kernel.F90').write_text(kernel_fcode)
        (tempdir/'driver.F90').write_text(driver_fcode)
        scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)
        scheduler.process(transformation, use_file_graph=True)

        kernel = scheduler['#kernel'].source
        driver = scheduler['#driver'].source

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel')
        driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    if frontend == OMNI:
        assert imports[0].module == 'kernel_test_mod'
        assert 'kernel_test' in [str(s) for s in imports[0].symbols]
    else:
        assert imports[0].module == 'KERNEL_test_mod'
        assert 'KERNEL_test' in [str(s) for s in imports[0].symbols]

    # Check that the newly generated USE statement appears before IMPLICIT NONE
    nodes = FindNodes((Intrinsic, Import)).visit(driver['driver'].spec)
    assert len(nodes) == 2
    assert isinstance(nodes[1], Intrinsic)
    assert nodes[1].text.lower() == 'implicit none'


@pytest.mark.parametrize('frontend', available_frontends(
                         xfail=[(OFP, 'OFP does not correctly handle result variable declaration.')]))
def test_dependency_transformation_inline_call(frontend):
    """
    Test injection of suffixed kernel, accessed through inline function call.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTERFACE
    INTEGER FUNCTION kernel(a)
      INTEGER, INTENT(IN) :: a
    END FUNCTION kernel
  END INTERFACE

  INTEGER, INTENT(INOUT) :: a, b, c

  a = kernel(a)
  b = kernel(a)
  c = kernel(c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
INTEGER FUNCTION kernel(a)
  INTEGER, INTENT(IN) :: a

  kernel = 2*a
END FUNCTION kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    kernel.apply(transformation, role='kernel')
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel['kernel_test'].is_function
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the return name has been added as a variable
    assert 'kernel_test' in kernel['kernel_test'].variables

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = tuple(FindInlineCalls().visit(driver['driver'].body))
    assert len(calls) == 2
    calls = tuple(FindInlineCalls(unique=False).visit(driver['driver'].body))
    assert len(calls) == 3
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    assert imports[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in imports[0].symbols]


@pytest.mark.parametrize('frontend', available_frontends(
                         xfail=[(OFP, 'OFP does not correctly handle result variable declaration.')]))
def test_dependency_transformation_inline_call_result_var(frontend):
    """
    Test injection of suffixed kernel, accessed through inline function call.
    """

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTERFACE
    FUNCTION kernel(a) RESULT(ret)
      INTEGER, INTENT(IN) :: a
      INTEGER :: ret
    END FUNCTION kernel
  END INTERFACE

  INTEGER, INTENT(INOUT) :: a, b, c

  a = kernel(a)
  b = kernel(a)
  c = kernel(c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
FUNCTION kernel(a) RESULT(ret)
  INTEGER, INTENT(IN) :: a
  INTEGER :: ret

  ret = 2*a
END FUNCTION kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    kernel.apply(transformation, role='kernel')
    driver['driver'].apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel['kernel_test'].is_function
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = tuple(FindInlineCalls().visit(driver['driver'].body))
    assert len(calls) == 2
    calls = tuple(FindInlineCalls(unique=False).visit(driver['driver'].body))
    assert len(calls) == 3
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    assert imports[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in imports[0].symbols]


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('use_scheduler', [False, True])
def test_dependency_transformation_contained_member(frontend, use_scheduler, tempdir, config):
    """
    The scheduler currently does not recognize or allow processing contained member routines as part
    of the scheduler graph traversal. This test ensures that the transformation class
    does not recurse into contained members.
    """

    kernel_fcode = """
MODULE kernel_mod
    IMPLICIT NONE
CONTAINS
    SUBROUTINE kernel(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    call set_a(1)
    b = get_b()
    c = 3

    CONTAINS

        SUBROUTINE SET_A(VAL)
            INTEGER, INTENT(IN) :: VAL
            A = VAL
        END SUBROUTINE SET_A

        FUNCTION GET_B()
            INTEGER GET_B
            GET_B = 2
        END FUNCTION GET_B
  END SUBROUTINE kernel
END MODULE kernel_mod
    """.strip()

    driver_fcode = """
SUBROUTINE driver(a, b, c)
    USE kernel_mod, only: kernel
    IMPLICIT NONE
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
END SUBROUTINE driver
    """.strip()

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')

    if use_scheduler:
        (tempdir/'kernel_mod.F90').write_text(kernel_fcode)
        (tempdir/'driver.F90').write_text(driver_fcode)
        scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)
        scheduler.process(transformation, use_file_graph=True)

        kernel = scheduler['kernel_mod#kernel'].source
        driver = scheduler['#driver'].source
    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel', targets=('set_a', 'get_b'))
        driver['driver'].apply(transformation, role='driver', targets=('kernel', 'some_const'))

    # Check that calls and matching import have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'
    imports = FindNodes(Import).visit(driver['driver'].spec)
    assert len(imports) == 1
    assert imports[0].module.lower() == 'kernel_test_mod'
    assert imports[0].symbols == ('kernel_test',)

    # Check that the kernel has been renamed
    assert kernel.modules[0].name.lower() == 'kernel_test_mod'
    assert kernel.modules[0].subroutines[0].name.lower() == 'kernel_test'

    # Check if contained member has been renamed
    assert kernel['kernel_test'].subroutines[0].name.lower() == 'set_a'
    assert kernel['kernel_test'].subroutines[1].name.lower() == 'get_b'

    # Check if kernel calls have been renamed
    calls = FindNodes(CallStatement).visit(kernel['kernel_test'].body)
    assert len(calls) == 1
    assert calls[0].name == 'set_a'

    calls = FindInlineCalls(unique=False).visit(kernel['kernel_test'].body)
    assert len(calls) == 1
    assert calls[0].name == 'get_b'


@pytest.mark.parametrize('frontend', available_frontends(
                         xfail=[(OFP, 'OFP does not correctly handle result variable declaration.')]))
def test_dependency_transformation_item_filter(frontend, tempdir, config):
    """
    Test that injection is not applied to modules that have no procedures
    in the scheduler graph, even if they have other item members.
    """

    driver_fcode = """
SUBROUTINE driver(a, b, c)
  USE HEADER_MOD, ONLY: HEADER_VAR
  USE KERNEL_MOD, ONLY: KERNEL
  IMPLICIT NONE

  INTEGER, INTENT(INOUT) :: a, b, c

  a = kernel(a)
  b = kernel(a)
  c = kernel(c) + HEADER_VAR
END SUBROUTINE driver
    """.strip()

    kernel_fcode = """
MODULE kernel_mod
IMPLICIT NONE
CONTAINS
FUNCTION kernel(a) RESULT(ret)
  INTEGER, INTENT(IN) :: a
  INTEGER :: ret

  ret = 2*a
END FUNCTION kernel
END MODULE kernel_mod
    """.strip()

    header_fcode = """
MODULE header_mod
    IMPLICIT NONE
    INTEGER :: HEADER_VAR
END MODULE header_mod
    """.strip()

    (tempdir/'kernel_mod.F90').write_text(kernel_fcode)
    (tempdir/'header_mod.F90').write_text(header_fcode)
    (tempdir/'driver.F90').write_text(driver_fcode)

    # Create the scheduler such that it chases imports
    config['default']['enable_imports'] = True
    scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)

    # Make sure the header var item exists
    assert 'header_mod#header_var' in scheduler.items

    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    scheduler.process(transformation, use_file_graph=True)

    kernel = scheduler['kernel_mod#kernel'].source
    header = scheduler['header_mod#header_var'].source
    driver = scheduler['#driver'].source

    # Check that the kernel mod has been changed
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel['kernel_test'].is_function
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the header name has not been changed
    assert len(header.modules) == 1
    assert header.modules[0].name == 'header_mod'
    assert header.modules[0].variables == ('header_var',)

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    calls = tuple(FindInlineCalls().visit(driver['driver'].body))
    assert len(calls) == 2
    calls = tuple(FindInlineCalls(unique=False).visit(driver['driver'].body))
    assert len(calls) == 3
    assert all(call.name == 'kernel_test' for call in calls)
    imports = FindNodes(Import).visit(driver['driver'].spec)
    imports = driver['driver'].import_map
    assert len(imports) == 2
    assert 'header_var' in imports and imports['header_var'].module.lower() == 'header_mod'
    assert 'kernel_test' in imports and imports['kernel_test'].module.lower() == 'kernel_test_mod'
