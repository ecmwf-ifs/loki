from pathlib import Path
from shutil import rmtree
import pytest

from loki import (
    gettempdir, Sourcefile, CallStatement, Import, Interface,
    FindNodes, FindInlineCalls, Intrinsic, Scheduler, SchedulerConfig
)
from loki.frontend import available_frontends, OMNI, OFP
from loki.transform import DependencyTransformation, ModuleWrapTransformation


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
        'routines': {
            'driver': {'role': 'driver'},
            # 'driver_mod': {'role': 'driver'}
        }
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
    IMPLICIT NONE
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
        scheduler.process(transformation)

        # Check that both, old and new module exist now in the scheduler graph
        assert 'kernel_test_mod#kernel_test' in scheduler.items  # for the subroutine
        assert 'kernel_mod' in scheduler.items  # for the global variable

        kernel = scheduler['kernel_test_mod#kernel_test'].source
        driver = scheduler['#driver'].source

        # Check that the not-renamed module is indeed the original one
        scheduler.item_factory.item_cache[str(tempdir/'kernel_mod.F90')].source.make_complete(frontend=frontend)
        assert (
            Sourcefile.from_source(kernel_fcode, frontend=frontend).to_fortran() ==
            scheduler.item_factory.item_cache[str(tempdir/'kernel_mod.F90')].source.to_fortran()
        )

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel')
        driver['driver'].apply(transformation, role='driver', targets=('kernel', 'kernel_mod'))

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
        scheduler.process(transformation)

        kernel = scheduler['kernel_test_mod#kernel_test'].source
        driver = scheduler['driver_mod#driver'].source

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel')
        driver.apply(transformation, role='driver', targets=('kernel', 'kernel_mod'))

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

#include "myfunc.intfb.h"
#include "myfunc.func.h"

  CALL myfunc(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
SUBROUTINE myfunc(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE myfunc
""", frontend=frontend)

    # Ensure header file does not exist a-priori
    header_file = here/'myfunc_test.intfb.h'
    if header_file.exists():
        header_file.unlink()

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', include_path=here)
    kernel['myfunc'].apply(transformation, role='kernel')
    driver['driver'].apply(transformation, role='driver', targets='myfunc')

    # Check that the subroutine name in the kernel source has changed
    assert len(kernel.modules) == 0
    assert len(kernel.subroutines) == 1
    assert kernel.subroutines[0].name == 'myfunc_test'
    assert kernel['myfunc_test'] == kernel.all_subroutines[0]

    # Check that the driver name has not changed
    assert len(kernel.modules) == 0
    assert len(kernel.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that the import has been updated
    assert '#include "myfunc.intfb.h"' not in driver.to_fortran()
    assert '#include "myfunc_test.intfb.h"' in driver.to_fortran()

    # Check that imported function was not modified
    assert '#include "myfunc.func.h"' in driver.to_fortran()

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

#include "kernel.func.h"
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

    transformations = (
        ModuleWrapTransformation(module_suffix='_mod'),
        DependencyTransformation(suffix='_test', module_suffix='_mod')
    )

    if use_scheduler:
        (tempdir/'kernel.F90').write_text(kernel_fcode)
        (tempdir/'driver.F90').write_text(driver_fcode)
        scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)
        for transformation in transformations:
            scheduler.process(transformation)

        kernel = scheduler['kernel_test_mod#kernel_test'].source
        driver = scheduler['#driver'].source

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformations[0], role='kernel')
        driver['driver'].apply(transformations[0], role='driver', targets=('kernel',))
        kernel.apply(transformations[1], role='kernel')
        driver['driver'].apply(transformations[1], role='driver', targets=('kernel_mod', 'kernel'))

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
    imports = FindNodes(Import).visit(driver['driver'].ir)
    assert len(imports) == 2
    assert imports[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in imports[0].symbols]
    assert imports[1].module == 'kernel.func.h'


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('use_scheduler', [False, True])
@pytest.mark.parametrize('module_wrap', [True, False])
def test_dependency_transformation_replace_interface(frontend, use_scheduler, module_wrap, tempdir, config):
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
    transformations = []
    if module_wrap:
        transformations += [ModuleWrapTransformation(module_suffix='_mod')]
    transformations += [DependencyTransformation(suffix='_test', include_path=tempdir, module_suffix='_mod')]

    if use_scheduler:
        (tempdir/'kernel.F90').write_text(kernel_fcode)
        (tempdir/'driver.F90').write_text(driver_fcode)
        scheduler = Scheduler(paths=[tempdir], config=SchedulerConfig.from_dict(config), frontend=frontend)
        for transformation in transformations:
            scheduler.process(transformation)

        if module_wrap:
            kernel = scheduler['kernel_test_mod#kernel_test'].source
        else:
            kernel = scheduler['#kernel_test'].source
        driver = scheduler['#driver'].source

    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        targets = ('kernel',)
        for transformation in transformations:
            kernel.apply(transformation, role='kernel')
            driver.apply(transformation, role='driver', targets=targets)
            # The import becomes another target after the ModuleWrapTransformation
            targets += ('kernel_mod',)

    # Check that the kernel has been wrapped
    if module_wrap:
        assert len(kernel.subroutines) == 0
        assert len(kernel.all_subroutines) == 1
        assert len(kernel.modules) == 1
        assert kernel.modules[0].name == 'kernel_test_mod'
        assert kernel['kernel_test_mod'] == kernel.modules[0]
    else:
        assert len(kernel.subroutines) == 1
        assert len(kernel.modules) == 0
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]

    # Check that the driver name has not changed
    assert len(driver.modules) == 0
    assert len(driver.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls have been diverted to the re-generated routine
    calls = FindNodes(CallStatement).visit(driver['driver'].body)
    assert len(calls) == 1
    assert calls[0].name == 'kernel_test'

    if module_wrap:
        # Check that imports have been generated
        imports = FindNodes(Import).visit(driver['driver'].spec)
        assert len(imports) == 1
        assert imports[0].module.lower() == 'kernel_test_mod'
        assert 'kernel_test' in imports[0].symbols

        # Check that the newly generated USE statement appears before IMPLICIT NONE
        nodes = FindNodes((Intrinsic, Import)).visit(driver['driver'].spec)
        assert len(nodes) == 2
        assert isinstance(nodes[1], Intrinsic)
        assert nodes[1].text.lower() == 'implicit none'

    else:
        # Check that the interface has been updated
        intfs = FindNodes(Interface).visit(driver['driver'].spec)
        assert len(intfs) == 1
        assert intfs[0].symbols == ('kernel_test',)


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
    transformations = (
        ModuleWrapTransformation(module_suffix='_mod'),
        DependencyTransformation(suffix='_test', module_suffix='_mod')
    )
    targets = ('kernel',)
    for transformation in transformations:
        kernel.apply(transformation, role='kernel')
        driver.apply(transformation, role='driver', targets=targets)
        # The import becomes another target after the ModuleWrapTransformation
        targets += ('kernel_mod',)

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 0
    assert len(kernel.all_subroutines) == 1
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel['kernel_test'].is_function
    assert len(kernel.modules) == 1
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the return name hasn't changed
    assert 'kernel' in kernel['kernel_test'].variables
    assert kernel['kernel_test'].result_name == 'kernel'

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
    transformations = (
        ModuleWrapTransformation(module_suffix='_mod'),
        DependencyTransformation(suffix='_test', module_suffix='_mod')
    )
    targets = ('kernel',)
    for transformation in transformations:
        kernel.apply(transformation, role='kernel')
        driver.apply(transformation, role='driver', targets=targets)
        # The import becomes another target after the ModuleWrapTransformation
        targets += ('kernel_mod',)

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
        scheduler.process(transformation)

        kernel = scheduler['kernel_test_mod#kernel_test'].source
        driver = scheduler['#driver'].source
    else:
        kernel = Sourcefile.from_source(kernel_fcode, frontend=frontend)
        driver = Sourcefile.from_source(driver_fcode, frontend=frontend)

        kernel.apply(transformation, role='kernel', targets=('set_a', 'get_b'))
        driver['driver'].apply(transformation, role='driver', targets=('kernel', 'kernel_mod'))

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

    # Make sure the header module item exists
    assert 'header_mod' in scheduler.items

    transformations = (
        ModuleWrapTransformation(module_suffix='_mod'),
        DependencyTransformation(suffix='_test', module_suffix='_mod')
    )
    for transformation in transformations:
        scheduler.process(transformation)

    kernel = scheduler['kernel_test_mod#kernel_test'].source
    header = scheduler['header_mod'].source
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


@pytest.mark.parametrize('frontend', available_frontends())
def test_dependency_transformation_filter_items_file_graph(frontend, config):
    """
    Ensure that the ``items`` list given to a transformation in
    a file graph traversal is filtered to include only used items
    """
    fcode = """
module test_dependency_transformation_filter_items1_mod
implicit none
contains
subroutine proc1(arg)
    integer, intent(inout) :: arg
    arg = arg + 1
end subroutine proc1

subroutine unused_proc(arg)
    integer, intent(inout) :: arg
    arg = arg - 1
end subroutine unused_proc
end module test_dependency_transformation_filter_items1_mod

module test_dependency_transformation_filter_items2_mod
implicit none
contains
subroutine proc2(arg)
    integer, intent(inout) :: arg
    arg = arg + 2
end subroutine proc2
end module test_dependency_transformation_filter_items2_mod

module test_dependency_transformation_filter_items3_mod
implicit none
integer, parameter :: param3 = 3
contains
subroutine proc3(arg)
    integer, intent(inout) :: arg
    arg = arg + 3
end subroutine proc3
end module test_dependency_transformation_filter_items3_mod

subroutine test_dependency_transformation_filter_items_driver
use test_dependency_transformation_filter_items1_mod, only: proc1
use test_dependency_transformation_filter_items3_mod, only: param3
implicit none
integer :: i
i = param3
call proc1(i)
end subroutine test_dependency_transformation_filter_items_driver
    """

    config['routines'] = {
        'test_dependency_transformation_filter_items_driver': {'role': 'driver'},
    }

    workdir = gettempdir()/'test_dependency_transformation_filter_items'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    filepath = workdir/'test_dependency_transformation_filter_items.F90'
    filepath.write_text(fcode)

    scheduler = Scheduler(
        paths=[workdir], config=config,
        seed_routines=['test_dependency_transformation_filter_items_driver'],
        frontend=frontend
    )

    # Only the driver and mod1 are in the Sgraph
    expected_dependencies = {
        '#test_dependency_transformation_filter_items_driver': {
            'test_dependency_transformation_filter_items1_mod#proc1',
            'test_dependency_transformation_filter_items3_mod'
        },
        'test_dependency_transformation_filter_items1_mod#proc1': set(),
        'test_dependency_transformation_filter_items3_mod': set()
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }

    # The other module and procedure are in the item_factory's cache...
    assert 'test_dependency_transformation_filter_items2_mod' in scheduler.item_factory.item_cache
    assert 'test_dependency_transformation_filter_items1_mod#unused_proc' in scheduler.item_factory.item_cache

    # ...and share the same sourcefile object
    assert (
        scheduler.item_factory.item_cache['test_dependency_transformation_filter_items2_mod'].source is
        scheduler.item_factory.item_cache['test_dependency_transformation_filter_items1_mod'].source
    )

    # The filegraph consists of the single file
    filegraph = scheduler.file_graph
    assert filegraph.items == (str(filepath).lower(),)

    # Check that the DependencyTransformation changes only the active items
    # and discards unused routines
    scheduler.process(transformation=DependencyTransformation(suffix='_foo', module_suffix='_mod'))

    expected_dependencies = {
        '#test_dependency_transformation_filter_items_driver': {
            'test_dependency_transformation_filter_items1_foo_mod#proc1_foo',
            'test_dependency_transformation_filter_items3_mod'
        },
        'test_dependency_transformation_filter_items1_foo_mod#proc1_foo': set(),
        'test_dependency_transformation_filter_items3_mod': set()
    }

    assert set(scheduler.items) == set(expected_dependencies)
    assert set(scheduler.dependencies) == {
        (a, b) for a, deps in expected_dependencies.items() for b in deps
    }


    # The other module is still in the item_factory's cache...
    assert 'test_dependency_transformation_filter_items2_mod' in scheduler.item_factory.item_cache

    # ...and so are the original modules
    assert 'test_dependency_transformation_filter_items1_mod' in scheduler.item_factory.item_cache
    assert 'test_dependency_transformation_filter_items3_mod' in scheduler.item_factory.item_cache

    # ...but they don't share the same sourcefile object anymore
    original_source = scheduler.item_factory.item_cache['test_dependency_transformation_filter_items2_mod'].source
    new_src = scheduler.item_factory.item_cache['test_dependency_transformation_filter_items1_foo_mod'].source
    assert new_src is not original_source

    # The new source does not contain the unused module
    assert [m.name.lower() for m in original_source.modules] == [
        'test_dependency_transformation_filter_items1_mod',
        'test_dependency_transformation_filter_items2_mod',
        'test_dependency_transformation_filter_items3_mod'
    ]
    assert [m.name.lower() for m in new_src.modules] == [
        'test_dependency_transformation_filter_items1_foo_mod',
        'test_dependency_transformation_filter_items3_mod'
    ]
    # Note the idiosyncratic behaviour:
    # items3_mod appears twice because the name is not updated but it is part of the
    # scheduler graph. We need to see whether this is what we want...

    # The new module does not contain the unused procedure
    original_mod1 = original_source['test_dependency_transformation_filter_items1_mod']
    new_mod1 = new_src['test_dependency_transformation_filter_items1_foo_mod']

    assert [r.name.lower() for r in original_mod1.subroutines] == ['proc1', 'unused_proc']
    assert [r.name.lower() for r in new_mod1.subroutines] == ['proc1_foo']

    rmtree(workdir)
