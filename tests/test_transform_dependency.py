from pathlib import Path
import pytest

from conftest import available_frontends
from loki import (
    OMNI, OFP, Sourcefile, CallStatement, Import,
    FindNodes, FindInlineCalls, Intrinsic
)
from loki.transform import DependencyTransformation


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends())
def test_dependency_transformation_globalvar_imports(frontend):
    """
    Test that global variable imports are not renamed as a
    call statement would be.
    """

    kernel = Sourcefile.from_source(source="""
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
""", frontend=frontend)

    driver = Sourcefile.from_source(source="""
SUBROUTINE driver(a, b, c)
    USE kernel_mod, only: kernel
    USE kernel_mod, only: some_const
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')
    # Because the renaming is intended to be applied to the routines as well as the enclosing module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
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
def test_dependency_transformation_globalvar_imports_driver_mod(frontend):
    """
    Test that global variable imports are not renamed as a
    call statement would be.
    """

    kernel = Sourcefile.from_source(source="""
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
""", frontend=frontend)

    driver = Sourcefile.from_source(source="""
MODULE DRIVER_MOD
    USE kernel_mod, only: kernel
    USE kernel_mod, only: some_const
CONTAINS
SUBROUTINE driver(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
END SUBROUTINE driver
END MODULE DRIVER_MOD
""", frontend=frontend)

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')
    # Because the renaming is intended to be applied to the routines as well as the enclosing module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
    driver.apply(transformation, role='driver', targets=('kernel', 'some_const'), recurse_to_contained_nodes=True)

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
def test_dependency_transformation_module_wrap(frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines automatic module wrapping of the kernel.
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

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
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
def test_dependency_transformation_replace_interface(frontend):
    """
    Test injection of suffixed kernels defined in interface block
    into unchanged driver routines automatic module wrapping of the kernel.
    """

    driver = Sourcefile.from_source(source="""
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
""", frontend=frontend)

    kernel = Sourcefile.from_source(source="""
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
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
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
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
    # Because the renaming is intended to also wrap the kernel in a module,
    # we need to invoke the transformation on the full source file and activate recursion to contained nodes
    kernel.apply(transformation, role='kernel', recurse_to_contained_nodes=True)
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
