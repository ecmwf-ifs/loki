import pytest
from pathlib import Path

from loki import OFP, OMNI, FP, SourceFile, CallStatement, Import
from loki.transform import Transformation, DependencyTransformation


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='rename_transform')
def fixture_rename_transform():

    class RenameTransform(Transformation):
        """
        Simple `Transformation` object that renames subroutine and modules.
        """

        def transform_subroutine(self, routine, **kwargs):
            routine.name += '_test'

        def transform_module(self, module, **kwargs):
            module.name += '_test'

    return RenameTransform()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transformation_apply(rename_transform, frontend):
    """
    Apply a simple transformation that renames routines and modules
    """
    fcode = """
module mymodule
  real(kind=4) :: myvar
end module mymodule

subroutine myroutine(a, b)
  real(kind=4), intent(inout) :: a, b

  a = a + b
end subroutine myroutine
"""
    # Let source apply transformation to all items and verify
    source = SourceFile.from_source(fcode, frontend=frontend)
    source.apply(rename_transform)
    assert source.modules[0].name == 'mymodule_test'
    assert source['mymodule_test'] == source.modules[0]
    assert source.subroutines[0].name == 'myroutine_test'
    assert source['myroutine_test'] == source.subroutines[0]

    # Apply transformation explicitly to whole source and verify
    source = SourceFile.from_source(fcode, frontend=frontend)
    rename_transform.apply(source)
    assert source.modules[0].name == 'mymodule_test'
    assert source['mymodule_test'] == source.modules[0]
    assert source.subroutines[0].name == 'myroutine_test'
    assert source['myroutine_test'] == source.subroutines[0]


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transformation_apply_subroutine(rename_transform, frontend):
    """
    Apply a simple transformation that renames routines and modules
    """
    fcode = """
module mymodule
  real(kind=4) :: myvar

contains

  subroutine module_routine(argument)
    real(kind=4), intent(inout) :: argument

    argument = member_func()

  contains
    function member_func() result(res)
      real(kind=4) :: res

      res = 4.
    end function member_func
  end subroutine module_routine
end module mymodule

subroutine myroutine(a, b)
  real(kind=4), intent(inout) :: a, b

  a = a + b
end subroutine myroutine
"""
    # Let only the inner module routine apply the transformation
    source = SourceFile.from_source(fcode, frontend=frontend)
    source['module_routine'].apply(rename_transform)
    assert source.modules[0].name == 'mymodule'
    assert source['mymodule'] == source.modules[0]
    assert source.subroutines[0].name == 'myroutine'
    assert source['myroutine'] == source.subroutines[0]
    assert len(source.all_subroutines) == 2  # Ignore member func
    assert source.all_subroutines[1].name == 'module_routine_test'
    assert source['module_routine_test'] == source.all_subroutines[1]
    assert len(source['module_routine_test'].members) == 1
    assert source['module_routine_test'].members[0].name == 'member_func_test'

    # Apply transformation explicitly to the outer routine
    source = SourceFile.from_source(fcode, frontend=frontend)
    rename_transform.apply_subroutine(source['myroutine'])
    assert source.modules[0].name == 'mymodule'
    assert source['mymodule'] == source.modules[0]
    assert source.subroutines[0].name == 'myroutine_test'
    assert source['myroutine_test'] == source.subroutines[0]
    assert len(source.all_subroutines) == 2
    assert source.all_subroutines[1].name == 'module_routine'
    assert source['module_routine'] == source.all_subroutines[1]


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transformation_apply_module(rename_transform, frontend):
    """
    Apply a simple transformation that renames routines and modules
    """
    fcode = """
module mymodule
  real(kind=4) :: myvar

contains

  subroutine module_routine(argument)
    real(kind=4), intent(inout) :: argument

    argument = argument  + 1.
  end subroutine module_routine
end module mymodule

subroutine myroutine(a, b)
  real(kind=4), intent(inout) :: a, b

  a = a + b
end subroutine myroutine
"""
    # Let the module and apply the transformation to everything it contains
    source = SourceFile.from_source(fcode, frontend=frontend)
    source['mymodule'].apply(rename_transform)
    assert source.modules[0].name == 'mymodule_test'
    assert source['mymodule_test'] == source.modules[0]
    assert len(source.all_subroutines) == 2
    # Outer subroutine is untouched, since we apply all
    # transformations to anything in the module.
    assert source.subroutines[0].name == 'myroutine'
    assert source['myroutine'] == source.subroutines[0]
    assert source.all_subroutines[1].name == 'module_routine_test'
    assert source['module_routine_test'] == source.all_subroutines[1]

    # Apply transformation only to modules, not subroutines, in the source
    source = SourceFile.from_source(fcode, frontend=frontend)
    rename_transform.apply_module(source['mymodule'])
    assert source.modules[0].name == 'mymodule_test'
    assert source['mymodule_test'] == source.modules[0]
    assert len(source.all_subroutines) == 2
    assert source.subroutines[0].name == 'myroutine'
    assert source['myroutine'] == source.subroutines[0]
    assert source.all_subroutines[1].name == 'module_routine_test'
    assert source['module_routine_test'] == source.all_subroutines[1]


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_dependency_transformation_module_imports(frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines via module imports.
    """

    kernel = SourceFile.from_source(source="""
MODULE kernel_mod
CONTAINS
    SUBROUTINE kernel(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    a = 1
    b = 2
    c = 3
  END SUBROUTINE kernel
END MODULE kernel_mod
""", frontend=frontend)

    driver = SourceFile.from_source(source="""
MODULE driver_mod
  USE kernel_mod, only: kernel
CONTAINS
  SUBROUTINE driver(a, b, c)
    INTEGER, INTENT(INOUT) :: a, b, c

    CALL kernel(a, b ,c)
  END SUBROUTINE driver
END MODULE driver_mod
""", frontend=frontend)

    transformation = DependencyTransformation(suffix='_test', module_suffix='_mod')
    kernel.apply(transformation, role='kernel')
    driver.apply(transformation, role='driver', targets='kernel')

    # Check that the basic entity names in the kernel source have changed
    assert kernel.all_subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    assert kernel.modules[0].name == 'kernel_test_mod'
    assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the entity names in the driver have not changed
    assert driver.all_subroutines[0].name == 'driver'
    assert driver['driver'] == driver.all_subroutines[0]
    assert driver.modules[0].name == 'driver_mod'
    assert driver['driver_mod'] == driver.modules[0]

    # Check that calls and imports have been diverted to the re-generated routine
    assert isinstance(driver['driver'].body[0], CallStatement)
    assert driver['driver'].body[0].name == 'kernel_test'
    assert isinstance(driver['driver_mod'].spec.body[0], Import)
    assert driver['driver_mod'].spec.body[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in driver['driver_mod'].spec.body[0].symbols]


@pytest.mark.parametrize('frontend', [
    OFP,
    FP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='C-imports need pre-processing for OMNI')),
])
def test_dependency_transformation_header_includes(here, frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines via c-header includes.
    """

    driver = SourceFile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

#include "kernel.intfb.h"

  CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = SourceFile.from_source(source="""
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
    kernel.apply(transformation, role='kernel')
    driver.apply(transformation, role='driver', targets='kernel')

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
    assert '#include "kernel.intfb.h"' not in driver.source
    assert '#include "kernel_test.intfb.h"' in driver.source

    # Check that header file was generated and clean up
    assert header_file.exists()
    header_file.unlink()


@pytest.mark.parametrize('frontend', [
    OFP,
    FP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='C-imports need pre-processing for OMNI')),
])
def test_dependency_transformation_module_wrap(here, frontend):
    """
    Test injection of suffixed kernels into unchanged driver
    routines automatic module wrapping of the kernel.
    """

    driver = SourceFile.from_source(source="""
SUBROUTINE driver(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

#include "kernel.intfb.h"

  CALL kernel(a, b ,c)
END SUBROUTINE driver
""", frontend=frontend)

    kernel = SourceFile.from_source(source="""
SUBROUTINE kernel(a, b, c)
  INTEGER, INTENT(INOUT) :: a, b, c

  a = 1
  b = 2
  c = 3
END SUBROUTINE kernel
""", frontend=frontend)

    # Apply injection transformation via C-style includes by giving `include_path`
    transformation = DependencyTransformation(suffix='_test', mode='module', module_suffix='_mod')
    kernel.apply(transformation, role='kernel')
    driver.apply(transformation, role='driver', targets='kernel')

    # Check that the kernel has been wrapped
    assert len(kernel.subroutines) == 1
    assert kernel.subroutines[0].name == 'kernel_test'
    assert kernel['kernel_test'] == kernel.all_subroutines[0]
    # TODO: Currently broken; needs transformation re-structuring....
    # assert len(kernel.modules) == 1
    # assert kernel.modules[0].name == 'kernel_test_mod'
    # assert kernel['kernel_test_mod'] == kernel.modules[0]

    # Check that the driver name has not changed
    assert len(kernel.modules) == 0
    assert len(kernel.subroutines) == 1
    assert driver.subroutines[0].name == 'driver'

    # Check that calls and imports have been diverted to the re-generated routine
    assert isinstance(driver['driver'].body[0], CallStatement)
    assert driver['driver'].body[0].name == 'kernel_test'
    assert isinstance(driver['driver'].spec.body[0], Import)
    assert driver['driver'].spec.body[0].module == 'kernel_test_mod'
    assert 'kernel_test' in [str(s) for s in driver['driver'].spec.body[0].symbols]
