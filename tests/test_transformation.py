import pytest

from loki import OFP, OMNI, FP, SourceFile
from loki.transform import Transformation

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
