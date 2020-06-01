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
