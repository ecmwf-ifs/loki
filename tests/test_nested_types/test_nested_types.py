from pathlib import Path
import pytest

from loki import OFP, OMNI, FP, SourceFile, fexprgen

@pytest.mark.parametrize('frontend', [OFP, FP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='Loki annotations break frontend parser'))
])
def test_nested_types(frontend):
    """
    Regression test that ensures that nested types are correctly
    propagated through manual construction from source files.
    """
    here = Path(__file__).parent

    # Check dimension annotation is honoured in sub_type
    types = SourceFile.from_file(here/'types.f90', frontend=frontend)['types']
    child = types.typedefs['sub_type']
    assert fexprgen(child.variables[0].shape) == '(size,)'

    # Check that dimension in sub_type has propagated to parent_type
    parent = types.typedefs['parent_type']
    x = parent.variables[1].type.variables['x']
    assert fexprgen(x.shape) == '(size,)'

    # Ensure that the driver has the correct shape info for pt%type_member%x
    driver = SourceFile.from_file(here/'driver.f90', typedefs=types.typedefs,
                                  frontend=frontend)['driver']
    pt_d = driver.routines[0].variables[0]
    x_d = pt_d.type.variables['type_member'].type.variables['x']
    assert fexprgen(x_d.shape) == '(size,)'
    

    kernel = SourceFile.from_file(here/'kernel.f90', typedefs=types.typedefs,
                                  frontend=frontend)['kernel']
    pt_k = kernel.routines[0].variables[1]
    x_k = pt_k.type.variables['type_member'].type.variables['x']
    assert fexprgen(x_k.shape) == '(size,)'
