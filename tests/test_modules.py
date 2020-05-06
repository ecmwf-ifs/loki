from pathlib import Path
import pytest

from loki import OFP, OMNI, FP, Module, Declaration, TypeDef


@pytest.mark.parametrize('frontend', [FP, OFP, OMNI])
def test_module_from_source(frontend):
    """
    Test the creation of `Module` objects from raw source strings.
    """
    fcode = """
module a_module
  integer, parameter :: x = 2
  integer, parameter :: y = 3

  type derived_type
    real :: array(x, y)
  end type derived_type
contains

  subroutine my_routine(pt)
    type(derived_type) :: pt
    pt%array(:,:) = 42.0
  end subroutine my_routine
end module a_module
"""
    module = Module.from_source(fcode, frontend=frontend)
    assert len([o for o in module.spec.body if isinstance(o, Declaration)]) == 2
    assert len([o for o in module.spec.body if isinstance(o, TypeDef)]) == 1
    assert 'derived_type' in module.typedefs
    assert len(module.routines) == 1
    assert module.routines[0].name == 'my_routine'
