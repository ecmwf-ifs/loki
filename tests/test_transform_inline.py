from pathlib import Path
import pytest

from conftest import jit_compile_lib
from loki import Builder, Module, Subroutine, OFP, OMNI, FP, FindNodes, Import
from loki.transform import inline_elemental_functions, inline_constant_parameters, replace_selected_kind


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(here):
    return Builder(source_dirs=here, build_dir=here/'build')


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_inline_elemental_functions(here, builder, frontend):
    """
    Test correct inlining of elemental functions.
    """
    fcode_module = """
module multiply_mod
  use iso_fortran_env, only: real64
  implicit none
contains

  elemental function multiply(a, b)
    real(kind=real64) :: multiply
    real(kind=real64), intent(in) :: a, b

    multiply = a * b
  end function multiply
end module multiply_mod
"""

    fcode = """
subroutine transform_inline_elemental_functions(v1, v2, v3)
  use iso_fortran_env, only: real64
  use multiply_mod, only: multiply
  real(kind=real64), intent(in) :: v1
  real(kind=real64), intent(out) :: v2, v3

  v2 = multiply(v1, 6._real64)
  v3 = 600. + multiply(6._real64, 11._real64)
end subroutine transform_inline_elemental_functions
"""
    # Generate reference code, compile run and verify
    module = Module.from_source(fcode_module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    refname = 'ref_%s_%s' % (routine.name, frontend)
    reference = jit_compile_lib([module, routine], path=here, name=refname, builder=builder)

    v2, v3 = reference.transform_inline_elemental_functions(11.)
    assert v2 == 66.
    assert v3 == 666.

    (here/'{}.f90'.format(module.name)).unlink()
    (here/'{}.f90'.format(routine.name)).unlink()

    # Now inline elemental functions
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)
    inline_elemental_functions(routine)

    # Hack: rename routine to use a different filename in the build
    routine.name = '%s_' % routine.name
    kernel = jit_compile_lib([routine], path=here, name=routine.name, builder=builder)

    v2, v3 = kernel.transform_inline_elemental_functions_(11.)
    assert v2 == 66.
    assert v3 == 666.

    builder.clean()
    (here/'{}.f90'.format(routine.name)).unlink()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_inline_constant_parameters(here, builder, frontend):
    """
    Test correct inlining of constant parameters.
    """
    fcode_module = """
module parameters_mod
  implicit none
  integer, parameter :: a = 1
  integer, parameter :: b = -1
contains
  subroutine dummy
  end subroutine dummy
end module parameters_mod
"""

    fcode = """
module transform_inline_constant_parameters_mod
  ! TODO: use parameters_mod, only: b
  implicit none
  integer, parameter :: c = 1+1
contains
  subroutine transform_inline_constant_parameters(v1, v2, v3)
    use parameters_mod, only: a, b
    integer, intent(in) :: v1
    integer, intent(out) :: v2, v3

    v2 = v1 + b - a
    v3 = c
  end subroutine transform_inline_constant_parameters
end module transform_inline_constant_parameters_mod
"""
    # Generate reference code, compile run and verify
    param_module = Module.from_source(fcode_module, frontend=frontend)
    module = Module.from_source(fcode, frontend=frontend)
    refname = 'ref_%s_%s' % (module.name, frontend)
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)

    v2, v3 = reference.transform_inline_constant_parameters_mod.transform_inline_constant_parameters(10)
    assert v2 == 8
    assert v3 == 2
    (here/'{}.f90'.format(module.name)).unlink()
    (here/'{}.f90'.format(param_module.name)).unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend)
    assert len(FindNodes(Import).visit(module['transform_inline_constant_parameters'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(Import).visit(module['transform_inline_constant_parameters'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = '%s_' % module.name
    obj = jit_compile_lib([module], path=here, name='%s_%s' % (module.name, frontend), builder=builder)

    v2, v3 = obj.transform_inline_constant_parameters_mod_.transform_inline_constant_parameters(10)
    assert v2 == 8
    assert v3 == 2

    (here/'{}.f90'.format(module.name)).unlink()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_inline_constant_parameters_kind(here, builder, frontend):
    """
    Test correct inlining of constant parameters for kind symbols.
    """
    fcode_module = """
module kind_parameters_mod
  implicit none
  integer, parameter :: jprb = selected_real_kind(13, 300)
end module kind_parameters_mod
"""

    fcode = """
module transform_inline_constant_parameters_kind_mod
  implicit none
contains
  subroutine transform_inline_constant_parameters_kind(v1)
    use kind_parameters_mod, only: jprb
    real(kind=jprb), intent(out) :: v1

    v1 = real(2, kind=jprb) + 3.
  end subroutine transform_inline_constant_parameters_kind
end module transform_inline_constant_parameters_kind_mod
"""
    # Generate reference code, compile run and verify
    param_module = Module.from_source(fcode_module, frontend=frontend)
    module = Module.from_source(fcode, frontend=frontend)
    refname = 'ref_%s_%s' % (module.name, frontend)
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)

    v1 = reference.transform_inline_constant_parameters_kind_mod.transform_inline_constant_parameters_kind()
    assert v1 == 5.
    (here/'{}.f90'.format(module.name)).unlink()
    (here/'{}.f90'.format(param_module.name)).unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend)
    assert len(FindNodes(Import).visit(module['transform_inline_constant_parameters_kind'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(Import).visit(module['transform_inline_constant_parameters_kind'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = '%s_' % module.name
    obj = jit_compile_lib([module], path=here, name='%s_%s' % (module.name, frontend), builder=builder)

    v1 = obj.transform_inline_constant_parameters_kind_mod_.transform_inline_constant_parameters_kind()
    assert v1 == 5.

    (here/'{}.f90'.format(module.name)).unlink()


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_transform_inline_constant_parameters_replace_kind(here, builder, frontend):
    """
    Test correct inlining of constant parameters for kind symbols.
    """
    fcode_module = """
module replace_kind_parameters_mod
  implicit none
  integer, parameter :: jprb = selected_real_kind(13, 300)
end module replace_kind_parameters_mod
"""

    fcode = """
module transform_inline_constant_parameters_replace_kind_mod
  implicit none
contains
  subroutine transform_inline_constant_parameters_replace_kind(v1)
    use replace_kind_parameters_mod, only: jprb
    real(kind=jprb), intent(out) :: v1

    v1 = 1._jprb + real(2, kind=jprb) + 3.
  end subroutine transform_inline_constant_parameters_replace_kind
end module transform_inline_constant_parameters_replace_kind_mod
"""
    # Generate reference code, compile run and verify
    param_module = Module.from_source(fcode_module, frontend=frontend)
    module = Module.from_source(fcode, frontend=frontend)
    refname = 'ref_%s_%s' % (module.name, frontend)
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)
    func = getattr(getattr(reference, module.name), module.subroutines[0].name)

    v1 = func()
    assert v1 == 6.
    (here/'{}.f90'.format(module.name)).unlink()
    (here/'{}.f90'.format(param_module.name)).unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend)
    imports = FindNodes(Import).visit(module.subroutines[0].spec)
    assert len(imports) == 1 and imports[0].module.lower() == param_module.name.lower()
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
        replace_selected_kind(routine)
    imports = FindNodes(Import).visit(module.subroutines[0].spec)
    assert len(imports) == 1 and imports[0].module.lower() == 'iso_fortran_env'

    # Hack: rename module to use a different filename in the build
    module.name = '%s_' % module.name
    obj = jit_compile_lib([module], path=here, name='%s_%s' % (module.name, frontend), builder=builder)

    func = getattr(getattr(obj, module.name), module.subroutines[0].name)
    v1 = func()
    assert v1 == 6.

    (here/'{}.f90'.format(module.name)).unlink()
