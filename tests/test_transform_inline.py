# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, jit_compile_lib, available_frontends
from loki import (
    Builder, Module, Subroutine, FindNodes, Import, FindVariables,
    CallStatement, Loop
)
from loki.ir import Assignment
from loki.transform import (
    inline_elemental_functions, inline_constant_parameters,
    replace_selected_kind, inline_member_procedures
)

@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(here):
    return Builder(source_dirs=here, build_dir=here/'build')


@pytest.mark.parametrize('frontend', available_frontends())
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
    refname = f'ref_{routine.name}_{frontend}'
    reference = jit_compile_lib([module, routine], path=here, name=refname, builder=builder)

    v2, v3 = reference.transform_inline_elemental_functions(11.)
    assert v2 == 66.
    assert v3 == 666.

    (here/f'{module.name}.f90').unlink()
    (here/f'{routine.name}.f90').unlink()

    # Now inline elemental functions
    routine = Subroutine.from_source(fcode, definitions=module, frontend=frontend)
    inline_elemental_functions(routine)

    # Verify correct scope of inlined elements
    assert all(v.scope is routine for v in FindVariables().visit(routine.body))

    # Hack: rename routine to use a different filename in the build
    routine.name = f'{routine.name}_'
    kernel = jit_compile_lib([routine], path=here, name=routine.name, builder=builder)

    v2, v3 = kernel.transform_inline_elemental_functions_(11.)
    assert v2 == 66.
    assert v3 == 666.

    builder.clean()
    (here/f'{routine.name}.f90').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
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
    refname = f'ref_{module.name}_{ frontend}'
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)

    v2, v3 = reference.transform_inline_constant_parameters_mod.transform_inline_constant_parameters(10)
    assert v2 == 8
    assert v3 == 2
    (here/f'{module.name}.f90').unlink()
    (here/f'{param_module.name}.f90').unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend)
    assert len(FindNodes(Import).visit(module['transform_inline_constant_parameters'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(Import).visit(module['transform_inline_constant_parameters'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=here, name=f'{module.name}_{frontend}', builder=builder)

    v2, v3 = obj.transform_inline_constant_parameters_mod_.transform_inline_constant_parameters(10)
    assert v2 == 8
    assert v3 == 2

    (here/f'{module.name}.f90').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
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
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)

    v1 = reference.transform_inline_constant_parameters_kind_mod.transform_inline_constant_parameters_kind()
    assert v1 == 5.
    (here/f'{module.name}.f90').unlink()
    (here/f'{param_module.name}.f90').unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend)
    assert len(FindNodes(Import).visit(module['transform_inline_constant_parameters_kind'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(Import).visit(module['transform_inline_constant_parameters_kind'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=here, name=f'{module.name}_{frontend}', builder=builder)

    v1 = obj.transform_inline_constant_parameters_kind_mod_.transform_inline_constant_parameters_kind()
    assert v1 == 5.

    (here/f'{module.name}.f90').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
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
    real(kind=jprb) :: a = 3._JPRB

    v1 = 1._jprb + real(2, kind=jprb) + a
  end subroutine transform_inline_constant_parameters_replace_kind
end module transform_inline_constant_parameters_replace_kind_mod
"""
    # Generate reference code, compile run and verify
    param_module = Module.from_source(fcode_module, frontend=frontend)
    module = Module.from_source(fcode, frontend=frontend)
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)
    func = getattr(getattr(reference, module.name), module.subroutines[0].name)

    v1 = func()
    assert v1 == 6.
    (here/f'{module.name}.f90').unlink()
    (here/f'{param_module.name}.f90').unlink()

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
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=here, name=f'{module.name}_{frontend}', builder=builder)

    func = getattr(getattr(obj, module.name), module.subroutines[0].name)
    v1 = func()
    assert v1 == 6.

    (here/f'{module.name}.f90').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_replacement_internal(frontend):
    """
    Test constant replacement for internally defined constants.
    """
    fcode = """
subroutine kernel(a, b)
  integer, parameter :: par = 10
  integer, intent(inout) :: a, b

  a = b + par
end subroutine kernel
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    inline_constant_parameters(routine=routine, external_only=False)

    assert len(routine.variables) == 2
    assert 'a' in routine.variables and 'b' in routine.variables

    stmts = FindNodes(Assignment).visit(routine.body)
    assert len(stmts) == 1
    assert stmts[0].rhs == 'b + 10'


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines(here, frontend):
    """
    Test inlining of member subroutines.
    """
    fcode = """
subroutine member_routines(a, b)
  real(kind=8), intent(inout) :: a(3), b(3)
  integer :: i

  do i=1, size(a)
    call add_one(a(i))
  end do

  call add_to_a(b)

  do i=1, size(a)
    call add_one(a(i))
  end do

  contains

    subroutine add_one(a)
      real(kind=8), intent(inout) :: a
      a = a + 1
    end subroutine

    subroutine add_to_a(b)
      real(kind=8), intent(inout) :: b(:)
      integer :: n

      n = size(a)
      do i = 1, n
        a(i) = a(i) + b(i)
      end do
    end subroutine
end subroutine member_routines
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    filepath = here/(f'ref_transform_inline_member_routines_{frontend}.f90')
    reference = jit_compile(routine, filepath=filepath, objname='member_routines')

    a = np.array([1., 2., 3.], order='F')
    b = np.array([3., 3., 3.], order='F')
    reference(a, b)

    assert (a == [6., 7., 8.]).all()
    assert (b == [3., 3., 3.]).all()

    # Now inline the member routines and check again
    inline_member_procedures(routine=routine)

    assert not routine.members
    assert not FindNodes(CallStatement).visit(routine.body)
    assert len(FindNodes(Loop).visit(routine.body)) == 3
    assert 'n' in routine.variables

    # An verify compiled behaviour
    filepath = here/(f'transform_inline_member_routines_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='member_routines')

    a = np.array([1., 2., 3.], order='F')
    b = np.array([3., 3., 3.], order='F')
    function(a, b)

    assert (a == [6., 7., 8.]).all()
    assert (b == [3., 3., 3.]).all()
