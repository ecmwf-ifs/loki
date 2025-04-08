# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Module, Subroutine
from loki.jit_build import jit_compile_lib, Builder, Obj, jit_compile, clean_test
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes
from loki.expression import symbols as sym, parse_expr

from loki.transformations.inline import inline_constant_parameters
from loki.transformations.utilities import replace_selected_kind


@pytest.fixture(name='builder')
def fixture_builder(tmp_path):
    yield Builder(source_dirs=tmp_path, build_dir=tmp_path)
    Obj.clear_cache()


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_inline_constant_parameters(tmp_path, builder, frontend):
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
module inline_const_param_mod
  ! TODO: use parameters_mod, only: b
  implicit none
  integer, parameter :: c = 1+1
contains
  subroutine inline_const_param(v1, v2, v3)
    use parameters_mod, only: a, b
    integer, intent(in) :: v1
    integer, intent(out) :: v2, v3

    v2 = v1 + b - a
    v3 = c
  end subroutine inline_const_param
end module inline_const_param_mod
"""
    # Generate reference code, compile run and verify
    orig_tmp_path = tmp_path/'orig'
    orig_tmp_path.mkdir()
    param_module = Module.from_source(fcode_module, frontend=frontend, xmods=[orig_tmp_path])
    module = Module.from_source(fcode, frontend=frontend, xmods=[orig_tmp_path])
    refname = f'ref_{module.name}_{ frontend}'
    reference = jit_compile_lib([module, param_module], path=orig_tmp_path, name=refname, builder=builder)

    v2, v3 = reference.inline_const_param_mod.inline_const_param(10)
    assert v2 == 8
    assert v3 == 2

    # Now transform with supplied elementals but without module
    new_tmp_path = tmp_path/'new'
    new_tmp_path.mkdir()
    param_module = Module.from_source(fcode_module, frontend=frontend, xmods=[new_tmp_path])
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend, xmods=[new_tmp_path])
    assert len(FindNodes(ir.Import).visit(module['inline_const_param'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(ir.Import).visit(module['inline_const_param'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=new_tmp_path, name=f'{module.name}_{frontend}', builder=builder)

    v2, v3 = obj.inline_const_param_mod_.inline_const_param(10)
    assert v2 == 8
    assert v3 == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_inline_constant_parameters_kind(tmp_path, builder, frontend):
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
module inline_const_param_kind_mod
  implicit none
contains
  subroutine inline_const_param_kind(v1)
    use kind_parameters_mod, only: jprb
    real(kind=jprb), intent(out) :: v1

    v1 = real(2, kind=jprb) + 3.
  end subroutine inline_const_param_kind
end module inline_const_param_kind_mod
"""
    # Generate reference code, compile run and verify
    param_module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module, param_module], path=tmp_path, name=refname, builder=builder)

    v1 = reference.inline_const_param_kind_mod.inline_const_param_kind()
    assert v1 == 5.
    (tmp_path/f'{module.name}.f90').unlink()
    (tmp_path/f'{param_module.name}.f90').unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend, xmods=[tmp_path])
    assert len(FindNodes(ir.Import).visit(module['inline_const_param_kind'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(ir.Import).visit(module['inline_const_param_kind'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=tmp_path, name=f'{module.name}_{frontend}', builder=builder)

    v1 = obj.inline_const_param_kind_mod_.inline_const_param_kind()
    assert v1 == 5.

    (tmp_path/f'{module.name}.f90').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_transform_inline_constant_parameters_replace_kind(tmp_path, builder, frontend):
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
module inline_param_repl_kind_mod
  implicit none
contains
  subroutine inline_param_repl_kind(v1)
    use replace_kind_parameters_mod, only: jprb
    real(kind=jprb), intent(out) :: v1
    real(kind=jprb) :: a = 3._JPRB

    v1 = 1._jprb + real(2, kind=jprb) + a
  end subroutine inline_param_repl_kind
end module inline_param_repl_kind_mod
"""
    # Generate reference code, compile run and verify
    param_module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module, param_module], path=tmp_path, name=refname, builder=builder)
    func = getattr(getattr(reference, module.name), module.subroutines[0].name)

    v1 = func()
    assert v1 == 6.
    (tmp_path/f'{module.name}.f90').unlink()
    (tmp_path/f'{param_module.name}.f90').unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend, xmods=[tmp_path])
    imports = FindNodes(ir.Import).visit(module.subroutines[0].spec)
    assert len(imports) == 1 and imports[0].module.lower() == param_module.name.lower()
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
        replace_selected_kind(routine)
    imports = FindNodes(ir.Import).visit(module.subroutines[0].spec)
    assert len(imports) == 1 and imports[0].module.lower() == 'iso_fortran_env'

    # Hack: rename module to use a different filename in the build
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=tmp_path, name=f'{module.name}_{frontend}', builder=builder)

    func = getattr(getattr(obj, module.name), module.subroutines[0].name)
    v1 = func()
    assert v1 == 6.

    (tmp_path/f'{module.name}.f90').unlink()


@pytest.mark.parametrize('frontend', available_frontends())
def test_constant_replacement_internal(tmp_path, frontend):
    """
    Test constant replacement for internally defined constants.
    """
    fcode = """
subroutine kernel(a, b)
  integer, parameter :: par = 10
  integer, parameter :: par2 = 0
  integer, intent(inout) :: a, b
  real, parameter :: par_x = 1.3
  real :: x, y
  logical, parameter :: flag1 = .true.

  x = 0.0
  y = 1.4
  if (flag1 .and. par2 .eq. 0) then
    x = y + par_x
  endif

  a = b + par + ceiling(x)
end subroutine kernel
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    filepath = tmp_path/(f'{routine.name}_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname=routine.name)
    # test original function
    a, b = np.array(1, dtype=np.int32), np.array(2, dtype=np.int32)
    function(a=a, b=b)
    assert a == 15

    inline_constant_parameters(routine=routine, external_only=False)

    transf_filepath = tmp_path/(f'{routine.name}_transf_{frontend}.f90')
    transf_function = jit_compile(routine, filepath=transf_filepath, objname=routine.name)
    # test transformed function
    transf_a, transf_b = np.array(1, dtype=np.int32), np.array(2, dtype=np.int32)
    transf_function(a=transf_a, b=transf_b)
    assert transf_a == 15

    # check IR
    assert len(routine.variables) == 4
    assert 'a' in routine.variables
    assert 'b' in routine.variables
    assert 'x' in routine.variables
    assert 'y' in routine.variables

    stmts = FindNodes(ir.Assignment).visit(routine.body)
    assert len(stmts) == 4
    assert stmts[2].rhs == 'y + 1.3'
    assert '10' in stmts[3].rhs

    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    cond = conditionals[0].condition
    assert isinstance(cond, sym.LogicalAnd)
    assert cond.children[0] == parse_expr('.true.')
    assert cond.children[1] == '0 == 0'

    clean_test(filepath)
    clean_test(transf_filepath)
