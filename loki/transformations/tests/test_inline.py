# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from loki import (
    Module, Subroutine, FindVariables, BasicType, DerivedType,
    FindInlineCalls
)
from loki.build import jit_compile, jit_compile_lib, Builder, Obj
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI, OFP
from loki.ir import nodes as ir, FindNodes
from loki.batch import Scheduler, SchedulerConfig

from loki.transformations.inline import (
    inline_elemental_functions, inline_constant_parameters,
    inline_member_procedures, inline_marked_subroutines,
    InlineTransformation,
)
from loki.transformations.sanitise import ResolveAssociatesTransformer
from loki.transformations.utilities import replace_selected_kind


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='builder')
def fixture_builder(here):
    yield Builder(source_dirs=here, build_dir=here/'build')
    Obj.clear_cache()


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

    # Make sure there are no more inline calls in the routine body
    assert not FindInlineCalls().visit(routine.body)

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
    param_module = Module.from_source(fcode_module, frontend=frontend)
    module = Module.from_source(fcode, frontend=frontend)
    refname = f'ref_{module.name}_{ frontend}'
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)

    v2, v3 = reference.inline_const_param_mod.inline_const_param(10)
    assert v2 == 8
    assert v3 == 2
    (here/f'{module.name}.f90').unlink()
    (here/f'{param_module.name}.f90').unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend)
    assert len(FindNodes(ir.Import).visit(module['inline_const_param'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(ir.Import).visit(module['inline_const_param'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=here, name=f'{module.name}_{frontend}', builder=builder)

    v2, v3 = obj.inline_const_param_mod_.inline_const_param(10)
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
    param_module = Module.from_source(fcode_module, frontend=frontend)
    module = Module.from_source(fcode, frontend=frontend)
    refname = f'ref_{module.name}_{frontend}'
    reference = jit_compile_lib([module, param_module], path=here, name=refname, builder=builder)

    v1 = reference.inline_const_param_kind_mod.inline_const_param_kind()
    assert v1 == 5.
    (here/f'{module.name}.f90').unlink()
    (here/f'{param_module.name}.f90').unlink()

    # Now transform with supplied elementals but without module
    module = Module.from_source(fcode, definitions=param_module, frontend=frontend)
    assert len(FindNodes(ir.Import).visit(module['inline_const_param_kind'].spec)) == 1
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
    assert not FindNodes(ir.Import).visit(module['inline_const_param_kind'].spec)

    # Hack: rename module to use a different filename in the build
    module.name = f'{module.name}_'
    obj = jit_compile_lib([module], path=here, name=f'{module.name}_{frontend}', builder=builder)

    v1 = obj.inline_const_param_kind_mod_.inline_const_param_kind()
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
    imports = FindNodes(ir.Import).visit(module.subroutines[0].spec)
    assert len(imports) == 1 and imports[0].module.lower() == param_module.name.lower()
    for routine in module.subroutines:
        inline_constant_parameters(routine, external_only=True)
        replace_selected_kind(routine)
    imports = FindNodes(ir.Import).visit(module.subroutines[0].spec)
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

    stmts = FindNodes(ir.Assignment).visit(routine.body)
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
    assert not FindNodes(ir.CallStatement).visit(routine.body)
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3
    assert 'n' in routine.variables

    # An verify compiled behaviour
    filepath = here/(f'transform_inline_member_routines_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='member_routines')

    a = np.array([1., 2., 3.], order='F')
    b = np.array([3., 3., 3.], order='F')
    function(a, b)

    assert (a == [6., 7., 8.]).all()
    assert (b == [3., 3., 3.]).all()


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines_arg_dimensions(frontend):
    """
    Test inlining of member subroutines when sub-arrays of rank less
    than the formal argument are passed.
    """
    fcode = """
subroutine member_routines_arg_dimensions(matrix, tensor)
  real(kind=8), intent(inout) :: matrix(3, 3), tensor(3, 3, 4)
  integer :: i
  do i=1, 3
    call add_one(3, matrix(1:3,i), tensor(:,i,:))
  end do
  contains
    subroutine add_one(n, a, b)
      integer, intent(in) :: n
      real(kind=8), intent(inout) :: a(3), b(3,1:n)
      integer :: j
      do j=1, n
        a(j) = a(j) + 1
        b(j,:) = 66.6
      end do
    end subroutine
end subroutine member_routines_arg_dimensions
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Ensure initial member arguments
    assert len(routine.routines) == 1
    assert routine.routines[0].name == 'add_one'
    assert len(routine.routines[0].arguments) == 3
    assert routine.routines[0].arguments[0].name == 'n'
    assert routine.routines[0].arguments[1].name == 'a'
    assert routine.routines[0].arguments[2].name == 'b'

    # Now inline the member routines and check again
    inline_member_procedures(routine=routine)

    # Ensure member has been inlined and arguments adapated
    assert len(routine.routines) == 0
    assert len([v for v in FindVariables().visit(routine.body) if v.name == 'a']) == 0
    assert len([v for v in FindVariables().visit(routine.body) if v.name == 'b']) == 0
    assert len([v for v in FindVariables().visit(routine.spec) if v.name == 'a']) == 0
    assert len([v for v in FindVariables().visit(routine.spec) if v.name == 'b']) == 0
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].lhs == 'matrix(j, i)' and assigns[0].rhs =='matrix(j, i) + 1'
    assert assigns[1].lhs == 'tensor(j, i, :)'

    # Ensure the `n` in the inner loop bound has been substituted too
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 2
    assert loops[0].bounds == '1:3'
    assert loops[1].bounds == '1:3'


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'No header information in test')]))
def test_inline_member_routines_derived_type_member(frontend):
    """
    Test inlining of member subroutines when the member routine
    handles arrays that are derived type components and thus might
    have the DEFERRED type.
    """
    fcode = """
subroutine outer(x, a)
  real, intent(inout) :: x
  type(my_type), intent(in) :: a

  ! Pass derived type arrays as arguments
  call inner(a%b(:), a%c, a%k, a%n)

contains
  subroutine inner(y, z, k, n)
    integer, intent(in) :: k, n
    real, intent(inout) :: y(n), z(:,:)
    integer :: j

    do j=1, n
      x = x + y(j)
      ! Use derived-type variable as index
      ! to test for nested substitution
      y(j) = z(k,j)
    end do
  end subroutine inner
end subroutine outer
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert routine.variable_map['x'].type.dtype == BasicType.REAL
    assert isinstance(routine.variable_map['a'].type.dtype, DerivedType)
    call = FindNodes(ir.CallStatement).visit(routine.body)[0]
    assert isinstance(call.arguments[0], sym.Array)
    assert isinstance(call.arguments[1], sym.DeferredTypeSymbol)
    assert isinstance(call.arguments[2], sym.DeferredTypeSymbol)

    # Now inline the member routines and check again
    inline_member_procedures(routine=routine)

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].rhs =='x + a%b(j)'
    assert assigns[1].lhs == 'a%b(j)' and assigns[1].rhs == 'a%c(a%k, j)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines_variable_shadowing(frontend):
    """
    Test inlining of member subroutines when variable allocations
    in child routine shadow different allocations in the parent.
    """
    fcode = """
subroutine outer()
     real :: x = 3 ! 'x' is real in outer.
     real :: y

     y = 1.0
     call inner(y)
     x = x + y

contains
    subroutine inner(y)
        real, intent(inout) :: Y
        real :: x(3) ! 'x' is array in inner.
        x = [1, 2, 3]
        y = y + sum(x)
    end subroutine inner
end subroutine outer
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Check outer and inner 'x'
    assert routine.variable_map['x'] == 'x'
    assert isinstance(routine.variable_map['x'], sym.Scalar)
    assert routine.variable_map['x'].type.initial == 3

    assert routine['inner'].variable_map['x'] in ['x(3)', 'x(1:3)']
    assert isinstance(routine['inner'].variable_map['x'], sym.Array)
    assert routine['inner'].variable_map['x'].type.shape == (3,)

    inline_member_procedures(routine=routine)

    # Check outer has not changed
    assert routine.variable_map['x'] == 'x'
    assert isinstance(routine.variable_map['x'], sym.Scalar)
    assert routine.variable_map['x'].type.initial == 3

    # Check inner 'x' was moved correctly
    assert routine.variable_map['inner_x'] in ['inner_x(3)', 'inner_x(1:3)']
    assert isinstance(routine.variable_map['inner_x'], sym.Array)
    assert routine.variable_map['inner_x'].type.shape == (3,)

    # Check inner 'y' was substituted, not renamed!
    assign = FindNodes(ir.Assignment).visit(routine.body)
    assert routine.variable_map['y'] == 'y'
    assert assign[2].lhs == 'y' and assign[2].rhs == 'y + sum(inner_x)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_internal_routines_aliasing_declaration(frontend):
    """
    Test declaration splitting when inlining internal procedures.
    """
    fcode = """
subroutine outer()
  integer :: z
  integer :: jlon
  z = 0
  jlon = 0

  call inner(z)

  jlon = z + 4
contains
  subroutine inner(z)
    integer, intent(inout) :: z
    integer :: jlon, jg ! These two need to get separated
    jlon = 1
    jg = 2
    z = jlon + jg
  end subroutine inner
end subroutine outer
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Check outer and inner variables
    assert len(routine.variable_map) == 2
    assert 'z' in routine.variable_map
    assert 'jlon' in routine.variable_map

    assert len(routine['inner'].variable_map) == 3
    assert 'z' in routine['inner'].variable_map
    assert 'jlon' in routine['inner'].variable_map
    assert 'jg' in routine['inner'].variable_map

    inline_member_procedures(routine, allowed_aliases=('jlon',))

    assert len(routine.variable_map) == 3
    assert 'z' in routine.variable_map
    assert 'jlon' in routine.variable_map
    assert 'jg' in routine.variable_map

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 6
    assert assigns[2].lhs == 'jlon' and assigns[2].rhs == '1'
    assert assigns[3].lhs == 'jg' and assigns[3].rhs == '2'
    assert assigns[4].lhs == 'z' and assigns[4].rhs == 'jlon + jg'

@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines_indexing_of_shadowed_array(frontend):
    """
    Test special case of inlining of member subroutines when inlined routine contains
    shadowed array and array indices.
    In particular, this test checks that also the variables indexing
    the array in the inlined result get renamed correctly.
    """
    fcode = """
    subroutine outer(klon)
        integer :: jg, jlon
        integer :: arr(3, 3)

        jg = 70000
        call inner2()

        contains

        subroutine inner2()
            integer :: jlon, jg
            integer :: arr(3, 3)
            do jg=1,3
                do jlon=1,3
                   arr(jlon, jg) = 11
                end do
            end do
        end subroutine inner2

    end subroutine outer
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    inline_member_procedures(routine)
    innerloop = FindNodes(ir.Loop).visit(routine.body)[1]
    innerloopvars = FindVariables().visit(innerloop)
    assert 'inner2_arr(inner2_jlon,inner2_jg)' in innerloopvars

@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines_sequence_assoc(frontend):
    """
    Test inlining of member subroutines in the presence of sequence
    associations. As this is not supported, we check for the
    appropriate error.
    """
    fcode = """
subroutine member_routines_sequence_assoc(vector)
  real(kind=8), intent(inout) :: vector(6)
  integer :: i

  i = 2
  call inner(3, vector(i))

  contains
    subroutine inner(n, a)
      integer, intent(in) :: n
      real(kind=8), intent(inout) :: a(3)
      integer :: j
      do j=1, n
        a(j) = a(j) + 1
      end do
    end subroutine
end subroutine member_routines_sequence_assoc
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Expect to fail here due to use of sequence association
    with pytest.raises(RuntimeError):
        inline_member_procedures(routine=routine)


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines_with_associate(frontend):
    """
    Ensure that internal routines with :any:`Associate` constructs get
    inlined as expected.
    """
    fcode = """
subroutine acraneb_transt(klon, klev, kidia, kfdia, ktdia)
  implicit none

  integer(kind=4), intent(in) :: klon, klev, kidia, kfdia, ktdia
  integer(kind=4) :: jlon, jlev

  real(kind=8) :: zq1(klon)
  real(kind=8) :: zq2(klon, klev)

  call delta_t(zq1)

  do jlev = ktdia, klev
    call delta_t(zq2(1:klon,jlev))

  enddo

contains

subroutine delta_t(pq)
  implicit none

  real(kind=8), intent(in) :: pq(klon)
  real(kind=8) :: x, z

  associate(zz => z)

  do jlon = 1,klon
    x = x + pq(jlon)
  enddo
  end associate
end subroutine

end subroutine acraneb_transt
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)

    inline_member_procedures(routine=routine)

    assert not routine.members
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 3

    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0].rhs == 'x + zq1(jlon)'
    assert assigns[1].rhs == 'x + zq2(jlon, jlev)'

    assocs = FindNodes(ir.Associate).visit(routine.body)
    assert len(assocs) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines_with_optionals(frontend):
    """
    Ensure that internal routines with optional arguments get
    inlined as expected (esp. present instrinsics are correctly 
    evaluated for all variables types)
    """
    fcode = """
subroutine test_inline(ydxfu, klon)
  implicit none

  integer(kind=4), intent(in) :: klon, klev, kidia, kfdia, ktdia

  type(txfu)              ,intent(inout)            :: ydxfu
  type(mf_phys_out_type)  ,intent(in)               :: ydmf_phys_out
 
  call member_rout (ydxfu%visicld, pvmin=ydmf_phys_out%visicld, psmax=1.0_8)

  contains

  subroutine member_rout (x, pvmin, pvmax, psmin, psmax)
    
    real(kind=8)         ,intent(inout)            :: x(1:klon)
    real(kind=8)         ,intent(in)    ,optional  :: pvmin(1:klon)
    real(kind=8)         ,intent(in)    ,optional  :: pvmax(1:klon)
    real(kind=8)         ,intent(in)    ,optional  :: psmin
    real(kind=8)         ,intent(in)    ,optional  :: psmax
    
    if (present (psmin)) x = psmin
    if (present (psmax)) x = psmax
    if (present (pvmin)) x = minval(pvmin(:))
    if (present (pvmax)) x = maxval(pvmax(:))
    
  end subroutine member_rout 
    
end subroutine test_inline
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)

    inline_member_procedures(routine=routine)

    assert not routine.members

    conds = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conds) == 4
    assert conds[0].condition == 'False'
    assert conds[1].condition == 'True'
    assert conds[2].condition == 'True'
    assert conds[3].condition == 'False'



@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('adjust_imports', [True, False])
def test_inline_marked_subroutines(frontend, adjust_imports):
    """ Test subroutine inlining via marker pragmas. """

    fcode_driver = """
subroutine test_pragma_inline(a, b)
  use util_mod, only: add_one, add_a_to_b
  implicit none

  real(kind=8), intent(inout) :: a(3), b(3)
  integer, parameter :: n = 3
  integer :: i

  do i=1, n
    !$loki inline
    call add_one(a(i))
  end do

  !$loki inline
  call add_a_to_b(a(:), b(:), 3)

  do i=1, n
    call add_one(b(i))
  end do

end subroutine test_pragma_inline
    """

    fcode_module = """
module util_mod
implicit none

contains
  subroutine add_one(a)
    interface
      subroutine do_something()
      end subroutine do_something
    end interface
    real(kind=8), intent(inout) :: a
    a = a + 1
  end subroutine add_one

  subroutine add_a_to_b(a, b, n)
    interface
      subroutine do_something_else()
      end subroutine do_something_else
    end interface
    real(kind=8), intent(inout) :: a(:), b(:)
    integer, intent(in) :: n
    integer :: i

    do i = 1, n
      a(i) = a(i) + b(i)
    end do
  end subroutine add_a_to_b
end module util_mod
"""
    module = Module.from_source(fcode_module, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich(module)

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert calls[0].routine == module['add_one']
    assert calls[1].routine == module['add_a_to_b']
    assert calls[2].routine == module['add_one']

    inline_marked_subroutines(
        routine=driver, allowed_aliases=('I',), adjust_imports=adjust_imports
    )

    # Check inlined loops and assignments
    assert len(FindNodes(ir.Loop).visit(driver.body)) == 3
    assign = FindNodes(ir.Assignment).visit(driver.body)
    assert len(assign) == 2
    assert assign[0].lhs == 'a(i)' and assign[0].rhs == 'a(i) + 1'
    assert assign[1].lhs == 'a(i)' and assign[1].rhs == 'a(i) + b(i)'

    # Check that the last call is left untouched
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].routine.name == 'add_one'
    assert calls[0].arguments == ('b(i)',)

    imports = FindNodes(ir.Import).visit(driver.spec)
    assert len(imports) == 1
    if adjust_imports:
        assert imports[0].symbols == ('add_one',)
    else:
        assert imports[0].symbols == ('add_one', 'add_a_to_b')

    if adjust_imports:
        # check that explicit interfaces were imported
        intfs = driver.interfaces
        assert len(intfs) == 1
        assert all(isinstance(s, sym.ProcedureSymbol) for s in driver.interface_symbols)
        assert 'do_something' in driver.interface_symbols
        assert 'do_something_else' in driver.interface_symbols


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_marked_subroutines_with_interfaces(frontend):
    """ Test inlining of subroutines with explicit interfaces via marker pragmas. """

    fcode_driver = """
subroutine test_pragma_inline(a, b)
  implicit none

  interface
    subroutine add_a_to_b(a, b, n)
      real(kind=8), intent(inout) :: a(:), b(:)
      integer, intent(in) :: n
    end subroutine add_a_to_b
    subroutine add_one(a)
      real(kind=8), intent(inout) :: a
    end subroutine add_one
  end interface

  interface
    subroutine add_two(a)
      real(kind=8), intent(inout) :: a
    end subroutine add_two
  end interface

  real(kind=8), intent(inout) :: a(3), b(3)
  integer, parameter :: n = 3
  integer :: i

  do i=1, n
    !$loki inline
    call add_one(a(i))
  end do

  !$loki inline
  call add_a_to_b(a(:), b(:), 3)

  do i=1, n
    call add_one(b(i))
    !$loki inline
    call add_two(b(i))
  end do

end subroutine test_pragma_inline
    """

    fcode_module = """
module util_mod
implicit none

contains
  subroutine add_one(a)
    real(kind=8), intent(inout) :: a
    a = a + 1
  end subroutine add_one

  subroutine add_two(a)
    real(kind=8), intent(inout) :: a
    a = a + 2
  end subroutine add_two

  subroutine add_a_to_b(a, b, n)
    real(kind=8), intent(inout) :: a(:), b(:)
    integer, intent(in) :: n
    integer :: i

    do i = 1, n
      a(i) = a(i) + b(i)
    end do
  end subroutine add_a_to_b
end module util_mod
"""

    module = Module.from_source(fcode_module, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich(module.subroutines)

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert calls[0].routine == module['add_one']
    assert calls[1].routine == module['add_a_to_b']
    assert calls[2].routine == module['add_one']
    assert calls[3].routine == module['add_two']

    inline_marked_subroutines(routine=driver, allowed_aliases=('I',))

    # Check inlined loops and assignments
    assert len(FindNodes(ir.Loop).visit(driver.body)) == 3
    assign = FindNodes(ir.Assignment).visit(driver.body)
    assert len(assign) == 3
    assert assign[0].lhs == 'a(i)' and assign[0].rhs == 'a(i) + 1'
    assert assign[1].lhs == 'a(i)' and assign[1].rhs == 'a(i) + b(i)'
    assert assign[2].lhs == 'b(i)' and assign[2].rhs == 'b(i) + 2'

    # Check that the last call is left untouched
    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert len(calls) == 1
    assert calls[0].routine.name == 'add_one'
    assert calls[0].arguments == ('b(i)',)

    intfs = FindNodes(ir.Interface).visit(driver.spec)
    assert len(intfs) == 1
    assert intfs[0].symbols == ('add_one',)


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('adjust_imports', [True, False])
def test_inline_marked_routine_with_optionals(frontend, adjust_imports):
    """ Test subroutine inlining via marker pragmas with omitted optionals. """

    fcode_driver = """
subroutine test_pragma_inline_optionals(a, b)
  use util_mod, only: add_one
  implicit none

  real(kind=8), intent(inout) :: a(3), b(3)
  integer, parameter :: n = 3
  integer :: i

  do i=1, n
    !$loki inline
    call add_one(a(i), two=2.0)
  end do

  do i=1, n
    !$loki inline
    call add_one(b(i))
  end do

end subroutine test_pragma_inline_optionals
    """

    fcode_module = """
module util_mod
implicit none

contains
  subroutine add_one(a, two)
    real(kind=8), intent(inout) :: a
    real(kind=8), optional, intent(inout) :: two
    a = a + 1

    if (present(two)) then
      a = a + two
    end if
  end subroutine add_one
end module util_mod
"""
    module = Module.from_source(fcode_module, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)
    driver.enrich(module)

    calls = FindNodes(ir.CallStatement).visit(driver.body)
    assert calls[0].routine == module['add_one']
    assert calls[1].routine == module['add_one']

    inline_marked_subroutines(routine=driver, adjust_imports=adjust_imports)

    # Check inlined loops and assignments
    assert len(FindNodes(ir.Loop).visit(driver.body)) == 2
    assign = FindNodes(ir.Assignment).visit(driver.body)
    assert len(assign) == 4
    assert assign[0].lhs == 'a(i)' and assign[0].rhs == 'a(i) + 1'
    assert assign[1].lhs == 'a(i)' and assign[1].rhs == 'a(i) + 2.0'
    assert assign[2].lhs == 'b(i)' and assign[2].rhs == 'b(i) + 1'
    # TODO: This is a problem, since it's not declared anymore
    assert assign[3].lhs == 'b(i)' and assign[3].rhs == 'b(i) + two'

    # Check that the PRESENT checks have been resolved
    assert len(FindNodes(ir.CallStatement).visit(driver.body)) == 0
    assert len(FindInlineCalls().visit(driver.body)) == 0
    checks = FindNodes(ir.Conditional).visit(driver.body)
    assert len(checks) == 2
    assert checks[0].condition == 'True'
    assert checks[1].condition == 'False'

    imports = FindNodes(ir.Import).visit(driver.spec)
    assert len(imports) == 0 if adjust_imports else 1


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI has no sense of humour!')])
)
def test_inline_marked_subroutines_with_associates(frontend):
    """ Test subroutine inlining via marker pragmas with nested associates. """

    fcode_outer = """
subroutine test_pragma_inline_associates(never)
  use peter_pan, only: neverland
  implicit none
  type(neverland), intent(inout) :: never

  associate(going=>never%going_to)

  associate(up=>give_you%up)

  !$loki inline
  call dave(going, up)

  end associate

  end associate
end subroutine test_pragma_inline_associates
    """

    fcode_inner = """
subroutine dave(going)
  use your_imagination, only: astley
  implicit none
  type(astley), intent(inout) :: going

  associate(give_you=>going%give_you)

  associate(up=>give_you%up)

  call rick_is(up)

  end associate

  end associate
end subroutine dave
    """

    outer = Subroutine.from_source(fcode_outer, frontend=frontend)
    inner = Subroutine.from_source(fcode_inner, frontend=frontend)
    outer.enrich(inner)

    assert FindNodes(ir.CallStatement).visit(outer.body)[0].routine == inner

    inline_marked_subroutines(routine=outer, adjust_imports=True)

    # Ensure that all associates are perfectly nested afterwards
    assocs = FindNodes(ir.Associate).visit(outer.body)
    assert len(assocs) == 4
    assert assocs[1].parent == assocs[0]
    assert assocs[2].parent == assocs[1]
    assert assocs[3].parent == assocs[2]

    # And, because we can...
    outer.body = ResolveAssociatesTransformer().visit(outer.body)
    call = FindNodes(ir.CallStatement).visit(outer.body)[0]
    assert call.name == 'rick_is'
    assert call.arguments == ('never%going_to%give_you%up',)
    # Q. E. D.


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_marked_subroutines_declarations(frontend):
    """Test symbol propagation to hoisted declaration when inlining."""
    fcode = """
module inline_declarations
  implicit none

  type bounds
    integer :: start, end
  end type bounds

  contains

  subroutine outer(a, bnds)
    real(kind=8), intent(inout) :: a(bnds%end)
    type(bounds), intent(in) :: bnds
    real(kind=8) :: b(bnds%end)

    b(bnds%start:bnds%end) = a(bnds%start:bnds%end) + 42.0

    !$loki inline
    call inner(a, dims=bnds)
  end subroutine outer

  subroutine inner(c, dims)
    real(kind=8), intent(inout) :: c(dims%end)
    type(bounds), intent(in) :: dims
    real(kind=8) :: d(dims%end)

    d(dims%start:dims%end) = c(dims%start:dims%end) - 66.6
    c(dims%start) = sum(d)
  end subroutine inner
end module inline_declarations
"""
    module = Module.from_source(fcode, frontend=frontend)
    outer = module['outer']

    inline_marked_subroutines(routine=outer, adjust_imports=True)

    # Check that all declarations are using the ``bnds`` symbol
    assert outer.symbols[0] == 'a(1:bnds%end)' if frontend == OMNI else 'a(bnds%end)'
    assert outer.symbols[2] == 'b(1:bnds%end)' if frontend == OMNI else 'b(bnds%end)'
    assert outer.symbols[3] == 'd(1:bnds%end)' if frontend == OMNI else 'd(bnds%end)'
    assert all(
        a.shape == ('bnds%end',) for a in outer.symbols if isinstance(a, sym.Array)
    )


@pytest.mark.parametrize('frontend', available_frontends(
    (OFP, 'Prefix/elemental support not implemented'))
)
def test_inline_transformation(frontend):
    """Test combining recursive inlining via :any:`InliningTransformation`."""

    fcode_module = """
module one_mod
  real(kind=8), parameter :: one = 1.0
end module one_mod
"""

    fcode_inner = """
subroutine add_one_and_two(a)
  use one_mod, only: one
  implicit none

  real(kind=8), intent(inout) :: a

  a = a + one

  a = add_two(a)

contains
  elemental function add_two(x)
    real(kind=8), intent(in) :: x
    real(kind=8) :: add_two

    add_two = x + 2.0
  end function add_two
end subroutine add_one_and_two
"""

    fcode = """
subroutine test_inline_pragma(a, b)
  implicit none
  real(kind=8), intent(inout) :: a(3), b(3)
  integer, parameter :: n = 3
  integer :: i

#include "add_one_and_two.intfb.h"

  do i=1, n
    !$loki inline
    call add_one_and_two(a(i))
  end do

  do i=1, n
    !$loki inline
    call add_one_and_two(b(i))
  end do

end subroutine test_inline_pragma
"""
    module = Module.from_source(fcode_module, frontend=frontend)
    inner = Subroutine.from_source(fcode_inner, definitions=module, frontend=frontend)
    routine = Subroutine.from_source(fcode, frontend=frontend)
    routine.enrich(inner)

    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True
    )

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 2
    assert all(c.routine == inner for c in calls)

    # Apply to the inner subroutine first to resolve parameter and calls
    trafo.apply(inner)

    assigns = FindNodes(ir.Assignment).visit(inner.body)
    assert len(assigns) == 2
    assert assigns[0].lhs == 'a' and assigns[0].rhs == 'a + 1.0'
    assert assigns[1].lhs == 'a' and assigns[1].rhs == 'a + 2.0'

    # Apply to the outer routine, but with resolved body of the inner
    trafo.apply(routine)

    calls = FindNodes(ir.CallStatement).visit(routine.body)
    assert len(calls) == 0
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 4
    assert assigns[0].lhs == 'a(i)' and assigns[0].rhs == 'a(i) + 1.0'
    assert assigns[1].lhs == 'a(i)' and assigns[1].rhs == 'a(i) + 2.0'
    assert assigns[2].lhs == 'b(i)' and assigns[2].rhs == 'b(i) + 1.0'
    assert assigns[3].lhs == 'b(i)' and assigns[3].rhs == 'b(i) + 2.0'


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_local_seq_assoc(frontend):
    fcode = """
module somemod
    implicit none
    contains

    subroutine minusone_second(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x(3)
        output = x(2) - 1
    end subroutine minusone_second

    subroutine plusone(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x
        output = x + 1
    end subroutine plusone

    subroutine outer()
      implicit none
      real :: x(3, 3)
      real :: y
      x = 10.0

      call inner(y, x(1, 1)) ! Sequence association here for member routine.

      !$loki inline
      call plusone(y, x(3, 3)) ! Marked for inlining.

      call minusone_second(y, x(1, 3)) ! Standard call with sequence association (never processed).

      contains

      subroutine inner(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x(3)

        output = x(2) + 2.0
      end subroutine inner
    end subroutine outer

end module somemod
"""
    # Test case that nothing happens if `resolve_sequence_association=True`
    # but inlining "marked" and "internals" is disabled.
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=False, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' in callnames
    assert 'inner' in callnames
    assert 'minusone_second' in callnames

    # Test case that only marked processed if
    # `resolve_sequence_association=True`
    # `inline_marked=True`,
    # `inline_internals=False`
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' not in callnames
    assert 'inner' in callnames
    assert 'minusone_second' in callnames

    # Test case that a crash occurs if sequence association is not enabled even if it is needed.
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=True, resolve_sequence_association=False
    )
    outer = module["outer"]
    with pytest.raises(RuntimeError):
        trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]

    # Test case that sequence association is run and corresponding call inlined, avoiding crash.
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=False, inline_internals=True, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' in callnames
    assert 'inner' not in callnames
    assert 'minusone_second' in callnames

    # Test case that everything is enabled.
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=True, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    callnames = [call.name for call in FindNodes(ir.CallStatement).visit(outer.body)]
    assert 'plusone' not in callnames
    assert 'inner' not in callnames
    assert 'minusone_second' in callnames


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_local_seq_assoc_crash_marked_no_seq_assoc(frontend):
    # Test case that a crash occurs if marked routine with sequence association is
    # attempted to inline without sequence association enabled.
    fcode = """
module somemod
    implicit none
    contains

    subroutine inner(output, x)
        real, intent(inout) :: output
        real, intent(in) :: x(3)

        output = x(2) + 2.0
    end subroutine inner

    subroutine outer()
      real :: x(3, 3)
      real :: y
      x = 10.0

      !$loki inline
      call inner(y, x(1, 1)) ! Sequence association here for marked routine.
    end subroutine outer

end module somemod
"""
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=False
    )
    outer = module["outer"]
    with pytest.raises(RuntimeError):
        trafo.apply(outer)

    # Test case that crash is avoided by activating sequence association.
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    trafo.apply(outer)
    assert len(FindNodes(ir.CallStatement).visit(outer.body)) == 0

@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_local_seq_assoc_crash_value_err_no_source(frontend):
    # Testing that ValueError is thrown if sequence association is requested with inlining,
    # but source code behind call is missing (not enough type information).
    fcode = """
module somemod
    implicit none
    contains

    subroutine outer()
      real :: x(3, 3)
      real :: y
      x = 10.0

      !$loki inline
      call inner(y, x(1, 1)) ! Sequence association here for marked routine.
    end subroutine outer

end module somemod
"""
    module = Module.from_source(fcode, frontend=frontend)
    trafo = InlineTransformation(
        inline_constants=True, external_only=True, inline_elementals=True,
        inline_marked=True, inline_internals=False, resolve_sequence_association=True
    )
    outer = module["outer"]
    with pytest.raises(ValueError):
        trafo.apply(outer)


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_adjust_imports(frontend):
    fcode_module = """
module bnds_module
  integer :: m
  integer :: n
  integer :: l
end module bnds_module
    """

    fcode_another = """
module another_module
  integer :: x
end module another_module
    """

    fcode_outer = """
subroutine test_inline_outer(a, b)
  use bnds_module, only: n
  use test_inline_mod, only: test_inline_inner
  use test_inline_another_mod, only: test_inline_another_inner
  implicit none

  real(kind=8), intent(inout) :: a(n), b(n)

  !$loki inline
  call test_inline_another_inner()
  !$loki inline
  call test_inline_inner(a, b)
end subroutine test_inline_outer
    """

    fcode_inner = """
module test_inline_mod
  implicit none
  contains

subroutine test_inline_inner(a, b)
  use BNDS_module, only: n, m
  use another_module, only: x

  real(kind=8), intent(inout) :: a(n), b(n)
  real(kind=8) :: tmp(m)
  integer :: i

  tmp(1:m) = x
  do i=1, n
    a(i) = b(i) + sum(tmp)
  end do
end subroutine test_inline_inner
end module test_inline_mod
    """

    fcode_another_inner = """
module test_inline_another_mod
  implicit none
  contains

subroutine test_inline_another_inner()
  use BNDS_module, only: n, m, l

end subroutine test_inline_another_inner
end module test_inline_another_mod
    """

    _ = Module.from_source(fcode_another, frontend=frontend)
    _ = Module.from_source(fcode_module, frontend=frontend)
    inner = Module.from_source(fcode_inner, frontend=frontend)
    another_inner = Module.from_source(fcode_another_inner, frontend=frontend)
    outer = Subroutine.from_source(
        fcode_outer, definitions=(inner, another_inner), frontend=frontend
    )

    trafo = InlineTransformation(
        inline_elementals=False, inline_marked=True, adjust_imports=True
    )
    trafo.apply(outer)

    # Check that the inlining has happened
    assign = FindNodes(ir.Assignment).visit(outer.body)
    assert len(assign) == 2
    assert assign[0].lhs == 'tmp(1:m)'
    assert assign[0].rhs == 'x'
    assert assign[1].lhs == 'a(i)'
    assert assign[1].rhs == 'b(i) + sum(tmp)'

    # Now check that the right modules have been moved,
    # and the import of the call has been removed
    imports = FindNodes(ir.Import).visit(outer.spec)
    assert len(imports) == 2
    assert imports[0].module == 'another_module'
    assert imports[0].symbols == ('x',)
    assert imports[1].module == 'bnds_module'
    assert all(_ in imports[1].symbols for _ in ['l', 'm', 'n'])


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_transformation_intermediate(tmp_path, frontend):
    fcode_outermost = """
module outermost_mod
implicit none
contains
subroutine outermost()
use intermediate_mod, only: intermediate

!$loki inline
call intermediate()

end subroutine outermost
end module outermost_mod
"""

    fcode_intermediate = """
module intermediate_mod
implicit none
contains
subroutine intermediate()
use innermost_mod, only: innermost

call innermost()

end subroutine intermediate
end module intermediate_mod
"""

    fcode_innermost = """
module innermost_mod
implicit none
contains
subroutine innermost()

end subroutine innermost
end module innermost_mod
"""

    (tmp_path/'outermost_mod.F90').write_text(fcode_outermost)
    (tmp_path/'intermediate_mod.F90').write_text(fcode_intermediate)
    (tmp_path/'innermost_mod.F90').write_text(fcode_innermost)

    config = {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True
        },
        'routines': {
            'outermost': {'role': 'kernel'}
        }
    }

    scheduler = Scheduler(paths=[tmp_path], config=SchedulerConfig.from_dict(config), frontend=frontend)

    def _get_successors(item):
        return scheduler.sgraph.successors(scheduler[item])

    # check graph edges before transformation
    assert len(scheduler.items) == 3
    assert len(_get_successors('outermost_mod#outermost')) == 1
    assert scheduler['intermediate_mod#intermediate'] in _get_successors('outermost_mod#outermost')
    assert len(_get_successors('intermediate_mod#intermediate')) == 1
    assert scheduler['innermost_mod#innermost'] in _get_successors('intermediate_mod#intermediate')

    scheduler.process( transformation=InlineTransformation() )

    # check graph edges were updated correctly
    assert len(scheduler.items) == 2
    assert len(_get_successors('outermost_mod#outermost')) == 1
    assert scheduler['innermost_mod#innermost'] in _get_successors('outermost_mod#outermost')
