# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import numpy as np

from loki import Module, Subroutine
from loki.build import jit_compile
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, FindNodes, FindVariables, FindInlineCalls
)
from loki.types import BasicType, DerivedType

from loki.transformations.inline import (
    inline_member_procedures, inline_marked_subroutines
)
from loki.transformations.sanitise import ResolveAssociatesTransformer


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_routines(tmp_path, frontend):
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

    filepath = tmp_path/(f'ref_transform_inline_member_routines_{frontend}.f90')
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
    filepath = tmp_path/(f'transform_inline_member_routines_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='member_routines')

    a = np.array([1., 2., 3.], order='F')
    b = np.array([3., 3., 3.], order='F')
    function(a, b)

    assert (a == [6., 7., 8.]).all()
    assert (b == [3., 3., 3.]).all()


@pytest.mark.parametrize('frontend', available_frontends())
def test_inline_member_functions(tmp_path, frontend):
    """
    Test inlining of member subroutines.
    """
    fcode = """
subroutine member_functions(a, b, c)
  implicit none
  real(kind=8), intent(inout) :: a(3), b(3), c(3)
  integer :: i

  do i=1, size(a)
    a(i) = add_one(a(i))
  end do

  c = add_to_a(b, 3)

  do i=1, size(a)
    a(i) = add_one(a(i))
  end do

  contains

    function add_one(a)
      real(kind=8) :: a
      real(kind=8) :: add_one
      add_one = a + 1
    end function

    function add_to_a(b, n)
      integer, intent(in) :: n
      real(kind=8), intent(in) :: b(n)
      real(kind=8) :: add_to_a(n)

      do i = 1, n
        add_to_a(i) = a(i) + b(i)
      end do
    end function
end subroutine member_functions
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)

    filepath = tmp_path/(f'ref_transform_inline_member_functions_{frontend}.f90')
    reference = jit_compile(routine, filepath=filepath, objname='member_functions')

    a = np.array([1., 2., 3.], order='F')
    b = np.array([3., 3., 3.], order='F')
    c = np.array([0., 0., 0.], order='F')
    reference(a, b, c)

    assert (a == [3., 4., 5.]).all()
    assert (b == [3., 3., 3.]).all()
    assert (c == [5., 6., 7.]).all()

    # Now inline the member routines and check again
    inline_member_procedures(routine=routine)

    assert not routine.members
    assert not FindNodes(ir.CallStatement).visit(routine.body)
    assert len(FindNodes(ir.Loop).visit(routine.body)) == 3

    # An verify compiled behaviour
    filepath = tmp_path/(f'transform_inline_member_functions_{frontend}.f90')
    function = jit_compile(routine, filepath=filepath, objname='member_functions')

    a = np.array([1., 2., 3.], order='F')
    b = np.array([3., 3., 3.], order='F')
    c = np.array([0., 0., 0.], order='F')
    function(a, b, c)

    assert (a == [3., 4., 5.]).all()
    assert (b == [3., 3., 3.]).all()
    assert (c == [5., 6., 7.]).all()


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

    # Expect to fail tmp_path due to use of sequence association
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


@pytest.mark.parametrize('frontend', available_frontends(
    xfail=[(OMNI, 'OMNI does not handle missing type definitions')]
))
def test_inline_member_routines_with_optionals(frontend):
    """
    Ensure that internal routines with optional arguments get
    inlined as expected (esp. present instrinsics are correctly 
    evaluated for all variables types)
    """
    fcode = """
subroutine test_inline(klon, ydxfu, ydmf_phys_out)

  use yomxfu                  , only : txfu
  use mf_phys_type_mod        , only : mf_phys_out_type

  implicit none

  integer(kind=4), intent(in) :: klon
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
def test_inline_marked_subroutines(frontend, adjust_imports, tmp_path):
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
    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path])
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
def test_inline_marked_subroutines_with_interfaces(frontend, tmp_path):
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

    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path])
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
def test_inline_marked_routine_with_optionals(frontend, adjust_imports, tmp_path):
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
    module = Module.from_source(fcode_module, frontend=frontend, xmods=[tmp_path])
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, xmods=[tmp_path])
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
def test_inline_marked_subroutines_declarations(frontend, tmp_path):
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
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    outer = module['outer']

    inline_marked_subroutines(routine=outer, adjust_imports=True)

    # Check that all declarations are using the ``bnds`` symbol
    assert outer.symbols[0] == 'a(bnds%end)'
    assert outer.symbols[2] == 'b(bnds%end)'
    assert outer.symbols[3] == 'd(bnds%end)'
    assert all(
        a.shape == ('bnds%end',) for a in outer.symbols if isinstance(a, sym.Array)
    )
