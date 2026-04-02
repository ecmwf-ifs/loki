# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for subroutine parse shape.
"""

import pytest

from loki import Subroutine
from loki.expression import symbols as sym
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes
from loki.types import BasicType


@pytest.mark.parametrize('frontend', available_frontends())
def test_routine_simple(frontend):
    """
    A simple standard looking routine to test argument declarations.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
  ! This is the docstring ...

  ! It spans multiple intersected lines ...
  ! ... and is followed by a ...

  !$loki routine fun

  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i

  do i=1, x
     vector(i) = vector(i) + scalar
     matrix(i, :) = i * vector(i)
  end do
end subroutine routine_simple
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    assert routine.arguments == ('x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)')
    assert routine.variables == ('jprb', 'x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)', 'i')

    assert len(routine.docstring) == 1
    assert isinstance(routine.docstring[0], ir.CommentBlock)
    if frontend == OMNI:
        assert len(routine.docstring[0].comments) == 3
        assert routine.docstring[0].comments[0].text == '! This is the docstring ...'
        assert routine.docstring[0].comments[1].text == '! It spans multiple intersected lines ...'
        assert routine.docstring[0].comments[2].text == '! ... and is followed by a ...'
    else:
        assert len(routine.docstring[0].comments) == 5
        assert routine.docstring[0].comments[0].text == '! This is the docstring ...'
        assert routine.docstring[0].comments[2].text == '! It spans multiple intersected lines ...'
        assert routine.docstring[0].comments[3].text == '! ... and is followed by a ...'
    assert routine.definitions == ()

    assert isinstance(routine.body, ir.Section)
    if frontend == OMNI:
        assert len(routine.spec) == 9
        assert isinstance(routine.spec[0], ir.Intrinsic)
        assert isinstance(routine.spec[1], ir.Pragma)
        assert all(isinstance(node, ir.VariableDeclaration) for node in routine.spec[2:])
        assert routine.spec[2].symbols == ('jprb',)
        assert routine.spec[3].symbols == ('x',)
        assert routine.spec[4].symbols == ('y',)
        assert routine.spec[5].symbols == ('scalar',)
        assert routine.spec[6].symbols == ('vector(x)',)
        assert routine.spec[7].symbols == ('matrix(x, y)',)
        assert routine.spec[8].symbols == ('i',)
    else:
        assert len(routine.spec) == 7
        assert isinstance(routine.spec[0], ir.Pragma)
        assert isinstance(routine.spec[1], ir.Comment)
        assert all(isinstance(node, ir.VariableDeclaration) for node in routine.spec[2:])
        assert routine.spec[2].symbols == ('jprb',)
        assert routine.spec[3].symbols == ('x', 'y')
        assert routine.spec[4].symbols == ('scalar',)
        assert routine.spec[5].symbols == ('vector(x)', 'matrix(x, y)')
        assert routine.spec[6].symbols == ('i',)

    assert isinstance(routine.spec, ir.Section)
    loops = FindNodes(ir.Loop).visit(routine.body)
    assert len(loops) == 1 and loops[0].variable == 'i'
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assigns) == 2
    assert assigns[0] in loops[0].body and assigns[1] in loops[0].body


@pytest.mark.parametrize('frontend', available_frontends())
def test_routine_arguments(frontend):
    """
    A set of test to test internalisation and handling of arguments.
    """

    fcode = """
subroutine routine_arguments &
 ! Test multiline dummy arguments with comments
 & (x, y, scalar, &
 ! Of course, not one...
 ! but two comment lines
 & vector, matrix)
  implicit none
  integer, parameter :: jprb = selected_real_kind(13,300)
  ! The order below is intentioanlly inverted
  real(kind=jprb), intent(inout) :: matrix(x, y)
  real(kind=jprb), intent(in)    :: scalar
  real(kind=jprb), dimension(x)  :: local_vector
  real(kind=jprb), dimension(x), intent(out) :: vector
  integer, intent(in) :: x, y

  integer :: i, j
  real(kind=jprb) :: local_matrix(x, y)

  do i=1, x
     local_vector(i) = i * 10.
     do j=1, y
        local_matrix(i, j) = local_vector(i) + j * scalar
     end do
  end do

  vector(:) = local_vector(:)
  matrix(:, :) = local_matrix(:, :)

end subroutine routine_arguments
"""

    routine = Subroutine.from_source(fcode, frontend=frontend)

    if frontend == OMNI:
        assert not routine.docstring
    else:
        assert len(routine.docstring) == 1
        assert len(routine.docstring[0].comments) == 3
        assert routine.docstring[0].comments[0].text == '! Test multiline dummy arguments with comments'
        assert routine.docstring[0].comments[1].text == '! Of course, not one...'
        assert routine.docstring[0].comments[2].text == '! but two comment lines'

    assert routine.arguments == ('x', 'y', 'scalar', 'vector(x)', 'matrix(x, y)')
    assert all(isinstance(arg, sym.Scalar) for arg in routine.arguments[0:3])
    assert all(arg.type.intent == 'in' for arg in routine.arguments[0:3])
    assert all(isinstance(arg, sym.Array) for arg in routine.arguments[3:])
    assert all(arg.type.dtype == BasicType.INTEGER for arg in routine.arguments[0:2])
    assert all(arg.type.dtype == BasicType.REAL for arg in routine.arguments[2:5])
    if frontend == OMNI:
        assert all(isinstance(arg.type.kind, sym.InlineCall) for arg in routine.arguments[2:5])
    else:
        assert all(arg.type.kind == 'jprb' for arg in routine.arguments[2:5])
    assert routine.arguments[3].shape == ('x',)
    assert routine.arguments[4].shape == ('x', 'y')
    assert routine.arguments[3].type.intent == 'out'
    assert routine.arguments[4].type.intent == 'inout'

    assert routine.variables == (
        'jprb', 'matrix(x, y)', 'scalar', 'local_vector(x)',
        'vector(x)', 'x', 'y', 'i', 'j', 'local_matrix(x, y)'
    )
    assert routine.variables[0].type.parameter
    assert isinstance(routine.variables[0].type.initial, sym.InlineCall)
    assert routine.variables[0].type.initial.function == 'selected_real_kind'
    assert routine.variables[1].type.dtype == BasicType.REAL
    assert routine.variables[1].shape == ('x', 'y')
    assert routine.variables[2].type.dtype == BasicType.REAL
    assert routine.variables[3].type.dtype == BasicType.REAL
    assert routine.variables[3].shape == ('x',)
    assert routine.variables[4].type.dtype == BasicType.REAL
    assert routine.variables[4].shape == ('x',)
    assert routine.variables[5].type.dtype == BasicType.INTEGER
    assert routine.variables[6].type.dtype == BasicType.INTEGER
    assert routine.variables[7].type.dtype == BasicType.INTEGER
    assert routine.variables[8].type.dtype == BasicType.INTEGER
    assert routine.variables[9].type.dtype == BasicType.REAL
    assert routine.variables[9].shape == ('x', 'y')


@pytest.mark.parametrize('frontend', available_frontends())
def test_empty_spec(frontend):
    routine = Subroutine.from_source(frontend=frontend, source="""
subroutine routine_empty_spec
write(*,*) 'Hello world!'
end subroutine routine_empty_spec
""")
    if frontend == OMNI:
        assert len(routine.spec) == 1
    else:
        assert not routine.spec
    assert len(routine.body.body) == 1
