# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pymbolic.primitives import Expression

from loki import Module, Subroutine
from loki.frontend import available_frontends, OMNI
from loki.ir import (
    nodes as ir, is_parent_of, is_child_of, FindNodes, FindScopes,
    FindVariables, ExpressionFinder
)
from loki.expression import (
    symbols as sym, ExpressionCallbackMapper, ExpressionRetriever
)


@pytest.mark.parametrize('frontend', available_frontends())
def test_find_nodes_greedy(frontend):
    """
    Test the FindNodes visitor's greedy property.
    """
    fcode = """
subroutine routine_find_nodes_greedy(n, m)
  integer, intent(in) :: n, m

  if (n > m) then
    if (n == 3) then
      print *,"Inner if"
    endif
    print *,"Outer if"
  endif
end subroutine routine_find_nodes_greedy
"""

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conditionals) == 2

    outer_cond = FindNodes(ir.Conditional, greedy=True).visit(routine.body)
    assert len(outer_cond) == 1
    assert outer_cond[0] in conditionals
    assert str(outer_cond[0].condition) == 'n > m'


@pytest.mark.parametrize('frontend', available_frontends())
def test_find_scopes(frontend):
    """
    Test the FindScopes visitor.
    """
    fcode = """
subroutine routine_find_nodes_greedy(n, m)
  integer, intent(in) :: n, m

  if (n > m) then
    if (n == 3) then
      print *,"Inner if"
    endif
    print *,"Outer if"
  endif
end subroutine routine_find_nodes_greedy
""".strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    intrinsics = FindNodes(ir.Intrinsic).visit(routine.body)
    assert len(intrinsics) == 2
    inner = [i for i in intrinsics if 'Inner' in i.text][0]
    outer = [i for i in intrinsics if 'Outer' in i.text][0]

    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(conditionals) == 2

    scopes = FindScopes(inner).visit(routine.body)
    assert len(scopes) == 1  # returns a list containing a list of nested nodes
    assert len(scopes[0]) == 4  # should have found 3 scopes and the node itself
    assert all(c in scopes[0] for c in conditionals)  # should have found all if
    assert routine.body is scopes[0][0]  # body section should be outermost scope
    assert str(scopes[0][1].condition) == 'n > m'  # outer if should come first
    assert inner is scopes[0][-1]  # node itself should be last in list

    scopes = FindScopes(outer).visit(routine.body)
    assert len(scopes) == 1  # returns a list containing a list of nested nodes
    assert len(scopes[0]) == 3  # should have found 2 scopes and the node itself
    assert all(c in scopes[0] or str(c.condition == 'n == 3')
               for c in conditionals)  # should have found only the outer if
    assert routine.body is scopes[0][0]  # body section should be outermost scope
    assert outer is scopes[0][-1]  # node itself should be last in list


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_finder(frontend):
    """
    Test the expression finder's ability to yield only all variables.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
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

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = FindVariables(unique=False).visit(routine.body)
    assert len(variables) == 12
    assert all(isinstance(v, Expression) for v in variables)

    assert sorted([str(v) for v in variables]) == (
        ['i'] * 6 + ['matrix(i, :)', 'scalar'] + ['vector(i)'] * 3 + ['x'])


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_finder_unique(frontend):
    """
    Test the expression finder's ability to yield unique variables.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
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

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = FindVariables().visit(routine.body)
    assert isinstance(variables, set)
    assert len(variables) == 5
    assert all(isinstance(v, Expression) for v in variables)

    assert sorted([str(v) for v in variables]) == ['i', 'matrix(i, :)', 'scalar', 'vector(i)', 'x']


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_finder_with_ir_node(frontend):
    """
    Test the expression finder's ability to yield the root node.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
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

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = FindVariables(unique=False, with_ir_node=True).visit(routine.body)
    assert len(variables) == 3
    assert all(isinstance(v, tuple) and len(v) == 2 for v in variables)

    # Verify that the variables in the loop definition are found
    loops = [v for v in variables if isinstance(v[0], ir.Loop)]
    assert len(loops) == 1
    assert sorted([str(v) for v in loops[0][1]]) == ['i', 'x']

    # Verify that the variables in the statements are found
    stmts = [v for v in variables if isinstance(v[0], ir.Assignment)]
    assert len(stmts) == 2

    assert sorted([str(v) for v in stmts[0][1]]) == ['i', 'i', 'scalar', 'vector(i)', 'vector(i)']
    assert sorted([str(v) for v in stmts[1][1]]) == ['i', 'i', 'i', 'matrix(i, :)', 'vector(i)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_finder_unique_with_ir_node(frontend):
    """
    Test the expression finder's ability to yield the ir node combined with only unique
    variables.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
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

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = FindVariables(with_ir_node=True).visit(routine.body)
    assert len(variables) == 3
    assert all(isinstance(v, tuple) and len(v) == 2 for v in variables)

    # Verify that the variables in the loop definition are found
    loops = [v for v in variables if isinstance(v[0], ir.Loop)]
    assert len(loops) == 1
    assert sorted([str(v) for v in loops[0][1]]) == ['i', 'x']

    # Verify that the variables in the statements are found
    stmts = [v for v in variables if isinstance(v[0], ir.Assignment)]
    assert len(stmts) == 2

    assert sorted([str(v) for v in stmts[0][1]]) == ['i', 'scalar', 'vector(i)']
    assert sorted([str(v) for v in stmts[1][1]]) == ['i', 'matrix(i, :)', 'vector(i)']


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_callback_mapper(frontend):
    """
    Test the ExpressionFinder together with ExpressionCallbackMapper. This is just a very basic
    sanity check and does not cover all angles.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i, j

  do i=1, x
    vector(i) = vector(i) + scalar
    do j=1, y
      if (j > i) then
        matrix(i, j) = real(i * j, kind=jprb) + 1.
      else
        matrix(i, j) = i * vector(j)
      end if
    end do
  end do
end subroutine routine_simple
"""

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Nonsense example that singles out anything that is a matrix
    def is_matrix(expr, *args, **kwargs):  # pylint: disable=unused-argument
        if isinstance(expr, sym.Array) and expr.type.shape and len(expr.type.shape) == 2:
            return expr
        return None

    class FindMatrix(ExpressionFinder):
        retriever = ExpressionCallbackMapper(
            callback=is_matrix,
            combine=lambda v: tuple(e for e in v if e is not None)
        )

    matrix_count = FindMatrix(unique=False).visit(routine.body)
    assert len(matrix_count) == 2

    matrix_count = FindMatrix().visit(routine.body)
    assert len(matrix_count) == 1
    assert str(matrix_count.pop()) == 'matrix(i, j)'


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_retriever_recurse_query(frontend):
    """
    Test the ExpressionRetriever with a custom recurse query that allows to terminate recursion
    early.
    """
    fcode = """
subroutine routine_simple (x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i, j

  do i=1, x
    vector(i) = vector(i) + scalar
    do j=1, y
      if (j > i) then
        matrix(i, j) = real(i * j + 2, kind=jprb) + 1.
      else
        matrix(i, j) = i * vector(j)
      end if
    end do
  end do
end subroutine routine_simple
"""

    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Find all literals except when they appear in array subscripts or loop ranges
    class FindLiteralsNotInSubscriptsOrRanges(ExpressionFinder):
        retriever = ExpressionRetriever(
            query=lambda expr: isinstance(expr, (sym.IntLiteral, sym.FloatLiteral, sym.LogicLiteral)),
            recurse_query=lambda expr, *args, **kwargs: not isinstance(expr, (sym.ArraySubscript, sym.LoopRange))
        )
    literals = FindLiteralsNotInSubscriptsOrRanges(unique=False).visit(routine.body)

    if frontend == OMNI:
        # OMNI substitutes jprb
        assert len(literals) == 4
        assert sorted([str(l) for l in literals]) == ['1.', '13', '2', '300']
    else:
        assert len(literals) == 2
        assert sorted([str(l) for l in literals]) == ['1.', '2']


@pytest.mark.parametrize('frontend', available_frontends())
def test_find_variables_associates(frontend):
    """
    Test correct discovery of variables in associates.
    """
    fcode = """
subroutine find_variables_associates (x, y, scalar, vector, matrix)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: scalar
  real(kind=jprb), intent(inout) :: vector(x), matrix(x, y)
  integer :: i, j

  do i=1, x
    associate (v => vector(i), m => matrix(i, :))
      vector(i) = vector(i) + scalar
      do j=1, y
        if (j > i) then
          m(j) = real(i * j, kind=jprb) + 1.
        else
          matrix(i, j) = i * vector(j)
        end if
      end do
    end associate
  end do
end subroutine find_variables_associates
"""
    # Test the internals of the subroutine
    routine = Subroutine.from_source(fcode, frontend=frontend)

    variables = FindVariables(unique=False).visit(routine.body)
    assert len(variables) == 27 if frontend == OMNI else 28
    assert len([v for v in variables if v.name == 'v']) == 1
    assert len([v for v in variables if v.name == 'm']) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_is_parent_of(frontend):
    """
    Test the ``is_parent_of`` utility.
    """
    fcode = """
subroutine test_is_parent_of
  implicit none
  integer :: a, j, n=10

  a = 0
  do j=1,n
    if (j > 3) then
      a = a + 1
    end if
  end do
end subroutine test_is_parent_of
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    loop = FindNodes(ir.Loop).visit(routine.body)[0]
    conditional = FindNodes(ir.Conditional).visit(routine.body)[0]
    assignments = FindNodes(ir.Assignment).visit(routine.body)

    assert is_parent_of(loop, conditional)
    assert not is_parent_of(conditional, loop)

    for node in [loop, conditional]:
        assert {is_parent_of(node, a) for a in assignments} == {True, False}
        assert all(not is_parent_of(a, node) for a in assignments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_is_child_of(frontend):
    """
    Test the ``is_child_of`` utility.
    """
    fcode = """
subroutine test_is_child_of
  implicit none
  integer :: a, j, n=10

  a = 0
  do j=1,n
    if (j > 3) then
      a = a + 1
    end if
  end do
end subroutine test_is_child_of
    """.strip()
    routine = Subroutine.from_source(fcode, frontend=frontend)

    loop = FindNodes(ir.Loop).visit(routine.body)[0]
    conditional = FindNodes(ir.Conditional).visit(routine.body)[0]
    assignments = FindNodes(ir.Assignment).visit(routine.body)

    assert not is_child_of(loop, conditional)
    assert is_child_of(conditional, loop)

    for node in [loop, conditional]:
        assert {is_child_of(a, node) for a in assignments} == {True, False}
        assert all(not is_child_of(node, a) for a in assignments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_attach_scopes_associates(frontend, tmp_path):
    fcode = """
module attach_scopes_associates_mod
    implicit none

    type other_type
        integer :: foo
    end type other_type

    type some_type
        type(other_type) :: var
    end type some_type

contains

    subroutine attach_scopes_associates
        type(some_type) :: blah
        integer :: a

        associate(var=>blah%var)
            associate(bar=>5+3)
                a = var%foo
            end associate
        end associate
    end subroutine attach_scopes_associates
end module attach_scopes_associates_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    routine = module['attach_scopes_associates']
    associates = FindNodes(ir.Associate).visit(routine.body)
    assert len(associates) == 2
    assignment = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assignment) == 1
    assert len(FindVariables().visit(assignment)) == 3
    var_map = {str(var): var for var in FindVariables().visit(assignment)}
    assert len(var_map) == 3
    assert associates[1].parent is associates[0]
    assert var_map['a'].scope is routine
    assert var_map['var%foo'].scope is associates[0]
    assert var_map['var%foo'].parent.scope is associates[0]
    assert var_map['var%foo'].parent is var_map['var']
