import pytest
from pymbolic.primitives import Expression

from loki import (
    OFP, OMNI, FP,
    Subroutine, Loop, Statement,
    Array, ArraySubscript, LoopRange, IntLiteral, FloatLiteral, LogicLiteral,
    FindVariables, ExpressionFinder, ExpressionCallbackMapper, ExpressionRetriever)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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

    variables = FindVariables(unique=False).visit(routine.ir)
    assert len(variables) == 12
    assert all(isinstance(v, Expression) for v in variables)

    assert sorted([str(v) for v in variables]) == (
        ['i'] * 6 + ['matrix(i, :)', 'scalar'] + ['vector(i)'] * 3 + ['x'])


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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

    variables = FindVariables().visit(routine.ir)
    assert isinstance(variables, set)
    assert len(variables) == 5
    assert all(isinstance(v, Expression) for v in variables)

    assert sorted([str(v) for v in variables]) == ['i', 'matrix(i, :)', 'scalar', 'vector(i)', 'x']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_expression_finder_with_root(frontend):
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

    variables = FindVariables(unique=False, with_expression_root=True).visit(routine.ir)
    assert len(variables) == 3
    assert all(isinstance(v, tuple) and len(v) == 2 for v in variables)

    # Verify that the variables in the loop definition are found
    loops = [v for v in variables if isinstance(v[0], Loop)]
    assert len(loops) == 1
    assert sorted([str(v) for v in loops[0][1]]) == ['i', 'x']

    # Verify that the variables in the statements are found
    stmts = [v for v in variables if isinstance(v[0], Statement)]
    assert len(stmts) == 2

    assert sorted([str(v) for v in stmts[0][1]]) == ['i', 'i', 'scalar', 'vector(i)', 'vector(i)']
    assert sorted([str(v) for v in stmts[1][1]]) == ['i', 'i', 'i', 'matrix(i, :)', 'vector(i)']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_expression_finder_unique_with_root(frontend):
    """
    Test the expression finder's ability to yield the root node combined with only unique
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

    variables = FindVariables(with_expression_root=True).visit(routine.ir)
    assert len(variables) == 3
    assert all(isinstance(v, tuple) and len(v) == 2 for v in variables)

    # Verify that the variables in the loop definition are found
    loops = [v for v in variables if isinstance(v[0], Loop)]
    assert len(loops) == 1
    assert sorted([str(v) for v in loops[0][1]]) == ['i', 'x']

    # Verify that the variables in the statements are found
    stmts = [v for v in variables if isinstance(v[0], Statement)]
    assert len(stmts) == 2

    assert sorted([str(v) for v in stmts[0][1]]) == ['i', 'scalar', 'vector(i)']
    assert sorted([str(v) for v in stmts[1][1]]) == ['i', 'matrix(i, :)', 'vector(i)']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
        if isinstance(expr, Array) and expr.type.shape and len(expr.type.shape) == 2:
            return expr
        return None

    retriever = ExpressionCallbackMapper(callback=is_matrix,
                                         combine=lambda v: tuple(e for e in v if e is not None))
    matrix_count = ExpressionFinder(retrieve=retriever, unique=False).visit(routine.ir)
    assert len(matrix_count) == 2

    matrix_count = ExpressionFinder(retrieve=retriever).visit(routine.ir)
    assert len(matrix_count) == 1
    assert str(matrix_count.pop()) == 'matrix(i, j)'


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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

    def retrieve(expr):
        var_types = (IntLiteral, FloatLiteral, LogicLiteral)
        excl_types = (ArraySubscript, LoopRange)
        retriever = ExpressionRetriever(
            lambda e: isinstance(e, var_types),
            recurse_query=lambda e, *args, **kwargs: not isinstance(e, excl_types))
        retriever(expr)
        return retriever.exprs

    literals = ExpressionFinder(unique=False, retrieve=retrieve).visit(routine.body)
    if frontend == OMNI:
        # OMNI substitutes jprb
        assert len(literals) == 4
        assert sorted([str(l) for l in literals]) == ['1.', '13', '2', '300']
    else:
        assert len(literals) == 2
        assert sorted([str(l) for l in literals]) == ['1.', '2']
