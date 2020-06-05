import pytest
from pymbolic.primitives import Expression

from loki import (
    OFP, OMNI, FP,
    Module, Subroutine, Loop, Statement, Conditional,
    Array, ArraySubscript, LoopRange, IntLiteral, FloatLiteral, LogicLiteral, Comparison, Cast,
    FindNodes, FindExpressions, FindVariables, ExpressionFinder, FindExpressionRoot,
    ExpressionCallbackMapper, retrieve_expressions, Stringifier)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 2

    outer_cond = FindNodes(Conditional, greedy=True).visit(routine.body)
    assert len(outer_cond) == 1
    assert outer_cond[0] in conditionals
    assert str(outer_cond[0].conditions[0]) == 'n > m'


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

    variables = FindVariables(unique=False).visit(routine.body)
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

    variables = FindVariables().visit(routine.body)
    assert isinstance(variables, set)
    assert len(variables) == 5
    assert all(isinstance(v, Expression) for v in variables)

    assert sorted([str(v) for v in variables]) == ['i', 'matrix(i, :)', 'scalar', 'vector(i)', 'x']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    loops = [v for v in variables if isinstance(v[0], Loop)]
    assert len(loops) == 1
    assert sorted([str(v) for v in loops[0][1]]) == ['i', 'x']

    # Verify that the variables in the statements are found
    stmts = [v for v in variables if isinstance(v[0], Statement)]
    assert len(stmts) == 2

    assert sorted([str(v) for v in stmts[0][1]]) == ['i', 'i', 'scalar', 'vector(i)', 'vector(i)']
    assert sorted([str(v) for v in stmts[1][1]]) == ['i', 'i', 'i', 'matrix(i, :)', 'vector(i)']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
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
    matrix_count = ExpressionFinder(retrieve=retriever, unique=False).visit(routine.body)
    assert len(matrix_count) == 2

    matrix_count = ExpressionFinder(retrieve=retriever).visit(routine.body)
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

    # Find all literals except when they appear in array subscripts or loop ranges
    cond = lambda expr: isinstance(expr, (IntLiteral, FloatLiteral, LogicLiteral))
    recurse_cond = lambda expr, *args, **kwargs: not isinstance(expr, (ArraySubscript, LoopRange))
    retrieve = lambda expr: retrieve_expressions(expr, cond=cond, recurse_cond=recurse_cond)
    literals = ExpressionFinder(unique=False, retrieve=retrieve).visit(routine.body)

    if frontend == OMNI:
        # OMNI substitutes jprb
        assert len(literals) == 4
        assert sorted([str(l) for l in literals]) == ['1.', '13', '2', '300']
    else:
        assert len(literals) == 2
        assert sorted([str(l) for l in literals]) == ['1.', '2']


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_find_expression_root(frontend):
    """
    Test basic functionality of FindExpressionRoot.
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

    exprs = FindExpressions().visit(routine.body)
    assert len(exprs) == 25 if frontend == OMNI else 21  # OMNI substitutes jprb in the Cast

    # Test ability to find root if searching for root
    comps = [e for e in exprs if isinstance(e, Comparison)]
    assert len(comps) == 1
    comp_root = FindExpressionRoot(comps[0]).visit(routine.body)
    assert len(comp_root) == 1
    assert comp_root[0] is comps[0]

    # Test ability to find root if searching for intermediate expression
    casts = [e for e in exprs if isinstance(e, Cast)]
    assert len(casts) == 1
    cast_root = FindExpressionRoot(casts[0]).visit(routine.body)
    assert len(cast_root) == 1
    cond = FindNodes(Conditional).visit(routine.body).pop()
    assert cast_root[0] is cond.bodies[0][0].expr

    # Test ability to find root if searching for a leaf expression
    literals = ExpressionFinder(
        retrieve=lambda e: retrieve_expressions(e, lambda _e: isinstance(_e, FloatLiteral)),
        with_ir_node=True).visit(routine.body)
    assert len(literals) == 1
    assert isinstance(literals[0][0], Statement) and literals[0][0]._source.lines == (13, 13)

    literal_root = FindExpressionRoot(literals[0][1].pop()).visit(literals[0][0])
    assert literal_root[0] is cast_root[0]

    # Check source properties of expressions (for FP only)
    if frontend == FP:
        assert comp_root[0].source.lines == (12, 12)
        assert cast_root[0].source.lines == (13, 13)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_stringifier(frontend):
    """
    Test basic stringifier capability.
    """
    fcode = """
MODULE some_mod
  INTEGER :: n
  CONTAINS
    SUBROUTINE some_routine (x, y)
      ! This is a basic subroutine with some loops
      IMPLICIT NONE
      REAL, INTENT(IN) :: x
      REAL, INTENT(OUT) :: y
      INTEGER :: i
      ! And now to the content
      IF (x < 1E-8 .and. x > -1E-8) THEN
        x = 0.
      ELSE IF (x > 0.) THEN
        DO WHILE (x > 1.)
          x = x / 2.
        ENDDO
      ELSE
        x = -x
      ENDIF
      y = 0
      DO i=1,n
        y = y + x*x
      ENDDO
      y = my_sqrt(y)
    END SUBROUTINE some_routine
    FUNCTION my_sqrt (arg)
      IMPLICIT NONE
      REAL, INTENT(IN) :: arg
      REAL :: my_sqrt
      my_sqrt = SQRT(arg)
    END FUNCTION my_sqrt
END MODULE some_mod
    """.strip()
    ref = """
<Module some_mod>
#<Section>
##<Declaration n>
#<Subroutine some_routine>
##<Comment:: ...>
##<Section>
###<Intrinsic:: IMPLICIT NONE>
###<Declaration x>
###<Declaration y>
###<Declaration i>
##<Section>
###<Comment:: ...>
###<Conditional>
####<If x < 1E-8 and x > -1E-8>
#####<Stmt:: x = 0.>
####<ElseIf x > 0.>
#####<WhileLoop x > 1.>
######<Stmt:: x = x / 2.>
####<Else>
#####<Stmt:: x = -x>
###<Stmt:: y = 0>
###<Loop i=1:n>
####<Stmt:: y = y + x*x>
###<Stmt:: y = my_sqrt(y)>
#<Function my_sqrt>
##<Section>
###<Intrinsic:: IMPLICIT NONE>
###<Declaration arg>
###<Declaration my_sqrt>
##<Section>
###<Stmt:: my_sqrt = SQRT(arg)>
    """.strip()

    if frontend == OMNI:
        ref_lines = ref.splitlines()
        # Replace ElseIf branch by nested if
        ref_lines = ref_lines[:15] + ['####<Else>', '#####<Conditional>'] + ref_lines[15:]  # Insert Conditional
        ref_lines[17] = ref_lines[17].replace('Else', '')  # ElseIf -> If
        ref_lines[17:22] = ['##' + line for line in ref_lines[17:22]]  # -> Indent
        # Some string inconsistencies
        ref_lines[13] = ref_lines[13].replace('1E-8', '1e-8')
        ref_lines[23] = ref_lines[23].replace('1:n', '1:n:1')
        ref_lines[32] = ref_lines[32].replace('SQRT', 'sqrt')
        ref = '\n'.join(ref_lines)

    module = Module.from_source(fcode, frontend=frontend)

    # Test custom indentation
    assert Stringifier(indent='#').visit(module).strip() == ref.strip()

    # Test default
    assert Stringifier().visit(module).strip() == ref.strip().replace('#', '  ')

    # Test custom initial depth
    ref_lines = ['#' + line if line else '' for line in ref.splitlines()]
    assert Stringifier(indent='#', depth=1).visit(module).strip() == '\n'.join(ref_lines).strip()
