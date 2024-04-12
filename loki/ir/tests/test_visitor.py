# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pymbolic.primitives import Expression

from loki import Module, Subroutine, fgen
from loki.frontend import available_frontends, OMNI
from loki.ir.nodes import (
    Assignment, Associate, Conditional, Loop, Intrinsic, Section
)
from loki.ir import (
    is_parent_of, is_child_of, FindNodes, FindScopes, Transformer,
    NestedTransformer, MaskedTransformer, NestedMaskedTransformer,
    Stringifier
)
from loki.expression import (
    symbols as sym, FindVariables, ExpressionFinder,
    ExpressionCallbackMapper, ExpressionRetriever,
    SubstituteExpressions
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

    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(conditionals) == 2

    outer_cond = FindNodes(Conditional, greedy=True).visit(routine.body)
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

    intrinsics = FindNodes(Intrinsic).visit(routine.body)
    assert len(intrinsics) == 2
    inner = [i for i in intrinsics if 'Inner' in i.text][0]
    outer = [i for i in intrinsics if 'Outer' in i.text][0]

    conditionals = FindNodes(Conditional).visit(routine.body)
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
    loops = [v for v in variables if isinstance(v[0], Loop)]
    assert len(loops) == 1
    assert sorted([str(v) for v in loops[0][1]]) == ['i', 'x']

    # Verify that the variables in the statements are found
    stmts = [v for v in variables if isinstance(v[0], Assignment)]
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
    loops = [v for v in variables if isinstance(v[0], Loop)]
    assert len(loops) == 1
    assert sorted([str(v) for v in loops[0][1]]) == ['i', 'x']

    # Verify that the variables in the statements are found
    stmts = [v for v in variables if isinstance(v[0], Assignment)]
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
    assert len(variables) == 29
    assert len([v for v in variables if v.name == 'v']) == 1
    assert len([v for v in variables if v.name == 'm']) == 2


@pytest.mark.parametrize('frontend', available_frontends())
def test_stringifier(frontend):
    """
    Test basic stringifier capability for most IR nodes.
    """
    fcode = """
MODULE some_mod
  INTEGER :: n
  !$loki dimension(klon)
  REAL :: arr(:)
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
      y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1.
    END SUBROUTINE some_routine
    FUNCTION my_sqrt (arg)
      IMPLICIT NONE
      REAL, INTENT(IN) :: arg
      REAL :: my_sqrt
      my_sqrt = SQRT(arg)
    END FUNCTION my_sqrt
  SUBROUTINE other_routine (m)
    ! This is just to have some more IR nodes
    ! with multi-line comments and everything...
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: m
    REAL, ALLOCATABLE :: var(:)
    !$loki some pragma
    SELECT CASE (m)
      CASE (0)
        m = 1
      CASE (1:10)
        PRINT *, '1 to 10'
      CASE (-1, -2)
        m = 10
      CASE DEFAULT
        PRINT *, 'Default case'
    END SELECT
    ASSOCIATE (x => arr(m))
      x = x * 2.
    END ASSOCIATE
    ALLOCATE(var, source=arr)
    CALL some_routine (arr(1), var(1))
    arr(:) = arr(:) + var(:)
    DEALLOCATE(var)
  END SUBROUTINE other_routine
END MODULE some_mod
    """.strip()
    ref_lines = [
        "<Module:: some_mod>",  # l. 1
        "#<Section::>",
        "##<VariableDeclaration:: n>",
        "##<Pragma:: loki dimension(klon)>",
        "##<VariableDeclaration:: arr(:)>",
        "#<Subroutine:: some_routine>",
        "##<Comment:: ! This is a b...>",
        "##<Section::>",
        "###<Intrinsic:: IMPLICIT NONE>",
        "###<VariableDeclaration:: x>",  # l. 10
        "###<VariableDeclaration:: y>",
        "###<VariableDeclaration:: i>",
        "##<Section::>",
        "###<Comment:: ! And now to ...>",
        "###<Conditional::>",
        "####<If x < 1E-8 and x > -1E-8>",
        "#####<Assignment:: x = 0.>",
        "####<Else>",
        "#####<Conditional::>",
        "######<If x > 0.>",  # l. 20
        "#######<WhileLoop:: x > 1.>",
        "########<Assignment:: x = x / 2.>",
        "######<Else>",
        "#######<Assignment:: x = -x>",
        "###<Assignment:: y = 0>",
        "###<Loop:: i=1:n>",
        "####<Assignment:: y = y + x*x>",
        "###<Assignment:: y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + ",
        "... 1. + 1.>",
        "#<Function:: my_sqrt>", # l. 30
        "##<Section::>",
        "###<Intrinsic:: IMPLICIT NONE>",
        "###<VariableDeclaration:: arg>",
        "###<VariableDeclaration:: my_sqrt>",
        "##<Section::>",
        "###<Assignment:: my_sqrt = SQRT(arg)>",
        "#<Subroutine:: other_routine>",
        "##<CommentBlock:: ! This is jus...>",
        "##<Section::>",
        "###<Intrinsic:: IMPLICIT NONE>",  # l. 40
        "###<VariableDeclaration:: m>",
        "###<VariableDeclaration:: var(:)>",
        "##<Section::>",
        "###<Pragma:: loki some pragma>",
        "###<MultiConditional:: m>",
        "####<Case (0)>",
        "#####<Assignment:: m = 1>",
        "####<Case (1:10)>",
        "#####<Intrinsic:: PRINT *, '1 t...>",
        "####<Case (-1, -2)>",  # l. 50
        "#####<Assignment:: m = 10>",
        "####<Default>",
        "#####<Intrinsic:: PRINT *, 'Def...>",
        "###<Associate:: arr(m)=x>",
        "####<Assignment:: x = x*2.>",
        "###<Allocation:: var>",
        "###<Call:: some_routine>",
        "###<Assignment:: arr(:) = arr(:) + var(:)>",
        "###<Deallocation:: var>",
    ]

    if frontend == OMNI:
        # Some string inconsistencies
        ref_lines[15] = ref_lines[15].replace('1E-8', '1e-8')
        ref_lines[35] = ref_lines[35].replace('SQRT', 'sqrt')
        ref_lines[48] = ref_lines[48].replace('PRINT', 'print')
        ref_lines[52] = ref_lines[52].replace('PRINT', 'print')

    cont_index = 27  # line number where line continuation is happening
    ref = '\n'.join(ref_lines)
    module = Module.from_source(fcode, frontend=frontend)

    # Test custom indentation
    def line_cont(indent):
        return f'\n{"...":{max(len(indent), 1)}} '
    assert Stringifier(indent='#', line_cont=line_cont).visit(module).strip() == ref.strip()

    # Test default
    ref_lines = ref.strip().replace('#', '  ').splitlines()
    ref_lines[cont_index] = '      <Assignment:: y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1.'
    ref_lines[cont_index + 1] = '       + 1. + 1.>'
    default_ref = '\n'.join(ref_lines)
    assert Stringifier().visit(module).strip() == default_ref

    # Test custom initial depth
    ref_lines = ['#' + line if line else '' for line in ref.splitlines()]
    ref_lines[cont_index] = '####<Assignment:: y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. +'
    ref_lines[cont_index + 1] = '...   1. + 1.>'
    depth_ref = '\n'.join(ref_lines)
    assert Stringifier(indent='#', depth=1, line_cont=line_cont).visit(module).strip() == depth_ref

    # Test custom linewidth
    ref_lines = ref.strip().splitlines()
    ref_lines = ref_lines[:cont_index] + ['###<Assignment:: y = my_sqrt(y) + 1. + 1. +',
                                          '...  1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. ',
                                          '... + 1. + 1. + 1.>'] + ref_lines[cont_index+2:]
    w_ref = '\n'.join(ref_lines)
    assert Stringifier(indent='#', linewidth=44, line_cont=line_cont).visit(module).strip() == w_ref


@pytest.mark.parametrize('frontend', available_frontends())
def test_transformer_source_invalidation_replace(frontend):
    """
    Test basic transformer functionality and verify source invalidation
    when replacing nodes.
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
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Replace the innermost statement in the body of the conditional
    def get_innermost_statement(ir):
        for stmt in FindNodes(Assignment).visit(ir):
            if 'matrix' in str(stmt.lhs) and isinstance(stmt.rhs, sym.Sum):
                return stmt
        return None

    stmt = get_innermost_statement(routine.ir)
    new_expr = sym.Sum((*stmt.rhs.children[:-1], sym.FloatLiteral(2.)))
    new_stmt = Assignment(stmt.lhs, new_expr)
    mapper = {stmt: new_stmt}

    body_without_source = Transformer(mapper, invalidate_source=True).visit(routine.body)
    body_with_source = Transformer(mapper, invalidate_source=False).visit(routine.body)

    # Find the original and new node in all bodies
    orig_node = stmt
    node_without_src = get_innermost_statement(body_without_source)
    node_with_src = get_innermost_statement(body_with_source)

    # Check that source of new statement is untouched
    assert orig_node.source is not None
    assert node_without_src.source is None
    assert node_with_src.source is None

    # Check recursively the presence or absence of the source property
    while True:
        node_without_src = FindNodes(node_without_src, mode='scope').visit(body_without_source)[0]
        node_with_src = FindNodes(node_with_src, mode='scope').visit(body_with_source)[0]
        orig_node = FindNodes(orig_node, mode='scope').visit(routine.body)[0]
        if isinstance(orig_node, Section):
            assert isinstance(node_without_src, Section)
            assert isinstance(node_with_src, Section)
            break
        assert node_without_src.source is None
        assert node_with_src.source and node_with_src.source == orig_node.source

    # Check that else body is untouched
    def get_else_stmt(ir):
        return FindNodes(Conditional).visit(ir)[0].else_body[0]

    else_stmt = get_else_stmt(routine.body)
    assert else_stmt.source is not None
    assert get_else_stmt(body_without_source).source == else_stmt.source
    assert get_else_stmt(body_with_source).source == else_stmt.source


@pytest.mark.parametrize('frontend', available_frontends())
def test_transformer_source_invalidation_prepend(frontend):
    """
    Test basic transformer functionality and verify source invalidation
    when adding items to a loop body.
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
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Insert a new statement before the conditional
    def get_conditional(ir):
        return FindNodes(Conditional).visit(ir)[0]

    cond = get_conditional(routine.ir)
    new_stmt = Assignment(lhs=routine.arguments[0], rhs=routine.arguments[1])
    mapper = {cond: (new_stmt, cond)}

    body_without_source = Transformer(mapper, invalidate_source=True).visit(routine.body)
    body_with_source = Transformer(mapper, invalidate_source=False).visit(routine.body)

    # Find the conditional in new bodies and check that source is untouched
    assert cond.source is not None
    cond_without_src = get_conditional(body_without_source)
    assert cond_without_src.source is None
    cond_with_src = get_conditional(body_with_source)
    assert cond_with_src.source == cond.source

    # Find the newly inserted statement and check that source is None
    def get_new_statement(ir):
        for stmt in FindNodes(Assignment).visit(ir):
            if stmt.lhs == routine.arguments[0]:
                return stmt
        return None

    node_without_src = get_new_statement(body_without_source)
    assert node_without_src.source is None
    node_with_src = get_new_statement(body_with_source)
    assert node_with_src.source is None

    # Check recursively the presence or absence of the source property
    orig_node = cond
    while True:
        node_without_src = FindNodes(node_without_src, mode='scope').visit(body_without_source)[0]
        node_with_src = FindNodes(node_with_src, mode='scope').visit(body_with_source)[0]
        orig_node = FindNodes(orig_node, mode='scope').visit(routine.body)[0]
        if isinstance(orig_node, Section):
            assert isinstance(node_without_src, Section)
            assert isinstance(node_with_src, Section)
            break
        assert node_without_src.source is None
        assert node_with_src.source and node_with_src.source == orig_node.source


@pytest.mark.parametrize('frontend', available_frontends())
def test_transformer_rebuild(frontend):
    """
    Test basic transformer functionality with and without node rebuilding.
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
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Replace the innermost statement in the body of the conditional
    def get_innermost_statement(ir):
        for stmt in FindNodes(Assignment).visit(ir):
            if 'matrix' in str(stmt.lhs) and isinstance(stmt.rhs, sym.Sum):
                return stmt
        return None

    stmt = get_innermost_statement(routine.ir)
    new_expr = sym.Sum((*stmt.rhs.children[:-1], sym.FloatLiteral(2.)))
    new_stmt = Assignment(stmt.lhs, new_expr)
    mapper = {stmt: new_stmt}

    loops = FindNodes(Loop).visit(routine.body)
    conds = FindNodes(Conditional).visit(routine.body)

    # Check that all loops and conditionals around statements are rebuilt
    body_rebuild = Transformer(mapper, inplace=False).visit(routine.body)
    stmts_rebuild = [str(s) for s in FindNodes(Assignment).visit(body_rebuild)]
    loops_rebuild = FindNodes(Loop).visit(body_rebuild)
    conds_rebuild = FindNodes(Conditional).visit(body_rebuild)
    assert str(stmt) not in stmts_rebuild
    assert str(new_stmt) in stmts_rebuild
    assert not any(l in loops for l in loops_rebuild)
    assert not any(c in conds for c in conds_rebuild)

    # Check that no loops or conditionals around statements are rebuilt
    body_no_rebuild = Transformer(mapper, inplace=True).visit(routine.body)
    stmts_no_rebuild = [str(s) for s in FindNodes(Assignment).visit(body_no_rebuild)]
    loops_no_rebuild = FindNodes(Loop).visit(body_no_rebuild)
    conds_no_rebuild = FindNodes(Conditional).visit(body_no_rebuild)
    assert str(stmt) not in stmts_no_rebuild
    assert str(new_stmt) in stmts_no_rebuild
    assert all(l in loops for l in loops_no_rebuild)
    assert all(c in conds for c in conds_no_rebuild)

    # Check that no loops or conditionals around statements are rebuilt,
    # even if source_invalidation is deactivated
    body_no_rebuild = Transformer(mapper, invalidate_source=False, inplace=True).visit(routine.body)
    stmts_no_rebuild = [str(s) for s in FindNodes(Assignment).visit(body_no_rebuild)]
    loops_no_rebuild = FindNodes(Loop).visit(body_no_rebuild)
    conds_no_rebuild = FindNodes(Conditional).visit(body_no_rebuild)
    assert str(stmt) not in stmts_no_rebuild
    assert str(new_stmt) in stmts_no_rebuild
    assert all(l in loops for l in loops_no_rebuild)
    assert all(c in conds for c in conds_no_rebuild)


@pytest.mark.parametrize('frontend', available_frontends())
def test_transformer_multinode_keys(frontend):
    """
    Test basic transformer functionality with nulti-node keys
    """
    fcode = """
subroutine routine_simple (x, y, a, b, c, d, e)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, intent(in) :: x, y
  real(kind=jprb), intent(in) :: a(x), b(x), c(x), d(x), e(x)
  integer :: i

  b(i) = a(i) + 1.
  c(i) = a(i) + 2.
  d(i) = c(i) + 3.
  e(i) = d(i) + 4.
end subroutine routine_simple
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    assigns = FindNodes(Assignment).visit(routine.body)
    bounds = sym.LoopRange((sym.IntLiteral(1), routine.variable_map['x']))

    # Filter out only the two middle assignments to wrap in a loop.
    # Note that we need to be careful to clone loop body nodes to
    # avoid infinite recursion.
    assigns = tuple(a for a in assigns if a.lhs in ['c(i)', 'd(i)'])
    loop = Loop(variable=routine.variable_map['i'], bounds=bounds,
                body=tuple(a.clone() for a in assigns))
    # Need to use NestedTransformer here, since replacement contains
    # the original nodes.
    transformed = NestedTransformer({assigns: loop}).visit(routine.body)

    new_loops = FindNodes(Loop).visit(transformed)
    assert len(new_loops) == 1
    assert len(FindNodes(Assignment).visit(new_loops)) == 2
    assert len(FindNodes(Assignment).visit(transformed)) == 4


@pytest.mark.parametrize('frontend', available_frontends())
def test_masked_transformer(frontend):
    """
    A very basic sanity test for the MaskedTransformer class.
    """
    fcode = """
subroutine masked_transformer(a)
  integer, intent(inout) :: a

  a = a + 1
  a = a + 2
  a = a + 3
  a = a + 4
  a = a + 5
  a = a + 6
  a = a + 7
  a = a + 8
  a = a + 9
  a = a + 10
end subroutine masked_transformer
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assignments = FindNodes(Assignment).visit(routine.body)

    # Removes all nodes
    body = MaskedTransformer(start=None, stop=None).visit(routine.body)
    assert not FindNodes(Assignment).visit(body)

    # Retains all nodes
    body = MaskedTransformer(start=None, stop=None, active=True).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 10

    # Removes all nodes but the last
    body = MaskedTransformer(start=assignments[-1], stop=None).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 1

    # Retains all nodes but the last
    body = MaskedTransformer(start=None, stop=assignments[-1], active=True).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == len(assignments) - 1

    # Retains the first two and last two nodes
    start = [assignments[0], assignments[-2]]
    stop = assignments[2]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 4

    # Retains the first two and the second to last node
    start = [assignments[0], assignments[-2]]
    stop = [assignments[2], assignments[-1]]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 3

    # Retains three nodes in the middle
    start = assignments[3]
    stop = assignments[6]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 3

    # Retains nodes two to four and replaces the third by the first node
    start = assignments[1]
    stop = assignments[4]
    mapper = {assignments[2]: assignments[0]}
    body = MaskedTransformer(start=start, stop=stop, mapper=mapper).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 3
    assert str(FindNodes(Assignment).visit(body)[1]) == str(assignments[0])

    # Retains nodes two to four and replaces the second by the first node
    start = assignments[1]
    stop = assignments[4]
    mapper = {assignments[1]: assignments[0]}
    body = MaskedTransformer(start=start, stop=stop, mapper=mapper).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 3
    assert str(FindNodes(Assignment).visit(body)[0]) == str(assignments[0])


@pytest.mark.parametrize('frontend', available_frontends())
def test_masked_transformer_minimum_set(frontend):
    """
    A very basic sanity test for the MaskedTransformer class with
    require_all_start or greedy_stop properties.
    """
    fcode = """
subroutine masked_transformer_minimum_set(a)
  integer, intent(inout) :: a

  a = a + 1
  a = a + 2
  a = a + 3
  a = a + 4
  a = a + 5
  a = a + 6
  a = a + 7
  a = a + 8
  a = a + 9
  a = a + 10
end subroutine masked_transformer_minimum_set
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assignments = FindNodes(Assignment).visit(routine.body)

    # Requires all nodes and thus retains only the last
    body = MaskedTransformer(start=assignments, require_all_start=True).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 1
    assert fgen(body) == fgen(assignments[-1])

    # Retains only the second node
    body = MaskedTransformer(start=assignments[:2], stop=assignments[2], require_all_start=True).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 1
    assert fgen(body) == fgen(assignments[1])

    # Retains only first node
    body = MaskedTransformer(start=assignments, stop=assignments[1], greedy_stop=True).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 1
    assert fgen(body) == fgen(assignments[0])


@pytest.mark.parametrize('frontend', available_frontends())
def test_masked_transformer_associates(frontend):
    """
    Test the masked transformer in conjunction with associate blocks
    """
    fcode = """
subroutine masked_transformer(a)
  integer, intent(inout) :: a

associate(b=>a)
  b = b + 1
  b = b + 2
  b = b + 3
  b = b + 4
  b = b + 5
  b = b + 6
  b = b + 7
  b = b + 8
  b = b + 9
  b = b + 10
end associate
end subroutine masked_transformer
    """

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assignments = FindNodes(Assignment).visit(routine.body)
    assert len(assignments) == 10
    assert len(FindNodes(Associate).visit(routine.body)) == 1

    # Removes all nodes
    body = MaskedTransformer(start=None, stop=None).visit(routine.body)
    assert not FindNodes(Assignment).visit(body)
    assert not FindNodes(Associate).visit(body)

    # Removes all nodes but the last
    body = MaskedTransformer(start=assignments[-1], stop=None).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 1
    assert not FindNodes(Associate).visit(body)

    # Retains all nodes but the last
    body = MaskedTransformer(start=None, stop=assignments[-1], active=True).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == len(assignments) - 1
    assert len(FindNodes(Associate).visit(body)) == 1

    # Retains the first two and last two nodes
    start = [assignments[0], assignments[-2]]
    stop = assignments[2]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 4
    assert not FindNodes(Associate).visit(body)

    # Retains the first two and the second to last node
    start = [assignments[0], assignments[-2]]
    stop = [assignments[2], assignments[-1]]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 3
    assert not FindNodes(Associate).visit(body)

    # Retains three nodes in the middle
    start = assignments[3]
    stop = assignments[6]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 3
    assert not FindNodes(Associate).visit(body)

    # Retains all nodes but the last, but check with ``inplace=True``
    body = MaskedTransformer(start=None, stop=assignments[-1], active=True, inplace=True).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == len(assignments) - 1
    assocs = FindNodes(Associate).visit(body)
    assert len(assocs) == 1
    assert len(assocs[0].body) == len(assignments) - 1
    assert all(isinstance(n, Assignment) for n in assocs[0].body)


@pytest.mark.parametrize('frontend', available_frontends())
def test_nested_masked_transformer(frontend):
    """
    Test the masked transformer in conjunction with nesting
    """
    fcode = """
subroutine nested_masked_transformer
  implicit none
  integer :: a=0, b, c, d
  integer :: i, j

  do i=1,10
    a = a + i
    if (a < 5) then
      b = 0
    else if (a == 5) then
      c = 0
    else
      do j=1,5
        d = a
      end do
    end if
  end do
end subroutine nested_masked_transformer
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend)
    assignments = FindNodes(Assignment).visit(routine.body)
    loops = FindNodes(Loop).visit(routine.body)
    conditionals = FindNodes(Conditional).visit(routine.body)
    assert len(assignments) == 4
    assert len(loops) == 2
    assert len(conditionals) == 2 if frontend == OMNI else 1

    # Drops the outermost loop
    start = [a for a in assignments if a.lhs == 'a']
    body = MaskedTransformer(start=start).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 4
    assert len(FindNodes(Loop).visit(body)) == 1
    assert len(FindNodes(Conditional).visit(body)) == len(conditionals)

    # Should produce the original version
    body = NestedMaskedTransformer(start=start).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 4
    assert len(FindNodes(Loop).visit(body)) == 2
    assert len(FindNodes(Conditional).visit(body)) == len(conditionals)
    assert fgen(routine.body).strip() == fgen(body).strip()

    # Should drop the first assignment
    body = NestedMaskedTransformer(start=conditionals[0]).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 3
    assert len(FindNodes(Loop).visit(body)) == 2
    assert len(FindNodes(Conditional).visit(body)) == len(conditionals)

    # Should leave no more than a single assignment
    start = [a for a in assignments if a.lhs == 'c']
    stop = [l for l in loops if l.variable == 'j']
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert fgen(start).strip() == fgen(body).strip()

    # Should leave a single assignment with the hierarchy of nested sections
    # in the else-if branch
    body = NestedMaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 1
    assert len(FindNodes(Loop).visit(body)) == 1
    assert len(FindNodes(Conditional).visit(body)) == 1

    # Should leave no more than a single assignment
    start = [a for a in assignments if a.lhs == 'd']
    body = MaskedTransformer(start=start, stop=start).visit(routine.body)
    assert fgen(start).strip() == fgen(body).strip()

    # Should leave a single assignment with the hierarchy of nested sections
    # in the else branch
    body = NestedMaskedTransformer(start=start, stop=start).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 1
    assert len(FindNodes(Loop).visit(body)) == 2
    assert len(FindNodes(Conditional).visit(body)) == 0

    # Should produce the original body
    start = [a for a in assignments if a.lhs in ('a', 'd')]
    body = NestedMaskedTransformer(start=start).visit(routine.body)
    assert fgen(routine.body).strip() == fgen(body).strip()

    # Should leave a single assignment with the hierarchy of nested sections
    # in the else branch
    body = NestedMaskedTransformer(start=start, require_all_start=True).visit(routine.body)
    assert [a.lhs == 'd' for a in FindNodes(Assignment).visit(body)] == [True]
    assert len(FindNodes(Loop).visit(body)) == 2
    assert len(FindNodes(Conditional).visit(body)) == 0

    # Drops everything
    stop = [a for a in assignments if a.lhs == 'a']
    body = NestedMaskedTransformer(start=start, stop=stop, greedy_stop=True).visit(routine.body)
    assert not body

    # Should drop the else-if branch
    start = [a for a in assignments if a.lhs in ('b', 'd')]
    stop = [a for a in assignments if a.lhs == 'c']
    body = NestedMaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(Assignment).visit(body)) == 2
    assert len(FindNodes(Loop).visit(body)) == 2
    assert len(FindNodes(Conditional).visit(body)) == 1

    # Should drop everything buth the if branch
    body = NestedMaskedTransformer(start=start, stop=stop, greedy_stop=True).visit(routine.body)
    assert [a.lhs == 'b' for a in FindNodes(Assignment).visit(body)] == [True]
    assert len(FindNodes(Loop).visit(body)) == 1
    assert len(FindNodes(Conditional).visit(body)) == 1


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

    loop = FindNodes(Loop).visit(routine.body)[0]
    conditional = FindNodes(Conditional).visit(routine.body)[0]
    assignments = FindNodes(Assignment).visit(routine.body)

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

    loop = FindNodes(Loop).visit(routine.body)[0]
    conditional = FindNodes(Conditional).visit(routine.body)[0]
    assignments = FindNodes(Assignment).visit(routine.body)

    assert not is_child_of(loop, conditional)
    assert is_child_of(conditional, loop)

    for node in [loop, conditional]:
        assert {is_child_of(a, node) for a in assignments} == {True, False}
        assert all(not is_child_of(node, a) for a in assignments)


@pytest.mark.parametrize('frontend', available_frontends())
def test_attach_scopes_associates(frontend):
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

    module = Module.from_source(fcode, frontend=frontend)
    routine = module['attach_scopes_associates']
    associates = FindNodes(Associate).visit(routine.body)
    assert len(associates) == 2
    assignment = FindNodes(Assignment).visit(routine.body)
    assert len(assignment) == 1
    assert len(FindVariables().visit(assignment)) == 3
    var_map = {str(var): var for var in FindVariables().visit(assignment)}
    assert len(var_map) == 3
    assert associates[1].parent is associates[0]
    assert var_map['a'].scope is routine
    assert var_map['var%foo'].scope is associates[0]
    assert var_map['var%foo'].parent.scope is associates[0]
    assert var_map['var%foo'].parent is var_map['var']


@pytest.mark.parametrize('invalidate_source', [True, False])
@pytest.mark.parametrize('replacement', ['body', 'self', 'self_tuple', 'duplicate'])
@pytest.mark.parametrize('frontend', available_frontends())
def test_transformer_duplicate_node_tuple_injection(frontend, invalidate_source, replacement):
    """Test for #41, where identical nodes in a tuple have not been
    correctly handled in the tuple injection mechanism."""
    fcode_kernel = """
SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end
    INTEGER, INTENT(IN) :: nlon, nz
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl
    DO JL = START, END
        Q(JL, NZ) = Q(JL, NZ) * 0.5
    END DO
    DO JL = START, END
        Q(JL, NZ) = Q(JL, NZ) * 0.5
    END DO
END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Empty substitution pass, which invalidates the source property
    kernel.body = SubstituteExpressions({}, invalidate_source=invalidate_source).visit(kernel.body)

    loops = FindNodes(Loop).visit(kernel.body)
    if replacement == 'body':
        # Replace loop by its body
        mapper = {l: l.body for l in loops}
    elif replacement == 'self':
        # Replace loop by itself
        mapper = {l: l for l in loops}
    elif replacement == 'self_tuple':
        # Replace loop by itself, but wrapped in a tuple
        mapper = {l: (l,) for l in loops}
    elif replacement == 'duplicate':
        # Duplicate the loop (will this trigger infinite recursion in tuple injection)?
        mapper = {l: (l, l) for l in loops}
    else:
        # We shouldn't be here!
        assert False
    kernel.body = Transformer(mapper).visit(kernel.body)
    # Make sure we don't have any nested tuples or similar nasty things, which would
    # cause a transformer pass to fail
    kernel.body = Transformer({}).visit(kernel.body)
    # If the code gen works, then it's probably not too broken...
    assert kernel.to_fortran()
    # Make sure the number of loops is correct
    assert len(FindNodes(Loop).visit(kernel.body)) == {
        'body': 0, # All loops replaced by the body
        'self': 2, 'self_tuple': 2,  # Loop replaced by itself
        'duplicate': 4  # Loops duplicated
    }[replacement]
