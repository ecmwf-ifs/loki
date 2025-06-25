# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, fgen
from loki.frontend import available_frontends, OMNI, SourceStatus
from loki.ir import (
    nodes as ir, FindNodes, Transformer, NestedTransformer,
    MaskedTransformer, NestedMaskedTransformer, SubstituteExpressions
)
from loki.expression import symbols as sym


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
    def get_innermost_statement(nodes):
        for stmt in FindNodes(ir.Assignment).visit(nodes):
            if 'matrix' in str(stmt.lhs) and isinstance(stmt.rhs, sym.Sum):
                return stmt
        return None

    stmt = get_innermost_statement(routine.ir)
    new_expr = sym.Sum((*stmt.rhs.children[:-1], sym.FloatLiteral(2.)))

    # Check source invalidation via status flags
    mapper = {stmt: stmt.clone(rhs=new_expr, source=stmt.source.clone().invalidate())}
    body_invalid_source = Transformer(mapper, invalidate_source=True).visit(routine.body)

    # Check that original source has not been modified
    assert stmt.source and stmt.source.status == SourceStatus.VALID

    # Check that directly and indirectly affected nodes have been invalidated
    assigns = FindNodes(ir.Assignment).visit(body_invalid_source)
    assert len(assigns) == 3
    assert assigns[0].source.status == SourceStatus.VALID
    assert assigns[1].source.status == SourceStatus.INVALID_NODE
    assert assigns[2].source.status == SourceStatus.VALID

    loops = FindNodes(ir.Loop).visit(body_invalid_source)
    assert len(loops) == 2
    assert loops[0].source.status == SourceStatus.INVALID_CHILDREN
    assert loops[1].source.status == SourceStatus.INVALID_CHILDREN

    conds = FindNodes(ir.Conditional).visit(body_invalid_source)
    assert len(conds) == 1
    assert conds[0].source.status == SourceStatus.INVALID_CHILDREN

    # Check manual source removal without invalidation
    mapper = {stmt: stmt.clone(rhs=new_expr, source=None)}
    body_valid_source = Transformer(mapper, invalidate_source=False).visit(routine.body)

    assigns = FindNodes(ir.Assignment).visit(body_valid_source)
    assert len(assigns) == 3
    assert assigns[0].source.status == SourceStatus.VALID
    assert assigns[1].source is None
    assert assigns[2].source.status == SourceStatus.VALID

    loops = FindNodes(ir.Loop).visit(body_valid_source)
    assert len(loops) == 2
    assert loops[0].source.status == SourceStatus.VALID
    assert loops[1].source.status == SourceStatus.VALID

    conds = FindNodes(ir.Conditional).visit(body_valid_source)
    assert len(conds) == 1
    assert conds[0].source.status == SourceStatus.VALID


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
    def get_conditional(nodes):
        return FindNodes(ir.Conditional).visit(nodes)[0]

    cond = get_conditional(routine.ir)
    new_stmt = ir.Assignment(lhs=routine.arguments[0], rhs=routine.arguments[1])
    mapper = {cond: (new_stmt, cond.clone(source=cond.source.clone().invalidate()))}

    body_invalid_source = Transformer(mapper, invalidate_source=True).visit(routine.body)

    # Check that original source has not been modified
    assert cond.source and cond.source.status == SourceStatus.VALID
    assert not new_stmt.source

    # Check that directly and indirectly affected nodes have been invalidated
    assigns = FindNodes(ir.Assignment).visit(body_invalid_source)
    assert len(assigns) == 4
    assert assigns[0].source.status == SourceStatus.VALID
    assert not assigns[1].source
    assert assigns[2].source.status == SourceStatus.VALID
    assert assigns[2].source.status == SourceStatus.VALID

    loops = FindNodes(ir.Loop).visit(body_invalid_source)
    assert len(loops) == 2
    assert loops[0].source.status == SourceStatus.INVALID_CHILDREN
    assert loops[1].source.status == SourceStatus.INVALID_CHILDREN

    conds = FindNodes(ir.Conditional).visit(body_invalid_source)
    assert len(conds) == 1
    assert conds[0].source.status == SourceStatus.INVALID_NODE

    # Check manual source removal without invalidation
    mapper = {cond: (new_stmt, cond.clone())}
    body_valid_source = Transformer(mapper, invalidate_source=False).visit(routine.body)

    assigns = FindNodes(ir.Assignment).visit(body_valid_source)
    assert len(assigns) == 4
    assert assigns[0].source.status == SourceStatus.VALID
    assert not assigns[1].source
    assert assigns[2].source.status == SourceStatus.VALID
    assert assigns[2].source.status == SourceStatus.VALID

    loops = FindNodes(ir.Loop).visit(body_valid_source)
    assert len(loops) == 2
    assert loops[0].source.status == SourceStatus.VALID
    assert loops[1].source.status == SourceStatus.VALID

    conds = FindNodes(ir.Conditional).visit(body_valid_source)
    assert len(conds) == 1
    assert conds[0].source.status == SourceStatus.VALID


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
    def get_innermost_statement(nodes):
        for stmt in FindNodes(ir.Assignment).visit(nodes):
            if 'matrix' in str(stmt.lhs) and isinstance(stmt.rhs, sym.Sum):
                return stmt
        return None

    stmt = get_innermost_statement(routine.ir)
    new_expr = sym.Sum((*stmt.rhs.children[:-1], sym.FloatLiteral(2.)))
    new_stmt = ir.Assignment(stmt.lhs, new_expr)
    mapper = {stmt: new_stmt}

    loops = FindNodes(ir.Loop).visit(routine.body)
    conds = FindNodes(ir.Conditional).visit(routine.body)

    # Check that all loops and conditionals around statements are rebuilt
    body_rebuild = Transformer(mapper, inplace=False).visit(routine.body)
    stmts_rebuild = [str(s) for s in FindNodes(ir.Assignment).visit(body_rebuild)]
    loops_rebuild = FindNodes(ir.Loop).visit(body_rebuild)
    conds_rebuild = FindNodes(ir.Conditional).visit(body_rebuild)
    assert str(stmt) not in stmts_rebuild
    assert str(new_stmt) in stmts_rebuild
    assert not any(l in loops for l in loops_rebuild)
    assert not any(c in conds for c in conds_rebuild)

    # Check that no loops or conditionals around statements are rebuilt
    body_no_rebuild = Transformer(mapper, inplace=True).visit(routine.body)
    stmts_no_rebuild = [str(s) for s in FindNodes(ir.Assignment).visit(body_no_rebuild)]
    loops_no_rebuild = FindNodes(ir.Loop).visit(body_no_rebuild)
    conds_no_rebuild = FindNodes(ir.Conditional).visit(body_no_rebuild)
    assert str(stmt) not in stmts_no_rebuild
    assert str(new_stmt) in stmts_no_rebuild
    assert all(l in loops for l in loops_no_rebuild)
    assert all(c in conds for c in conds_no_rebuild)

    # Check that no loops or conditionals around statements are rebuilt,
    # even if source_invalidation is deactivated
    body_no_rebuild = Transformer(mapper, invalidate_source=False, inplace=True).visit(routine.body)
    stmts_no_rebuild = [str(s) for s in FindNodes(ir.Assignment).visit(body_no_rebuild)]
    loops_no_rebuild = FindNodes(ir.Loop).visit(body_no_rebuild)
    conds_no_rebuild = FindNodes(ir.Conditional).visit(body_no_rebuild)
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
    assigns = FindNodes(ir.Assignment).visit(routine.body)
    bounds = sym.LoopRange((sym.IntLiteral(1), routine.variable_map['x']))

    # Filter out only the two middle assignments to wrap in a loop.
    # Note that we need to be careful to clone loop body nodes to
    # avoid infinite recursion.
    assigns = tuple(a for a in assigns if a.lhs in ['c(i)', 'd(i)'])
    loop = ir.Loop(variable=routine.variable_map['i'], bounds=bounds,
                   body=tuple(a.clone() for a in assigns))
    # Need to use NestedTransformer here, since replacement contains
    # the original nodes.
    transformed = NestedTransformer({assigns: loop}).visit(routine.body)

    new_loops = FindNodes(ir.Loop).visit(transformed)
    assert len(new_loops) == 1
    assert len(FindNodes(ir.Assignment).visit(new_loops)) == 2
    assert len(FindNodes(ir.Assignment).visit(transformed)) == 4


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
    assignments = FindNodes(ir.Assignment).visit(routine.body)

    # Removes all nodes
    body = MaskedTransformer(start=None, stop=None).visit(routine.body)
    assert not FindNodes(ir.Assignment).visit(body)

    # Retains all nodes
    body = MaskedTransformer(start=None, stop=None, active=True).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 10

    # Removes all nodes but the last
    body = MaskedTransformer(start=assignments[-1], stop=None).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 1

    # Retains all nodes but the last
    body = MaskedTransformer(start=None, stop=assignments[-1], active=True).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == len(assignments) - 1

    # Retains the first two and last two nodes
    start = [assignments[0], assignments[-2]]
    stop = assignments[2]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 4

    # Retains the first two and the second to last node
    start = [assignments[0], assignments[-2]]
    stop = [assignments[2], assignments[-1]]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 3

    # Retains three nodes in the middle
    start = assignments[3]
    stop = assignments[6]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 3

    # Retains nodes two to four and replaces the third by the first node
    start = assignments[1]
    stop = assignments[4]
    mapper = {assignments[2]: assignments[0]}
    body = MaskedTransformer(start=start, stop=stop, mapper=mapper).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 3
    assert str(FindNodes(ir.Assignment).visit(body)[1]) == str(assignments[0])

    # Retains nodes two to four and replaces the second by the first node
    start = assignments[1]
    stop = assignments[4]
    mapper = {assignments[1]: assignments[0]}
    body = MaskedTransformer(start=start, stop=stop, mapper=mapper).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 3
    assert str(FindNodes(ir.Assignment).visit(body)[0]) == str(assignments[0])


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
    assignments = FindNodes(ir.Assignment).visit(routine.body)

    # Requires all nodes and thus retains only the last
    body = MaskedTransformer(start=assignments, require_all_start=True).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 1
    assert fgen(body) == fgen(assignments[-1])

    # Retains only the second node
    body = MaskedTransformer(start=assignments[:2], stop=assignments[2], require_all_start=True).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 1
    assert fgen(body) == fgen(assignments[1])

    # Retains only first node
    body = MaskedTransformer(start=assignments, stop=assignments[1], greedy_stop=True).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 1
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
    assignments = FindNodes(ir.Assignment).visit(routine.body)
    assert len(assignments) == 10
    assert len(FindNodes(ir.Associate).visit(routine.body)) == 1

    # Removes all nodes
    body = MaskedTransformer(start=None, stop=None).visit(routine.body)
    assert not FindNodes(ir.Assignment).visit(body)
    assert not FindNodes(ir.Associate).visit(body)

    # Removes all nodes but the last
    body = MaskedTransformer(start=assignments[-1], stop=None).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 1
    assert not FindNodes(ir.Associate).visit(body)

    # Retains all nodes but the last
    body = MaskedTransformer(start=None, stop=assignments[-1], active=True).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == len(assignments) - 1
    assert len(FindNodes(ir.Associate).visit(body)) == 1

    # Retains the first two and last two nodes
    start = [assignments[0], assignments[-2]]
    stop = assignments[2]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 4
    assert not FindNodes(ir.Associate).visit(body)

    # Retains the first two and the second to last node
    start = [assignments[0], assignments[-2]]
    stop = [assignments[2], assignments[-1]]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 3
    assert not FindNodes(ir.Associate).visit(body)

    # Retains three nodes in the middle
    start = assignments[3]
    stop = assignments[6]
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 3
    assert not FindNodes(ir.Associate).visit(body)

    # Retains all nodes but the last, but check with ``inplace=True``
    body = MaskedTransformer(start=None, stop=assignments[-1], active=True, inplace=True).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == len(assignments) - 1
    assocs = FindNodes(ir.Associate).visit(body)
    assert len(assocs) == 1
    assert len(assocs[0].body) == len(assignments) - 1
    assert all(isinstance(n, ir.Assignment) for n in assocs[0].body)


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
    assignments = FindNodes(ir.Assignment).visit(routine.body)
    loops = FindNodes(ir.Loop).visit(routine.body)
    conditionals = FindNodes(ir.Conditional).visit(routine.body)
    assert len(assignments) == 4
    assert len(loops) == 2
    assert len(conditionals) == 2 if frontend == OMNI else 1

    # Drops the outermost loop
    start = [a for a in assignments if a.lhs == 'a']
    body = MaskedTransformer(start=start).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 4
    assert len(FindNodes(ir.Loop).visit(body)) == 1
    assert len(FindNodes(ir.Conditional).visit(body)) == len(conditionals)

    # Should produce the original version
    body = NestedMaskedTransformer(start=start).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 4
    assert len(FindNodes(ir.Loop).visit(body)) == 2
    assert len(FindNodes(ir.Conditional).visit(body)) == len(conditionals)
    assert fgen(routine.body).strip() == fgen(body).strip()

    # Should drop the first assignment
    body = NestedMaskedTransformer(start=conditionals[0]).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 3
    assert len(FindNodes(ir.Loop).visit(body)) == 2
    assert len(FindNodes(ir.Conditional).visit(body)) == len(conditionals)

    # Should leave no more than a single assignment
    start = [a for a in assignments if a.lhs == 'c']
    stop = [l for l in loops if l.variable == 'j']
    body = MaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert fgen(start).strip() == fgen(body).strip()

    # Should leave a single assignment with the hierarchy of nested sections
    # in the else-if branch
    body = NestedMaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 1
    assert len(FindNodes(ir.Loop).visit(body)) == 1
    assert len(FindNodes(ir.Conditional).visit(body)) == 1

    # Should leave no more than a single assignment
    start = [a for a in assignments if a.lhs == 'd']
    body = MaskedTransformer(start=start, stop=start).visit(routine.body)
    assert fgen(start).strip() == fgen(body).strip()

    # Should leave a single assignment with the hierarchy of nested sections
    # in the else branch
    body = NestedMaskedTransformer(start=start, stop=start).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 1
    assert len(FindNodes(ir.Loop).visit(body)) == 2
    assert len(FindNodes(ir.Conditional).visit(body)) == 0

    # Should produce the original body
    start = [a for a in assignments if a.lhs in ('a', 'd')]
    body = NestedMaskedTransformer(start=start).visit(routine.body)
    assert fgen(routine.body).strip() == fgen(body).strip()

    # Should leave a single assignment with the hierarchy of nested sections
    # in the else branch
    body = NestedMaskedTransformer(start=start, require_all_start=True).visit(routine.body)
    assert [a.lhs == 'd' for a in FindNodes(ir.Assignment).visit(body)] == [True]
    assert len(FindNodes(ir.Loop).visit(body)) == 2
    assert len(FindNodes(ir.Conditional).visit(body)) == 0

    # Drops everything
    stop = [a for a in assignments if a.lhs == 'a']
    body = NestedMaskedTransformer(start=start, stop=stop, greedy_stop=True).visit(routine.body)
    assert not body

    # Should drop the else-if branch
    start = [a for a in assignments if a.lhs in ('b', 'd')]
    stop = [a for a in assignments if a.lhs == 'c']
    body = NestedMaskedTransformer(start=start, stop=stop).visit(routine.body)
    assert len(FindNodes(ir.Assignment).visit(body)) == 2
    assert len(FindNodes(ir.Loop).visit(body)) == 2
    assert len(FindNodes(ir.Conditional).visit(body)) == 1

    # Should drop everything buth the if branch
    body = NestedMaskedTransformer(start=start, stop=stop, greedy_stop=True).visit(routine.body)
    assert [a.lhs == 'b' for a in FindNodes(ir.Assignment).visit(body)] == [True]
    assert len(FindNodes(ir.Loop).visit(body)) == 1
    assert len(FindNodes(ir.Conditional).visit(body)) == 1


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

    loops = FindNodes(ir.Loop).visit(kernel.body)
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
    assert len(FindNodes(ir.Loop).visit(kernel.body)) == {
        'body': 0, # All loops replaced by the body
        'self': 2, 'self_tuple': 2,  # Loop replaced by itself
        'duplicate': 4  # Loops duplicated
    }[replacement]
