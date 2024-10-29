# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Scope
from loki.expression import symbols as sym, parse_expr
from loki.expression.mappers import (
    ExpressionRetriever, LokiIdentityMapper
)
from loki.frontend import available_frontends
from loki.ir import nodes as ir, FindNodes


@pytest.mark.parametrize('frontend', available_frontends())
def test_expression_retriever(frontend):
    """ Test for :any:`ExpressionRetriever` (a :any:`LokiWalkMapper`) """

    fcode = """
subroutine test_expr_retriever(n, a, b, c)
  integer, intent(inout) :: n, a, b(n), c

  a = 5 * a + 4 * b(c) + a
end subroutine test_expr_retriever
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    expr = FindNodes(ir.Assignment).visit(routine.body)[0].rhs

    def q_symbol(n):
        return isinstance(n, sym.TypedSymbol)

    def q_array(n):
        return isinstance(n, sym.Array)

    def q_scalar(n):
        return isinstance(n, sym.Scalar)

    def q_deferred(n):
        return isinstance(n, sym.DeferredTypeSymbol)

    def q_literal(n):
        return isinstance(n, sym.IntLiteral)

    assert ExpressionRetriever(q_symbol).retrieve(expr) == ['a', 'b', 'c', 'a']
    assert ExpressionRetriever(q_array).retrieve(expr) == ['b(c)']
    assert ExpressionRetriever(q_scalar).retrieve(expr) == ['a', 'c', 'a']
    assert ExpressionRetriever(q_literal).retrieve(expr) == [5, 4]

    scope = Scope()
    expr = parse_expr('5 * a + 4 * b(c) + a', scope=scope)

    assert ExpressionRetriever(q_symbol).retrieve(expr) == ['a', 'b', 'c', 'a']
    assert ExpressionRetriever(q_array).retrieve(expr) == ['b(c)']
    # Cannot determine Scalar without declarations, so check for deferred
    assert ExpressionRetriever(q_deferred).retrieve(expr) == ['a', 'c', 'a']
    assert ExpressionRetriever(q_literal).retrieve(expr) == [5, 4]


@pytest.mark.parametrize('frontend', available_frontends())
def test_identity_mapper(frontend):
    """
    Test for :any:`LokiIdentityMapper`, in particular deep-copying
    expression nodes.
    """

    fcode = """
subroutine test_expr_retriever(n, a, b, c)
  integer, intent(inout) :: n, a, b(n), c

  a = 5 * a + 4 * b(c) + a
end subroutine test_expr_retriever
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)
    expr = FindNodes(ir.Assignment).visit(routine.body)[0].rhs

    # Run the identity mapper over the expression
    new_expr = LokiIdentityMapper()(expr)

    # Check that symbols and literals are equivalent, but distinct objects!
    get_symbols = ExpressionRetriever(lambda e: isinstance(e, sym.TypedSymbol)).retrieve
    get_literals = ExpressionRetriever(lambda e: isinstance(e, sym.IntLiteral)).retrieve

    for old, new in zip(get_symbols(expr), get_symbols(new_expr)):
        assert old == new
        assert not old is new

    for old, new in zip(get_literals(expr), get_literals(new_expr)):
        assert old == new
        assert not old is new
