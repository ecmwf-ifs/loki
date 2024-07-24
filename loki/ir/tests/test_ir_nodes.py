# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from dataclasses import FrozenInstanceError
from pymbolic.primitives import Expression
from pydantic import ValidationError

from loki.expression import symbols as sym, parse_expr
from loki.ir import nodes as ir
from loki.scope import Scope
from loki.subroutine import Subroutine


@pytest.fixture(name='scope')
def fixture_scope():
    return Scope()

@pytest.fixture(name='one')
def fixture_one():
    return sym.Literal(1)

@pytest.fixture(name='i')
def fixture_i(scope):
    return sym.Scalar('i', scope=scope)

@pytest.fixture(name='n')
def fixture_n(scope):
    return sym.Scalar('n', scope=scope)

@pytest.fixture(name='a_i')
def fixture_a_i(scope, i):
    return sym.Array('a', dimensions=(i,), scope=scope)

@pytest.fixture(name='a_n')
def fixture_a_n(scope, n):
    return sym.Array('a', dimensions=(n,), scope=scope)


def test_assignment(scope, a_i):
    """
    Test constructors of :any:`Assignment`.
    """
    assign = ir.Assignment(lhs=a_i, rhs=sym.Literal(42.0))
    assert isinstance(assign.lhs, Expression)
    assert isinstance(assign.rhs, Expression)
    assert assign.comment is None

    # Ensure "frozen" status of node objects
    with pytest.raises(FrozenInstanceError) as error:
        assign.lhs = sym.Scalar('b', scope=scope)
    with pytest.raises(FrozenInstanceError) as error:
        assign.rhs = sym.Scalar('b', scope=scope)

    # Test errors for wrong contructor usage
    with pytest.raises(ValidationError) as error:
        ir.Assignment(lhs='a', rhs=sym.Literal(42.0))
    with pytest.raises(ValidationError) as error:
        ir.Assignment(lhs=a_i, rhs='42.0 + 6.0')
    with pytest.raises(ValidationError) as error:
        ir.Assignment(lhs=a_i, rhs=sym.Literal(42.0), comment=a_i)


def test_loop(scope, one, i, n, a_i):
    """
    Test constructors of :any:`Loop`.
    """
    assign = ir.Assignment(lhs=a_i, rhs=sym.Literal(42.0))
    bounds = sym.Range((one, n))

    loop = ir.Loop(variable=i, bounds=bounds, body=(assign,))
    assert isinstance(loop.variable, Expression)
    assert isinstance(loop.bounds, Expression)
    assert isinstance(loop.body, tuple)
    assert all(isinstance(n, ir.Node) for n in loop.body)
    assert loop.children == ( i, bounds, (assign,) )

    # Ensure "frozen" status of node objects
    with pytest.raises(FrozenInstanceError) as error:
        loop.variable = sym.Scalar('j', scope=scope)
    with pytest.raises(FrozenInstanceError) as error:
        loop.bounds = sym.Range((n, sym.Scalar('k', scope=scope)))
    with pytest.raises(FrozenInstanceError) as error:
        loop.body = (assign, assign, assign)

    # Test auto-casting of the body to tuple
    loop = ir.Loop(variable=i, bounds=bounds, body=assign)
    assert loop.body == (assign,)
    loop = ir.Loop(variable=i, bounds=bounds, body=( (assign,), ))
    assert loop.body == (assign,)
    loop = ir.Loop(variable=i, bounds=bounds, body=( assign, (assign,), assign, None))
    assert loop.body == (assign, assign, assign)

    # Test errors for wrong contructor usage
    with pytest.raises(ValidationError) as error:
        ir.Loop(variable=i, bounds=bounds, body=n)
    with pytest.raises(ValidationError) as error:
        ir.Loop(variable=None, bounds=bounds, body=(assign,))
    with pytest.raises(ValidationError) as error:
        ir.Loop(variable=i, bounds=None, body=(assign,))

    # TODO: Test pragmas, names and labels


def test_conditional(scope, one, i, n, a_i):
    """
    Test constructors of :any:`Conditional`.
    """
    assign = ir.Assignment(lhs=a_i, rhs=sym.Literal(42.0))
    condition = parse_expr('i >= 2', scope=scope)

    cond = ir.Conditional(
        condition=condition, body=(assign,assign,), else_body=(assign,)
    )
    assert isinstance(cond.condition, Expression)
    assert isinstance(cond.body, tuple) and len(cond.body) == 2
    assert all(isinstance(n, ir.Node) for n in cond.body)
    assert isinstance(cond.else_body, tuple) and len(cond.else_body) == 1
    assert all(isinstance(n, ir.Node) for n in cond.else_body)
    assert cond.children == ( condition, (assign, assign), (assign,) )

    with pytest.raises(FrozenInstanceError) as error:
        cond.condition = parse_expr('k == 0', scope=scope)
    with pytest.raises(FrozenInstanceError) as error:
        cond.body = (assign, assign, assign)
    with pytest.raises(FrozenInstanceError) as error:
        cond.else_body = (assign, assign, assign)

    # Test auto-casting of the body / else_body to tuple
    cond = ir.Conditional(condition=condition, body=assign)
    assert cond.body == (assign,) and cond.else_body == ()
    cond = ir.Conditional(condition=condition, body=( (assign,), ))
    assert cond.body == (assign,) and cond.else_body == ()
    cond = ir.Conditional(condition=condition, body=( assign, (assign,), assign, None))
    assert cond.body == (assign, assign, assign) and cond.else_body == ()

    cond = ir.Conditional(condition=condition, body=(), else_body=assign)
    assert cond.body == () and cond.else_body == (assign,)
    cond = ir.Conditional(condition=condition, body=(), else_body=( (assign,), ))
    assert cond.body == () and cond.else_body == (assign,)
    cond = ir.Conditional(
        condition=condition, body=(), else_body=( assign, (assign,), assign, None)
    )
    assert cond.body == () and cond.else_body == (assign, assign, assign)

    # TODO: Test inline, name, has_elseif


def test_section(scope, one, i, n, a_n, a_i):
    """
    Test constructors and behaviour of :any:`Section` nodes.
    """
    assign = ir.Assignment(lhs=a_i, rhs=sym.Literal(42.0))
    decl = ir.VariableDeclaration(symbols=(a_n,))
    func = Subroutine(
        name='F', is_function=True, spec=(decl,), body=(assign,)
    )

    # Test constructor for nodes and subroutine objects
    sec = ir.Section(body=(assign, assign))
    assert isinstance(sec.body, tuple) and len(sec.body) == 2
    assert all(isinstance(n, ir.Node) for n in sec.body)
    with pytest.raises(FrozenInstanceError) as error:
        sec.body = (assign, assign)

    sec = ir.Section(body=(func, func))
    assert isinstance(sec.body, tuple) and len(sec.body) == 2
    assert all(isinstance(n, Scope) for n in sec.body)
    with pytest.raises(FrozenInstanceError) as error:
        sec.body = (func, func)

    # Test auto-casting of the body to tuple
    sec = ir.Section(body=assign)
    assert sec.body == (assign,)
    sec = ir.Section(body=( (assign,), ))
    assert sec.body == (assign,)
    sec = ir.Section(body=( assign, (assign,), assign, None))
    assert sec.body == (assign, assign, assign)

    # Test prepend/insert/append additions
    sec = ir.Section(body=func)
    assert sec.body == (func,)
    sec.prepend(assign)
    assert sec.body == (assign, func)
    sec.append((assign, assign))
    assert sec.body == (assign, func, assign, assign)
    sec.insert(pos=3, node=func)
    assert sec.body == (assign, func, assign, func, assign)
