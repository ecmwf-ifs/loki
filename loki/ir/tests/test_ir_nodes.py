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


def test_multi_conditional(scope, one, i, n, a_i):
    """
    Test nested chains of constructors of :any:`Conditional` to form
    multi-conditional.
    """
    multicond = ir.Conditional(
        condition=sym.Comparison(i, '==', sym.IntLiteral(1)),
        body=ir.Assignment(lhs=a_i, rhs=sym.Literal(1.0)),
        else_body=ir.Assignment(lhs=a_i, rhs=sym.Literal(42.0))
    )
    for idx in range(2, 4):
        multicond = ir.Conditional(
            condition=sym.Comparison(i, '==', sym.IntLiteral(idx)),
            body=ir.Assignment(lhs=a_i, rhs=sym.Literal(float(idx))),
            else_body=multicond, has_elseif=True
        )

    # Check that we can recover all bodies from a nested else-if construct
    else_bodies = multicond.else_bodies
    assert len(else_bodies) == 3
    assert all(isinstance(b, tuple) for b in else_bodies)
    assert isinstance(else_bodies[0][0], ir.Assignment)
    assert else_bodies[0][0].lhs == 'a(i)' and else_bodies[0][0].rhs == '2.0'
    assert isinstance(else_bodies[1][0], ir.Assignment)
    assert else_bodies[1][0].lhs == 'a(i)' and else_bodies[1][0].rhs == '1.0'
    assert isinstance(else_bodies[2][0], ir.Assignment)
    assert else_bodies[2][0].lhs == 'a(i)' and else_bodies[2][0].rhs == '42.0'

    # Not try without the final else
    multicond = ir.Conditional(
        condition=sym.Comparison(i, '==', sym.IntLiteral(1)),
        body=ir.Assignment(lhs=a_i, rhs=sym.Literal(1.0)),
    )
    for idx in range(2, 4):
        multicond = ir.Conditional(
            condition=sym.Comparison(i, '==', sym.IntLiteral(idx)),
            body=ir.Assignment(lhs=a_i, rhs=sym.Literal(float(idx))),
            else_body=multicond, has_elseif=True
        )
    else_bodies = multicond.else_bodies
    assert len(else_bodies) == 2
    assert all(isinstance(b, tuple) for b in else_bodies)
    assert isinstance(else_bodies[0][0], ir.Assignment)
    assert else_bodies[0][0].lhs == 'a(i)' and else_bodies[0][0].rhs == '2.0'
    assert isinstance(else_bodies[1][0], ir.Assignment)
    assert else_bodies[1][0].lhs == 'a(i)' and else_bodies[1][0].rhs == '1.0'


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

    sec = ir.Section((assign, assign))
    assert sec.body == (assign, assign)

    # Test auto-casting of the body to tuple
    sec = ir.Section(body=assign)
    assert sec.body == (assign,)
    sec = ir.Section(body=( (assign,), ))
    assert sec.body == (assign,)
    sec = ir.Section(body=( assign, (assign,), assign, None))
    assert sec.body == (assign, assign, assign)
    sec = ir.Section((assign, (func,), assign, None))
    assert sec.body == (assign, func, assign)

    # Test prepend/insert/append additions
    sec = ir.Section(body=func)
    assert sec.body == (func,)
    sec.prepend(assign)
    assert sec.body == (assign, func)
    sec.append((assign, assign))
    assert sec.body == (assign, func, assign, assign)
    sec.insert(pos=3, node=func)
    assert sec.body == (assign, func, assign, func, assign)


def test_callstatement(scope, one, i, n, a_i):
    """ Test constructor of :any:`CallStatement` nodes. """

    cname = sym.ProcedureSymbol(name='test', scope=scope)
    call = ir.CallStatement(
        name=cname, arguments=(n, a_i), kwarguments=(('i', i), ('j', one))
    )
    assert isinstance(call.name, Expression)
    assert isinstance(call.arguments, tuple)
    assert all(isinstance(e, Expression) for e in call.arguments)
    assert isinstance(call.kwarguments, tuple)
    assert all(isinstance(e, tuple) for e in call.kwarguments)
    assert all(
        isinstance(k, str) and isinstance(v, Expression)
        for k, v in call.kwarguments
    )

    # Ensure "frozen" status of node objects
    with pytest.raises(FrozenInstanceError) as error:
        call.name = sym.ProcedureSymbol('dave', scope=scope)
    with pytest.raises(FrozenInstanceError) as error:
        call.arguments = (a_i, n, one)
    with pytest.raises(FrozenInstanceError) as error:
        call.kwarguments = (('i', one), ('j', i))

    # Test auto-casting of the body to tuple
    call = ir.CallStatement(name=cname, arguments=[a_i, one])
    assert call.arguments == (a_i, one) and call.kwarguments == ()
    call = ir.CallStatement(name=cname, arguments=None)
    assert call.arguments == () and call.kwarguments == ()
    call = ir.CallStatement(name=cname, kwarguments=[('i', i), ('j', one)])
    assert call.arguments == () and call.kwarguments == (('i', i), ('j', one))
    call = ir.CallStatement(name=cname, kwarguments=None)
    assert call.arguments == () and call.kwarguments == ()

    # Test errors for wrong contructor usage
    with pytest.raises(ValidationError) as error:
        ir.CallStatement(name='a', arguments=(sym.Literal(42.0),))
    with pytest.raises(ValidationError) as error:
        ir.CallStatement(name=cname, arguments=('a',))
    with pytest.raises(ValidationError) as error:
        ir.Assignment(
            name=cname, arguments=(sym.Literal(42.0),), kwarguments=('i', 'i')
        )

    # TODO: Test pragmas, active and chevron


def test_associate(scope, a_i):
    """
    Test constructors and scoping bahviour of :any:`Associate`.
    """
    b = sym.Scalar(name='b', scope=scope)
    b_a = sym.Array(name='a', parent=b, scope=scope)
    a = sym.Array(name='a', scope=scope)
    assign = ir.Assignment(lhs=a_i, rhs=sym.Literal(42.0))
    assign2 = ir.Assignment(lhs=a_i.clone(parent=b), rhs=sym.Literal(66.6))

    assoc = ir.Associate(associations=((b_a, a),), body=(assign, assign2), parent=scope)
    assert isinstance(assoc.associations, tuple)
    assert all(isinstance(n, tuple) and len(n) == 2 for n in assoc.associations)
    assert isinstance(assoc.body, tuple)
    assert all(isinstance(n, ir.Node) for n in assoc.body)

    # TODO: Check constructor failures, auto-casting and frozen status

    # Check provided symbol maps
    assert 'B%a' in assoc.association_map and assoc.association_map['B%a'] is a
    assert b_a in assoc.association_map and assoc.association_map[b_a] is a
    assert 'a' in assoc.inverse_map and assoc.inverse_map['a'] is b_a
    assert a in assoc.inverse_map and assoc.inverse_map[a] is b_a

    # Check rescoping facility
    assert assign.lhs.scope is scope
    assert assign2.lhs.scope is scope
    assoc.rescope_symbols()
    assert assign.lhs.scope is assoc
    assert assign2.lhs.scope is scope
