# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
A selection of tests for symbolic computations using expression tree nodes.
"""
import operator as op
import pytest

import pymbolic.primitives as pmbl
from loki import parse_fparser_expression, Scope, HAVE_FP
from loki.expression import symbols as sym, simplify, Simplification, symbolic_op


@pytest.mark.parametrize('a, b, lt, eq', [
    (sym.Literal(1),      sym.Literal(0),   False, False),
    (sym.Literal(0),      sym.Literal(1),   True,  False),
    (sym.Literal(1),      sym.Literal(1),   False, True),
    (sym.Literal(-3),     sym.Literal(-1),  True,  False),
    (sym.Literal(-3),     sym.Literal(3),   True,  False),
    (sym.Literal(-2),     sym.Literal(-4),  False, False),
    (sym.Literal(3.0),    sym.Literal(5.0), True,  False),
    (sym.Literal(7.0),    sym.Literal(2.0), False, False),
    (sym.Literal(4.0),    sym.Literal(4.0), False, True),
    (sym.Literal(3.9999), sym.Literal(4.0), True,  False),
    (sym.Literal(2),      sym.Literal(4.0), None,  False),
    (sym.Literal(5.0),    sym.Literal(8),   None,  False),
    (sym.Literal(3.0),    sym.Literal(3),   None,  False),
    (sym.Literal(3),      sym.Literal(3.0), None,  False),
    (sym.Literal(2),      5,                True,  False),
    (sym.Literal(5),      2,                False, False),
    (sym.Literal(1),      3.1,              None,  False),
    (sym.Literal(4),      2.2,              None,  False),
    (3,                   sym.Literal(4),   True,  False),
    (4,                   sym.Literal(3),   False, False),
    (3.1,                 sym.Literal(1),   None,  False),
    (2.2,                 sym.Literal(4),   None,  False),
    (sym.Literal(9.1),    13,               True,  False),
    (sym.Literal(7.4),    9.1,              True,  False),
    (sym.Literal(8.2),    4,                False, False),
    (sym.Literal(6.5),    3.7,              False, False),
    (13,                  sym.Literal(9.1), False, False),
    (9.1,                 sym.Literal(7.4), False, False),
    (4,                   sym.Literal(8.2), True,  False),
    (3.7,                 sym.Literal(6.5), True,  False),
    (sym.Literal(3.1),    3.1,              False, True),
    (3.1,                 sym.Literal(3.1), False, True),
])
def test_symbolic_literal_comparison(a, b, lt, eq):
    """
    Test correct evaluation of a<b, a<=b, a>b, a>=b, a==b for literals
    """
    if lt is None:
        with pytest.raises(TypeError):
            _ = (a < b)
        with pytest.raises(TypeError):
            _ = (a <= b)
        with pytest.raises(TypeError):
            _ = (a > b)
        with pytest.raises(TypeError):
            _ = (a >= b)
    else:
        assert (a < b) is lt
        assert (a <= b) is (lt or eq)
        assert (a > b) is not (lt or eq)
        assert (a >= b) is (not lt)
    assert (a == b) is eq


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
@pytest.mark.parametrize('a, _op, b, ref', [
    ('1', op.eq, '2', False),
    ('1', op.ne, '2', True),
    ('1', op.lt, '2', True),
    ('1', op.le, '2', True),
    ('1', op.gt, '2', False),
    ('1', op.ge, '2', False),
    ('a', op.eq, 'a', True),
    ('a', op.ne, 'a', False),
    ('a', op.lt, 'a', False),
    ('a', op.le, 'a', True),
    ('a', op.gt, 'a', False),
    ('a', op.ge, 'a', True),
    ('a', op.eq, 'a+1', False),
    ('a', op.ne, 'a+1', True),
    ('a', op.lt, 'a+1', True),
    ('a', op.le, 'a+1', True),
    ('a', op.gt, 'a+1', False),
    ('a', op.ge, 'a+1', False),
    ('a', op.sub, 'a+1', '-1'),
])
def test_symbolic_op(a, _op, b, ref):
    """
    Test correct evaluation of operators on expressions.
    """
    scope = Scope()
    expr_a = parse_fparser_expression(a, scope)
    expr_b = parse_fparser_expression(b, scope)
    ret = symbolic_op(expr_a, _op, expr_b)
    if isinstance(ret, pmbl.Expression):
        assert simplify(ret) == ref
    else:
        assert ret == ref


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
@pytest.mark.parametrize('source, ref', [
    ('1 + 1', '1 + 1'),
    ('1 + (2 + (3 + (4 + 5) + 6)) + 7', '1 + 2 + 3 + 4 + 5 + 6 + 7'),
    ('1 - (2 + (3 + (4 - 5) - 6)) - 7', '1 - 2 - 3 - 4 + 5 + 6 - 7'),
    ('1 - (-1 - (-1 - (-1 - (-1 - 1) - 1) - 1) - 1) - 1', '1 + 1 - 1 + 1 - 1 - 1 + 1 - 1 + 1 - 1'),
    ('a + (b - (c + d))', 'a + b - c - d'),
    ('5 * (4 + 3 * (2 + 1) )', '5*4 + 5*3*2 + 5*3'),
    ('5 + a * (3 - b * (2 + c) / 7) * 5 - 4', '5 + a*3*5 - a*b*2*5 / 7 - a*b*c*5 / 7 - 4'),
    ('(((0)))', '0'),
    ('0*0', '0'),
    ('1*1', '1'),
    ('(-1)*(-1)', '1'),
    ('1*(1*(1*1))', '1'),
    ('(6 + 4) / 3', '6 / 3 + 4 / 3'),
    ('6 * (5/3) * 2', '6*5*2 / 3'),
    ('(3 + 4) * (5/3) * 2', '3*5*2 / 3 + 4*5*2 / 3'),
    ('a * (b + c/d) * e', 'a*b*e + a*c*e / d'),
])
def test_simplify_flattened(source, ref):
    scope = Scope()
    expr = parse_fparser_expression(source, scope)
    expr = simplify(expr, enabled_simplifications=Simplification.Flatten)
    assert str(expr) == ref


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
@pytest.mark.parametrize('source, ref', [
    ('1 + 1', '2'),
    ('2 - 1', '1'),
    ('1 - 1', '0'),
    ('0 + 1 - 0 - 1 + 0', '0'),
    ('1 + 1 + 1 + 1', '4'),
    ('1 + 1 - 1 + 1 - 1 + 1', '2'),
    ('(1 + 1) - (1 + 1)', '0'),
    ('5*4', '20'),
    ('-3*7', '-21'),
    ('3*7*0*10', '0'),
    ('1/1', '1'),
    ('0/1', '0'),
    ('4/2', '2'),
    ('-1/1', '-1'),
    ('7/(-1)', '-7'),
    ('10*a/5', '2*a'),
    ('2*(-2)/(-4)', '1'),
    ('(-8)/4', '-2'),
    ('(5 + 3) * a - 8 * a / 2 + a * ((7 - 1) / 3)', '8*a - 4*a + 2*a')
])
def test_simplify_integer_arithmetic(source, ref):
    scope = Scope()
    expr = parse_fparser_expression(source, scope)
    expr = simplify(expr, enabled_simplifications=Simplification.IntegerArithmetic)
    assert str(expr) == ref


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
@pytest.mark.parametrize('source, ref', [
    ('a + a + a', '3*a'),
    ('2*a + 1*a + a*3', '6*a'),
    ('(a + a)*(b + b)', '2*a*2*b'),
    ('(a + b) + a + b', 'a + b + a + b'),  # We lose the parenthesis but it does not reduce without flattening
    ('a - a', '0'),
    ('(a + a)*(b - b)', '2*a*0'),
    ('3*a + (-2)*a', 'a'),
    ('3*a - 2*a', 'a'),
    ('1*a + 0*a', 'a'),
    ('1*a*b + 0*a*b', '1*a*b + 0*a*b'),  # Note that this does not reduce without flattening
    ('5*5 + 3*3', '34'),
    ('5 + (-1)', '4'),
    ('(5 + 3) * a - 8 * a / 2 + a * ((7 - 1) / 3)', '8*a - 8*a / 2 + 6 / 3*a')
])
def test_simplify_collect_coefficients(source, ref):
    scope = Scope()
    expr = parse_fparser_expression(source, scope)
    expr = simplify(expr, enabled_simplifications=Simplification.CollectCoefficients)
    assert str(expr) == ref


@pytest.mark.skipif(not HAVE_FP, reason='Fparser not available')
@pytest.mark.parametrize('source, ref', [
    ('5 * (4 + 3 * (2 + 1) )', '65'),
    ('1 - (-1 - (-1 - (-1 - (-1 - 1) - 1) - 1) - 1) - 1', '0'),
    ('5 + a * (3 - b * (2 + c)) * 5 - 4', '1 + 15*a - 10*a*b - 5*a*b*c'),
    ('(a + b) + a + b', '2*a + 2*b'),
    ('(a+b)*(a+b)', 'a*a + 2*a*b + b*b'),
    ('(a-b)*(a-b)', 'a*a - 2*a*b + b*b'),
    ('-(a+b)*(a-b)', '-a*a + b*b'),
    ('a*a + b*(a - b) - a*(b + a) + b*b', '0'),
    ('0*(a + b - a - b)', '0'),
    ('(a + b) * c - c*a - c*b + 1', '1'),
    ('1*a*b + 0*a*b', 'a*b'),
    ('n+(((-1)*1)*n)', '0'),
    ('5 + a * (3 - b * (2 + c) / 7) * 5 - 4', '1 + 15*a - 10*a*b / 7 - 5*a*b*c / 7'),
    ('(5 + 3) * a - 8 * a / 2 + a * ((7 - 1) / 3)', '6*a')
])
def test_simplify(source,ref):
    scope = Scope()
    expr = parse_fparser_expression(source, scope)
    expr = simplify(expr)
    assert str(expr) == ref
