import pytest

from conftest import parse_expression
from loki.expression import simplify, Simplification


@pytest.mark.parametrize('source, ref', [
    ('1 + 1', '1 + 1'),
    ('1 + (2 + (3 + (4 + 5) + 6)) + 7', '1 + 2 + 3 + 4 + 5 + 6 + 7'),
    ('1 - (2 + (3 + (4 - 5) - 6)) - 7', '1 - 2 - 3 - 4 + 5 + 6 - 7'),
    ('1 - (-1 - (-1 - (-1 - (-1 - 1) - 1) - 1) - 1) - 1', '1 + 1 - 1 + 1 - 1 - 1 + 1 - 1 + 1 - 1'),
    ('a + (b - (c + d))', 'a + b - c - d'),
    ('5 * (4 + 3 * (2 + 1) )', '5*4 + 5*3*2 + 5*3'),
    ('5 + a * (3 - b * (2 + c) / 7) * 5 - 4', '5 + a*3*5 - (a*b*2*5) / 7 - (a*b*c*5) / 7 - 4'),
    ('(((0)))', '0'),
    ('0*0', '0'),
    ('1*1', '1'),
    ('(-1)*(-1)', '1'),
    ('1*(1*(1*1))', '1'),
    ('(6 + 4) / 3', '6 / 3 + 4 / 3'),
    ('6 * (5/3) * 2', '(6*5*2) / 3'),
    ('(3 + 4) * (5/3) * 2', '(3*5*2) / 3 + (4*5*2) / 3'),
    ('a * (b + c/d) * e', 'a*b*e + (a*c*e) / d'),
])
def test_simplify_flattened(source, ref):
    expr, _ = parse_expression(source)
    expr = simplify(expr, enabled_simplifications=Simplification.Flatten)
    assert str(expr) == ref


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
    ('7/-1', '-7'),
    ('10*a/5', '2*a'),
    ('2*(-2)/(-4)', '1'),
    ('(-8)/4', '-2'),
    ('(5 + 3) * a - 8 * a / 2 + a * ((7 - 1) / 3)', '8*a - 4*a + 2*a')
])
def test_simplify_integer_arithmetic(source, ref):
    expr, _ = parse_expression(source)
    expr = simplify(expr, enabled_simplifications=Simplification.IntegerArithmetic)
    assert str(expr) == ref


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
    ('(5 + 3) * a - 8 * a / 2 + a * ((7 - 1) / 3)', '8*a - (8*a) / 2 + (6 / 3)*a')
])
def test_simplify_collect_coefficients(source, ref):
    expr, _ = parse_expression(source)
    expr = simplify(expr, enabled_simplifications=Simplification.CollectCoefficients)
    assert str(expr) == ref


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
    ('5 + a * (3 - b * (2 + c) / 7) * 5 - 4', '1 + 15*a - (10*a*b) / 7 - (5*a*b*c) / 7'),
    ('(5 + 3) * a - 8 * a / 2 + a * ((7 - 1) / 3)', '6*a')
])
def test_simplify(source,ref):
    expr, _ = parse_expression(source)
    expr = simplify(expr)
    assert str(expr) == ref
