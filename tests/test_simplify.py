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
    ('5 + a * (3 - b * (2 + c) / 7) * 5 - 4', '5 + a*3*5 - a*((b*2 + b*c) / 7)*5 - 4'),
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
    ('3*7*0*10', '0')
])
def test_simplify_integer_arithmetic(source, ref):
    expr, _ = parse_expression(source)
    expr = simplify(expr, enabled_simplifications=Simplification.IntegerArithmetic)
    assert str(expr) == ref


@pytest.mark.parametrize('source, ref', [
    ('5 * (4 + 3 * (2 + 1) )', '65'),
    ('1 - (-1 - (-1 - (-1 - (-1 - 1) - 1) - 1) - 1) - 1', '0'),
    ('5 + a * (3 - b * (2 + c)) * 5 - 4', '1 + 15*a - 10*a*b - 5*a*b*c'),
])
def test_simplify(source,ref):
    expr, _ = parse_expression(source)
    expr = simplify(expr)
    assert str(expr) == ref
