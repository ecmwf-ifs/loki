"""
A selection of tests for symbolic computations using expression tree nodes.
"""
import pytest
from loki.expression import symbols as sym


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
def test_symbolics_comparison(a, b, lt, eq):
    """
    Test for correct evaluation of a<b, a<=b, a>b, a>=b, a==b.
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
