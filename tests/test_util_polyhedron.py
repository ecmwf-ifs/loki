# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from loki import Scope
from loki.frontend.fparser import parse_fparser_expression, HAVE_FP
from loki.analyse.util_polyhedron import Polyhedron
from loki.expression import symbols as sym


@pytest.fixture(scope="module", name="here")
def fixture_here():
    return Path(__file__).parent


# Polyhedron functionality relies on FParser's expression parsing
pytestmark = pytest.mark.skipif(not HAVE_FP, reason="Fparser not available")


@pytest.mark.parametrize(
    "variables, lbounds, ubounds, A, b, variable_names",
    [
        # do i=0,5: do j=i,7: ...
        (
            ["i", "j"],
            ["0", "i"],
            ["5", "7"],
            [[-1, 0], [1, 0], [1, -1], [0, 1]],
            [0, 5, 0, 7],
            ["i", "j"],
        ),
        # do i=1,n: do j=0,2*i+1: do k=a,b: ...
        (
            ["i", "j", "k"],
            ["1", "0", "a"],
            ["n", "2*i+1", "b"],
            [
                [-1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, -1],
                [0, -1, 0, 0, 0, 0],
                [-2, 1, 0, 0, 0, 0],
                [0, 0, -1, 1, 0, 0],
                [0, 0, 1, 0, -1, 0],
            ],
            [-1, 0, 0, 1, 0, 0],
            ["i", "j", "k", "a", "b", "n"],
        ),
        # do jk=1,klev: ...
        (["jk"], ["1"], ["klev"], [[-1, 0], [1, -1]], [-1, 0], ["jk", "klev"]),
        # do JK=1,klev-1: ...
        (["JK"], ["1"], ["klev - 1"], [[-1, 0], [1, -1]], [-1, -1], ["jk", "klev"]),
        # do jk=ncldtop,klev: ...
        (
            ["jk"],
            ["ncldtop"],
            ["klev"],
            [[-1, 0, 1], [1, -1, 0]],
            [0, 0],
            ["jk", "klev", "ncldtop"],
        ),
        # do jk=1,KLEV+1: ...
        (["jk"], ["1"], ["KLEV+1"], [[-1, 0], [1, -1]], [-1, 1], ["jk", "klev"]),
    ],
)
def test_polyhedron_from_loop_ranges(variables, lbounds, ubounds, A, b, variable_names):
    """
    Test converting loop ranges to polyedron representation of iteration space.
    """
    scope = Scope()
    loop_variables = [parse_fparser_expression(expr, scope) for expr in variables]
    loop_lbounds = [parse_fparser_expression(expr, scope) for expr in lbounds]
    loop_ubounds = [parse_fparser_expression(expr, scope) for expr in ubounds]
    loop_ranges = [sym.LoopRange((l, u)) for l, u in zip(loop_lbounds, loop_ubounds)]
    p = Polyhedron.from_loop_ranges(loop_variables, loop_ranges)
    assert np.all(p.A == np.array(A, dtype=np.dtype(int)))
    assert np.all(p.b == np.array(b, dtype=np.dtype(int)))
    assert p.variables == variable_names


def test_polyhedron_from_loop_ranges_failures():
    """
    Test known limitation of the conversion from loop ranges to polyhedron.
    """
    # m*n is non-affine and thus can't be represented
    scope = Scope()
    loop_variable = parse_fparser_expression("i", scope)
    lower_bound = parse_fparser_expression("1", scope)
    upper_bound = parse_fparser_expression("m * n", scope)
    loop_range = sym.LoopRange((lower_bound, upper_bound))
    with pytest.raises(ValueError):
        _ = Polyhedron.from_loop_ranges([loop_variable], [loop_range])

    # no functionality to flatten exponentials, yet
    upper_bound = parse_fparser_expression("5**2", scope)
    loop_range = sym.LoopRange((lower_bound, upper_bound))
    with pytest.raises(ValueError):
        _ = Polyhedron.from_loop_ranges([loop_variable], [loop_range])


@pytest.mark.parametrize(
    "A, b, variable_names, lower_bounds, upper_bounds",
    [
        # do i=1,n: ...
        ([[-1, 0], [1, -1]], [-1, 0], ["i", "n"], [["1"], ["i"]], [["n"], []]),
        # do i=1,10: ...
        ([[-1], [1]], [-1, 10], ["i"], [["1"]], [["10"]]),
        # do i=0,5: do j=i,7: ...
        (
            [[-1, 0], [1, 0], [1, -1], [0, 1]],
            [0, 5, 0, 7],
            ["i", "j"],
            [["0"], ["i"]],
            [["5", "j"], ["7"]],
        ),
        # do i=1,n: do j=0,2*i+1: do k=a,b: ...
        (
            [
                [-1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, -1],
                [0, -1, 0, 0, 0, 0],
                [-2, 1, 0, 0, 0, 0],
                [0, 0, -1, 1, 0, 0],
                [0, 0, 1, 0, -1, 0],
            ],
            [-1, 0, 0, 1, 0, 0],
            ["i", "j", "k", "a", "b", "n"],  # variable names
            [["1", "-1 / 2 + j / 2"], ["0"], ["a"], [], ["k"], ["i"]],  # lower bounds
            [["n"], ["1 + 2*i"], ["b"], ["k"], [], []],
        ),  # upper bounds
    ],
)
def test_polyhedron_bounds(A, b, variable_names, lower_bounds, upper_bounds):
    """
    Test the production of lower and upper bounds.
    """
    scope = Scope()
    variables = [parse_fparser_expression(v, scope) for v in variable_names]
    p = Polyhedron(A, b, variables)
    for var, ref_bounds in zip(variables, lower_bounds):
        lbounds = p.lower_bounds(var)
        assert len(lbounds) == len(ref_bounds)
        assert all(str(b1) == b2 for b1, b2 in zip(lbounds, ref_bounds))
    for var, ref_bounds in zip(variables, upper_bounds):
        ubounds = p.upper_bounds(var)
        assert len(ubounds) == len(ref_bounds)
        assert all(str(b1) == b2 for b1, b2 in zip(ubounds, ref_bounds))


@pytest.mark.parametrize(
    "polyhedron,is_empty,will_fail",
    [
        # totaly empty polyhedron
        (Polyhedron.from_nested_loops([]), True, False),
        # full matrix --> non trivial problem
        (Polyhedron([[1]], [1]), None, True),
        # empty matrix, full and fullfiled b --> non empty polyhedron
        (Polyhedron([[]], [1]), False, False),
        # empty matrix, full b but not fullfiled b --> empty polyhedron
        (Polyhedron([[]], [-1]), True, False),
    ],
)
def test_check_empty_polyhedron(polyhedron, is_empty, will_fail):
    if will_fail:
        with pytest.raises(RuntimeError):
            _ = polyhedron.is_empty()
    else:
        assert polyhedron.is_empty() == is_empty
