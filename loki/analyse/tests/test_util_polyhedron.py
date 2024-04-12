# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from loki.ir import Loop, FindNodes
from loki.scope import Scope
from loki.sourcefile import Sourcefile
from loki.frontend.fparser import parse_fparser_expression, HAVE_FP
from loki.analyse.util_polyhedron import Polyhedron
from loki.expression import symbols as sym


@pytest.fixture(scope="module", name="here")
def fixture_here():
    return Path(__file__).parent


@pytest.fixture(scope='module', name='testdir')
def fixture_testdir(here):
    return here.parent.parent/'tests'


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


def simple_loop_extractor(start_node):
    """Find all loops in the AST and structure them depending on their nesting level"""
    start_loops = FindNodes(Loop, greedy=True).visit(start_node)
    return [FindNodes(Loop).visit(node) for node in start_loops]


def assert_equal_polyhedron(poly_A, poly_B):
    assert poly_A.variables == poly_B.variables
    assert (poly_A.A == poly_B.A).all()
    assert (poly_A.b == poly_B.b).all()


@pytest.mark.parametrize(
    "filename, loop_extractor, polyhedrons_per_subroutine",
    [
        (
            "sources/data_dependency_detection/loop_carried_dependencies.f90",
            simple_loop_extractor,
            {
                "SimpleDependency": [
                    Polyhedron(
                        [[-1, 0], [1, -1]], [-1, 0], [sym.Scalar("i"), sym.Scalar("n")]
                    ),
                ],
                "NestedDependency": [
                    Polyhedron(
                        [[-1, 0, 0], [1, 0, -1], [0, -1, 0], [-1, 1, 0]],
                        [-2, 0, -1, -1],
                        [sym.Scalar("i"), sym.Scalar("j"), sym.Scalar("n")],
                    ),
                ],
                "ConditionalDependency": [
                    Polyhedron(
                        [[-1, 0], [1, -1]],
                        [-2, 0],
                        [sym.Scalar("i"), sym.Scalar("n")],
                    ),
                ],
                "NoDependency": [
                    Polyhedron(
                        [[-1], [1]],
                        [-1, 10],
                        [sym.Scalar("i")],
                    ),
                    Polyhedron(
                        [[-1], [1]],
                        [-1, 5],
                        [sym.Scalar("i")],
                    ),
                ],
            },
        ),
        (
            "sources/data_dependency_detection/various_loops.f90",
            simple_loop_extractor,
            {
                "single_loop": [
                    Polyhedron(
                        [[-1, 0], [1, -1]],
                        [-1, 0],
                        [sym.Scalar("i"), sym.Scalar("n")],
                    ),
                ],
                "single_loop_split_access": [
                    Polyhedron(
                        [[-1, 0], [1, -1]],
                        [-1, 0],
                        [sym.Scalar("i"), sym.Scalar("nhalf")],
                    ),
                ],
                "single_loop_arithmetic_operations_for_access": [
                    Polyhedron(
                        [[-1, 0], [1, -1]],
                        [-1, 0],
                        [sym.Scalar("i"), sym.Scalar("n")],
                    ),
                ],
                "nested_loop_single_dimensions_access": [
                    Polyhedron(
                        [[-1, 0, 0], [1, 0, -1], [0, -1, 0], [0, 1, -1]],
                        [-1, 0, -1, 0],
                        [sym.Scalar("i"), sym.Scalar("j"), sym.Scalar("nhalf")],
                    ),
                ],
                "nested_loop_partially_used": [
                    Polyhedron(
                        [[-1, 0, 0], [1, 0, -1], [0, -1, 0], [0, 1, -1]],
                        [-1, 0, -1, 0],
                        [sym.Scalar("i"), sym.Scalar("j"), sym.Scalar("nfourth")],
                    ),
                ],
                "partially_used_array": [
                    Polyhedron(
                        [[-1, 0], [1, -1]],
                        [-2, 0],
                        [sym.Scalar("i"), sym.Scalar("nhalf")],
                    ),
                ],
            },
        ),
    ],
)
def test_polyhedron_construction_from_nested_loops(
    testdir, filename, loop_extractor, polyhedrons_per_subroutine
):
    source = Sourcefile.from_file(testdir / filename)

    for subroutine in source.all_subroutines:
        expected_polyhedrons = polyhedrons_per_subroutine[subroutine.name]

        list_of_loops = loop_extractor(subroutine.body)

        polyhedrons = [Polyhedron.from_nested_loops(loops) for loops in list_of_loops]

        for polyhedron, expected_polyhedron in zip(polyhedrons, expected_polyhedrons):
            assert_equal_polyhedron(polyhedron, expected_polyhedron)
