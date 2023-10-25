# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from math import gcd as math_gcd
import pytest
import numpy as np

try:
    _ = math_gcd(4,3,2)
    gcd = math_gcd
except TypeError: #Python 3.8 can only handle two arguments
    from functools import reduce
    def gcd(*args):
        return reduce(math_gcd, args)

from loki.analyse.util_linear_algebra import (
    back_substitution,
    generate_row_echelon_form,
    is_independent_system,
    yield_one_d_systems,
)


@pytest.mark.parametrize(
    "upper_triangular_square_matrix, right_hand_side, expected, divison_operation",
    [
        (
            [[2, 1, -1], [0, 0.5, 0.5], [0, 0, -1]],
            [[8], [1], [1]],
            [[2], [3], [-1]],
            lambda x, y: x / y,
        ),
        (
            [[2, 0], [0, 1]],
            [[10], [11]],
            [[5], [11]],
            lambda x, y: x // y,
        ),
    ],
)
def test_backsubstitution(
    upper_triangular_square_matrix, right_hand_side, expected, divison_operation
):
    assert np.allclose(
        back_substitution(
            np.array(upper_triangular_square_matrix),
            np.array(right_hand_side),
            divison_operation,
        ),
        np.array(expected),
    )


@pytest.mark.parametrize(
    "matrix, result",
    [
        ([[2, 0, 1], [0, 2, 0]], [[1, 0, 0.5], [0, 1, 0]]),
        ([[1, -2, 1, 0], [3, 2, 1, 5]], [[1, -2, 1, 0], [0, 1, -0.25, 0.625]]),
        ([[1, -1, -10]], [[1, -1, -10]]),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [0, 1, 2], [0, 0, 0]]),
        ([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        (
            [[2, 4, 6, 8], [1, 2, 3, 4], [3, 6, 9, 12]],
            [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
        ),
        ([[0, 0, 0], [1, 0, 2]], [[1, 0, 2], [0, 0, 0]]),
    ],
)
def test_generate_row_echelon_form(matrix, result):
    matrix = np.array(matrix, dtype=float)
    result = np.array(result, dtype=float)

    assert np.allclose(generate_row_echelon_form(matrix), result)


@pytest.mark.parametrize(
    "matrix, result",
    [
        ([[]], [[]]),
        ([[2, 0, 1], [0, 2, 0]], [[1, 0, 0], [0, 1, 0]]),
        ([[1, -2, 1, 0], [3, 2, 1, 5]], [[1, -2, 1, 0], [0, 1, -1, 0]]),
        ([[1, -1, -10]], [[1, -1, -10]]),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [0, 1, 2], [0, 0, 0]]),
        ([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        (
            [[2, 4, 6, 8], [1, 2, 3, 4], [3, 6, 9, 12]],
            [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
        ),
    ],
)
def test_enforce_integer_arithmetics_for_row_echelon_form(matrix, result):
    matrix = np.array(matrix, dtype=float)
    result = np.array(result, dtype=float)

    assert np.allclose(
        generate_row_echelon_form(matrix, division_operator=lambda x, y: x // y), result
    )


def _raise_assertion_error(A):
    raise ValueError()


def _require_gcd_condition(A):
    """Check that gcd condition of linear Diophantine equation is satisfied"""
    if A[0, -1] % gcd(*A[0, :-1]) != 0:
        raise ValueError()


@pytest.mark.parametrize(
    "matrix, condition, result",
    [
        ([[1, 2, 3], [4, 5, 6]], _raise_assertion_error, None),
        (
            [[2, 0, 0, -2, -20], [0, 2, -2, 0, -22]],
            _require_gcd_condition,
            [[1, 0, 0, -1, -10], [0, 1, -1, 0, -11]],
        ),
        ([[2, 0, 0, -2, -20], [0, 2, -2, 0, -21]], _require_gcd_condition, None),
    ],
)
def test_require_conditions(matrix, condition, result):
    matrix = np.array(matrix)

    if result is None:
        with pytest.raises(ValueError):
            _ = generate_row_echelon_form(matrix, conditional_check=condition)
    else:
        result = np.array(result)
        assert np.allclose(
            generate_row_echelon_form(matrix, conditional_check=condition), result
        )


@pytest.mark.parametrize(
    "matrix, expected_result",
    [
        (np.array([[1, 0], [0, 1], [0, 0]]), True),
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), True),
        (np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]]), False),
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), True),
        (np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), True),
    ],
)
def test_is_independent_system(matrix, expected_result):
    assert is_independent_system(matrix) == expected_result


@pytest.mark.parametrize(
    "matrix, rhs, list_of_lhs_column, list_of_rhs_column",
    [
        (
            np.array([[1, 0], [0, 1], [0, 0]]),
            np.array([[1], [2], [0]]),
            [[1], [1]],
            [[1], [2]],
        ),
        (
            np.array([[1, 0], [0, 1], [0, 0]]),
            np.array([[1], [2], [1]]),
            [[0], [1], [1]],
            [[1], [1], [2]],
        ),
        (
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[1], [2], [3]]),
            [[0]],
            [[1, 2, 3]],
        ),
        (  # will even split non independent systems, call is_independent_system before
            np.array([[2, 1], [1, 3]]),
            np.array([[3], [4]]),
            [[2], [1]],
            [[3], [4]],
        ),
    ],
)
def test_yield_one_d_systems(matrix, rhs, list_of_lhs_column, list_of_rhs_column):
    for index, (A, b) in enumerate(yield_one_d_systems(matrix, rhs)):
        assert np.allclose(A, list_of_lhs_column[index])
        assert np.allclose(b, list_of_rhs_column[index])
